# Edited from Llama 3: https://github.com/meta-llama/llama3/blob/main/llama/model.py

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048
    head_dim: int = -1
    q_rank: int = 12
    rank: int = 2
    using_groupnorm: bool = False

class T6GroupNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)



class TPA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        self.n_heads = args.n_heads
        self.head_dim = args.head_dim if args.head_dim > 0 else args.dim // args.n_heads
        # maybe different from args.dim // args.n_heads

        self.n_head = args.n_heads
        self.q_rank = args.q_rank
        self.rank = args.rank
        self.dim = args.dim
        
        self.using_groupnorm = args.using_groupnorm
        
        self.W_A_q = nn.Linear(args.dim, self.n_head * self.q_rank, bias=False)
        self.W_A_k = nn.Linear(args.dim, self.n_head * self.rank, bias=False)
        self.W_A_v = nn.Linear(args.dim, self.n_head * self.rank, bias=False)

        # Define B projection parameters for Q, K, V
        self.W_B_q = nn.Linear(args.dim, self.q_rank * self.head_dim, bias=False)
        self.W_B_k = nn.Linear(args.dim, self.rank * self.head_dim, bias=False)
        self.W_B_v = nn.Linear(args.dim, self.rank * self.head_dim, bias=False)
        
        self.cache_kA = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_heads, self.rank,)).cuda()
        self.cache_vA = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_heads, self.rank,)).cuda()
        self.cache_kB = torch.zeros((args.max_batch_size, args.max_seq_len, self.rank, self.head_dim,)).cuda()
        self.cache_vB = torch.zeros((args.max_batch_size, args.max_seq_len, self.rank, self.head_dim,)).cuda()
        self.wo = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        self.wo.weight.data.zero_()   
             
        self.reset_parameters()

        if self.using_groupnorm:
            self.subln = T6GroupNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
    def reset_parameters(self, args):
        W_A_q_tensor = self.W_A_q.weight.view(self.dim, self.n_head, self.q_rank)
        W_A_k_tensor = self.W_A_k.weight.view(self.dim, self.n_head, self.rank)
        W_A_v_tensor = self.W_A_v.weight.view(self.dim, self.n_head, self.rank)
        nn.init.xavier_uniform_(W_A_q_tensor)
        nn.init.xavier_uniform_(W_A_k_tensor)
        nn.init.xavier_uniform_(W_A_v_tensor)
        self.W_A_q.weight.data = W_A_q_tensor.view_as(self.W_A_q.weight)
        self.W_A_k.weight.data = W_A_k_tensor.view_as(self.W_A_k.weight)
        self.W_A_v.weight.data = W_A_v_tensor.view_as(self.W_A_v.weight)

        W_B_q_tensor = self.W_B_q.weight.view(self.dim, self.q_rank, self.head_dim)
        W_B_k_tensor = self.W_B_k.weight.view(self.dim, self.rank, self.head_dim)
        W_B_v_tensor = self.W_B_v.weight.view(self.dim, self.rank, self.head_dim)
        nn.init.xavier_uniform_(W_B_q_tensor)
        nn.init.xavier_uniform_(W_B_k_tensor)
        nn.init.xavier_uniform_(W_B_v_tensor)
        self.W_B_q.weight.data = W_B_q_tensor.view_as(self.W_B_q.weight)
        self.W_B_k.weight.data = W_B_k_tensor.view_as(self.W_B_k.weight)
        self.W_B_v.weight.data = W_B_v_tensor.view_as(self.W_B_v.weight)
        
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape

        A_q = self.W_A_q(x).view(bsz, seqlen, self.n_head, self.q_rank)
        A_k = self.W_A_k(x).view(bsz, seqlen, self.n_head, self.rank)
        A_v = self.W_A_v(x).view(bsz, seqlen, self.n_head, self.rank)

        # Compute intermediate variables B for Q, K, and V
        B_q = self.W_B_q(x).view(bsz, seqlen, self.q_rank, self.head_dim)
        B_k = self.W_B_k(x).view(bsz, seqlen, self.rank, self.head_dim)
        B_v = self.W_B_v(x).view(bsz, seqlen, self.rank, self.head_dim)

        B_q, B_k = apply_rotary_emb(B_q, B_k, freqs_cis=freqs_cis)
        
        # Cache A_k, A_v
        self.cache_kA = self.cache_kA.to(A_k)
        self.cache_vA = self.cache_vA.to(A_v)
        
        self.cache_kA[:bsz, start_pos : start_pos + seqlen] = A_k
        self.cache_vA[:bsz, start_pos : start_pos + seqlen] = A_v
        
        A_k = self.cache_kA[:bsz, : start_pos + seqlen]
        A_v = self.cache_vA[:bsz, : start_pos + seqlen]
        
        # Cache B_k, B_v
        
        self.cache_kB = self.cache_kB.to(B_k)
        self.cache_vB = self.cache_vB.to(B_v)
        
        self.cache_kB[:bsz, start_pos : start_pos + seqlen] = B_k
        self.cache_vB[:bsz, start_pos : start_pos + seqlen] = B_v
        
        B_k = self.cache_kB[:bsz, : start_pos + seqlen]
        B_v = self.cache_vB[:bsz, : start_pos + seqlen]
        
        # Reshape A_q, A_k, A_v
        A_q = A_q.view(bsz * seqlen, self.n_head, self.q_rank)
        A_k = A_k.view(bsz * seqlen, self.n_head, self.rank)
        A_v = A_v.view(bsz * seqlen, self.n_head, self.rank)

        # Reshape B_k, B_v  
        B_q = B_q.view(bsz * seqlen, self.q_rank, self.head_dim)
        B_k = B_k.view(bsz * seqlen, self.rank, self.head_dim)
        B_v = B_v.view(bsz * seqlen, self.rank, self.head_dim)
        
        q = torch.bmm(A_q, B_q).div_(self.q_rank).view(bsz, seqlen, self.n_head, self.head_dim)
        k = torch.bmm(A_k, B_k).div_(self.rank).view(bsz, seqlen, self.n_head, self.head_dim)
        v = torch.bmm(A_v, B_v).div_(self.rank).view(bsz, seqlen, self.n_head, self.head_dim)
        
        k = k.transpose(1, 2) 
        scores = torch.matmul(q.transpose(1, 2), k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v.transpose(1, 2))  
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class T6MLP(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        # Calculate the floored hidden dimension size
        hidden_dim = math.floor(8 / 3 * dim)

        # Split the linear projection into two parts for SwiGLU
        self.c_fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(dim, hidden_dim, bias=False)

        # Output projection
        self.c_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977

    def forward(self, x):
        # Apply the first linear layer to produce two projections
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)

        # Apply the SwiGLU gating: SILU on one projection, and gate with the other
        x = F.silu(x1) * x2

        # Apply the final output projection
        x = self.c_proj(x)
        return x
    
class T6Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = TPA(args)
        self.feed_forward = T6MLP(dim=args.dim)
        self.layer_id = layer_id

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(F.rms_norm(x, (x.size(-1),)), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(F.rms_norm(h, (h.size(-1),)))
        return out


class T6Model(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(T6Block(layer_id, params))

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = F.rms_norm(h, (h.size(-1),))
        output = self.output(h).float()
        return output