from typing import Optional

import torch
import torch.nn.functional as F


def flashtpa_decode_torch(
    aq: torch.Tensor,
    ak: torch.Tensor,
    av: torch.Tensor,
    bq: torch.Tensor,
    bk: torch.Tensor,
    bv: torch.Tensor,
    scale: Optional[float] = None,
    scale_q: Optional[float] = None,
    scale_k: Optional[float] = None,
    scale_v: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Apply Flash Attention for Tensor Product Attention.

    Args:
        aq: Query A tensor of shape (B, N, H, R)
        ak: Key A tensor of shape (B, M, H)
        av: Value A tensor of shape (B, M, H)
        bq: Query B tensor of shape (B, N, R, D)
        bk: Key B tensor of shape (B, M, D)
        bv: Value B tensor of shape (B, M, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        Output tensor of shape (B, N, H, E)
    """
    b, n, h, r = aq.shape
    assert n == 1, "n must be 1 when using tpa_decode_torch"
    d = bq.shape[-1]
    bv.shape[-1]

    if scale is None:
        scale = d**-0.5
    if scale_q is None:
        scale_q = 1 / r
    if scale_k is None:
        scale_k = 1
    if scale_v is None:
        scale_v = 1

    # equivant to compute (q * k ^ T)
    score1 = torch.einsum("b n r d, b m d -> b n m r", bq, bk)
    score2 = torch.einsum("b n h r, b n m r -> b n m h", aq, score1)
    score3 = torch.einsum("b n m h, b m h -> b h n m", score2, ak)

    prob = F.softmax(score3 * scale_q * scale_k * scale, dim=-1)
    o = torch.einsum("b h n m, b m h -> b n m h", prob, av)
    o = torch.einsum("b n m h, b m e -> b n h e", o, bv) * scale_v

    return o


def tpa_decode_naive_torch(
    aq: torch.Tensor,
    ak: torch.Tensor,
    av: torch.Tensor,
    bq: torch.Tensor,
    bk: torch.Tensor,
    bv: torch.Tensor,
    scale: Optional[float] = None,
    scale_q: Optional[float] = None,
    scale_k: Optional[float] = None,
    scale_v: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Apply Flash Attention for Tensor Product Attention.

    Args:
        aq: Query A tensor of shape (B, N, H, R)
        ak: Key A tensor of shape (B, M, H)
        av: Value A tensor of shape (B, M, H)
        bq: Query B tensor of shape (B, N, R, D)
        bk: Key B tensor of shape (B, M, D)
        bv: Value B tensor of shape (B, M, E)
        cu_seqlens: Cumulative sequence lengths tensor, this is used for varlen training

    Returns:
        Output tensor of shape (B, N, H, E)
    """
    b, n, h, r = aq.shape
    assert n == 1, "n must be 1 when using tpa_decode_torch"
    d = bq.shape[-1]
    bv.shape[-1]

    if scale is None:
        scale = d**-0.5
    if scale_q is None:
        scale_q = 1 / r
    if scale_k is None:
        scale_k = 1
    if scale_v is None:
        scale_v = 1

    q = torch.einsum("b n h r, b n r d -> b n h d", aq, bq) * scale_q
    k = torch.einsum("b m h, b m d -> b m h d", ak, bk) * scale_k
    v = torch.einsum("b m h, b m e -> b m h e", av, bv) * scale_v

    score = torch.einsum("b n h d, b m h d -> b h n m", q, k) * scale
    prob = F.softmax(score, dim=-1)
    o = torch.einsum("b h n m, b m h e -> b n h e", prob, v)

    return o


if __name__ == "__main__":
    b, n, h, r, d, e = 2, 512, 32, 16, 128, 64
    dtype = torch.bfloat16
    aq = torch.randn((b, n, h, r), dtype=dtype).cuda()
    ak = torch.randn((b, n, h), dtype=dtype).cuda()
    av = torch.randn((b, n, h), dtype=dtype).cuda()
    bq = torch.randn((b, n, r, d), dtype=dtype).cuda()
    bk = torch.randn((b, n, d), dtype=dtype).cuda()
    bv = torch.randn((b, n, e), dtype=dtype).cuda()
    o = tpa_decode_torch(aq, ak, av, bq, bk, bv)
    print(o.shape)
