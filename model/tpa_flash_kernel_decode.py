from typing import Optional
import itertools
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import os
import pytest
XOPES_DEBUG = eval(os.environ.get("XOPES_DEBUG", default="False"))

def generate_configs(input_dict):
    num_stages_list = input_dict.pop("num_stages", [2])
    num_warps_list = input_dict.pop("num_warps", [4])

    # Extract keys and values from the input dictionary
    keys = list(input_dict.keys())
    values = list(input_dict.values())

    # Generate the Cartesian product of the values
    combinations = list(itertools.product(*values))

    # Create a list of dictionaries from the combinations
    results = [{keys[i]: combo[i] for i in range(len(keys))} for combo in combinations]

    configs = []
    for num_stages in num_stages_list:
        for num_warps in num_warps_list:
            for config in results:
                configs.append(
                    triton.Config(config, num_stages=num_stages, num_warps=num_warps)
                )

    # we only need one config for debug
    if XOPES_DEBUG:
        return configs[:1]
    else:
        return configs
    
THRESHOLD_DICT = {
    torch.float32: [1e-2, 1e-2],
    torch.float16: [5e-2, 1e-2],
    torch.bfloat16: [5e-2, 1e-2],
}

def get_threshold(dtype):
    assert dtype in [
        torch.float16,
        torch.float32,
        torch.bfloat16,
    ], "dtype {dtype} not supported"
    return THRESHOLD_DICT[dtype]

@triton.autotune(
    generate_configs(
        {
            "num_warps": [
                2,
                4,
                8,
            ],
            "BLOCK": [
                128,
            ],
        }
    ),
    key=["B", "H", "D", "E", "M"],
)
@triton.jit
def _tpa_decode_parallel_b(
    AQ,  # B N H R
    AK,  # B M H
    AV,  # B M H
    BQ,  # B N R D
    BK,  # B M D
    BV,  # B M E
    O,  # B N H E
    CU_SEQLENS,  # L
    SCALE: tl.constexpr,
    SCALE_Q: tl.constexpr,
    SCALE_K: tl.constexpr,
    SCALE_V: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(0)

    # compute offset
    offset_aq = off_b * N * H * R
    offset_ak = off_b * M * H
    offset_av = off_b * M * H
    offset_bq = off_b * N * R * D
    offset_bk = off_b * M * D
    offset_bv = off_b * M * E
    offset_o = off_b * N * H * E

    # compute block ptr and mask
    array_h = tl.arange(0, BLOCK_H)
    array_r = tl.arange(0, BLOCK_R)
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    array_m = tl.arange(0, BLOCK)

    mask_h = array_h < H
    mask_r = array_r < R
    mask_d = array_d < D
    mask_e = array_e < E

    aq_block_ptr = AQ + offset_aq + array_h[None, :] * R + array_r[:, None]  # R H
    ak_block_ptr = AK + offset_ak + array_m[:, None] * H + array_h[None, :]  # M H
    av_block_ptr = AV + offset_av + array_m[:, None] * H + array_h[None, :]  # M H
    bq_block_ptr = BQ + offset_bq + array_r[None, :] * D + array_d[:, None]  # D R
    bk_block_ptr = BK + offset_bk + array_m[:, None] * D + array_d[None, :]  # M D
    bv_block_ptr = BV + offset_bv + array_m[None, :] * E + array_e[:, None]  # E M

    NUM_BLOCKS = tl.cdiv(M, BLOCK)
    o = tl.zeros([BLOCK_E, BLOCK_H], dtype=tl.float32)
    m = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    sse = tl.full([BLOCK_H], 0, dtype=tl.float32)
    c = SCALE * SCALE_Q * SCALE_K

    aq = tl.load(aq_block_ptr, mask=mask_r[:, None] & mask_h[None, :], other=0)
    bq = tl.load(bq_block_ptr, mask=mask_d[:, None] & mask_r[None, :], other=0)

    for i in range(NUM_BLOCKS):
        mask_m = (i * BLOCK + array_m) < M

        ak = tl.load(ak_block_ptr, mask=mask_m[:, None] & mask_h[None, :], other=0)
        av = tl.load(av_block_ptr, mask=mask_m[:, None] & mask_h[None, :], other=0)
        bk = tl.load(bk_block_ptr, mask=mask_m[:, None] & mask_d[None, :], other=0)
        bv = (
            tl.load(bv_block_ptr, mask=mask_e[:, None] & mask_m[None, :], other=0)
            * SCALE_V
        )

        # M D, D R -> M R
        score1 = tl.dot(bk, bq).to(aq.dtype)
        # M R, R H -> M H
        score2 = tl.dot(score1, aq)
        # M H, M H -> N H
        score3 = score2 * ak * c

        # safe softmax
        # local attention
        # M H -> H
        mi = tl.max(score3, axis=0)
        m_ = tl.maximum(m, mi)
        # M H -> H
        sse_local = tl.sum(tl.exp(score3 - m_), axis=0)
        # M H, H -> M H
        p = tl.exp(score3 - m_) / sse_local * av
        # E M, M H -> E H
        o_ = tl.dot(bv.to(p.dtype), p)

        # update
        sse = tl.exp(m - m_) * sse + sse_local
        ratio = sse_local / sse
        o = (1 - ratio) * o + ratio * o_

        ak_block_ptr += BLOCK * H
        av_block_ptr += BLOCK * H
        bk_block_ptr += BLOCK * D
        bv_block_ptr += BLOCK * E
        m = m_

    o_block_ptr = O + offset_o + array_h[None, :] * E + array_e[:, None]  # E H

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_e[:, None] & mask_h[None, :],
    )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [
                2,
                4,
                8,
            ],
            "BLOCK": [
                128,
                256,
            ],
        }
    ),
    key=["B", "H", "D", "E", "M"],
)
@triton.jit
def _tpa_decode_parallel_bh(
    AQ,  # B N H R
    AK,  # B M H
    AV,  # B M H
    BQ,  # B N R D
    BK,  # B M D
    BV,  # B M E
    O,  # B N H E
    CU_SEQLENS,  # L
    SCALE: tl.constexpr,
    SCALE_Q: tl.constexpr,
    SCALE_K: tl.constexpr,
    SCALE_V: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    # compute offset
    offset_aq = off_b * N * H * R + off_h * R
    offset_ak = off_b * M * H + off_h
    offset_av = off_b * M * H + off_h
    offset_bq = off_b * N * R * D
    offset_bk = off_b * M * D
    offset_bv = off_b * M * E
    offset_o = off_b * N * H * E + off_h * E

    # compute block ptr and mask
    array_r = tl.arange(0, BLOCK_R)
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    array_m = tl.arange(0, BLOCK)

    mask_r = array_r < R
    mask_d = array_d < D
    mask_e = array_e < E

    aq_block_ptr = AQ + offset_aq + array_r  # R
    ak_block_ptr = AK + offset_ak + array_m * H  # M
    av_block_ptr = AV + offset_av + array_m * H  # M
    bq_block_ptr = BQ + offset_bq + array_r[None, :] * D + array_d[:, None]  # D R
    bk_block_ptr = BK + offset_bk + array_m[:, None] * D + array_d[None, :]  # M D
    bv_block_ptr = BV + offset_bv + array_m[None, :] * E + array_e[:, None]  # E M

    NUM_BLOCKS = tl.cdiv(M, BLOCK)
    o = tl.zeros([BLOCK_E], dtype=tl.float32)
    m = tl.full([1], -float("inf"), dtype=tl.float32)
    sse = tl.full([1], 0, dtype=tl.float32)
    c = SCALE * SCALE_Q * SCALE_K

    aq = tl.load(aq_block_ptr, mask=mask_r, other=0)
    bq = tl.load(bq_block_ptr, mask=mask_d[:, None] & mask_r[None, :], other=0)

    for i in range(NUM_BLOCKS):
        mask_m = (i * BLOCK + array_m) < M

        ak = tl.load(ak_block_ptr, mask=mask_m, other=0)
        av = tl.load(av_block_ptr, mask=mask_m, other=0)
        bk = tl.load(bk_block_ptr, mask=mask_m[:, None] & mask_d[None, :], other=0)
        bv = (
            tl.load(bv_block_ptr, mask=mask_e[:, None] & mask_m[None, :], other=0)
            * SCALE_V
        )

        # M D, D R -> M R
        score1 = tl.dot(bk, bq).to(aq.dtype)
        # M R, R -> M
        score2 = tl.sum(score1 * aq, axis=1)
        # M, M -> M
        score3 = score2 * ak * c

        # safe softmax
        # local attention
        # M -> 1
        mi = tl.max(score3, axis=0)
        m_ = tl.maximum(m, mi)
        # M -> 1
        sse_local = tl.sum(tl.exp(score3 - m_), axis=0, keep_dims=True)
        # M, 1 -> M
        p = tl.exp(score3 - m_) / sse_local * av
        # E M, M -> E
        o_ = tl.sum(bv.to(p.dtype) * p, axis=1)

        # update
        sse = tl.exp(m - m_) * sse + sse_local
        ratio = sse_local / sse
        o = (1 - ratio) * o + ratio * o_

        ak_block_ptr += BLOCK * H
        av_block_ptr += BLOCK * H
        bk_block_ptr += BLOCK * D
        bv_block_ptr += BLOCK * E
        m = m_

    o_block_ptr = O + offset_o + array_e  # E

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_e,
    )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [
                2,
                4,
                8,
            ],
            "BLOCK": [
                128,
            ],
            "BLOCK_H": [
                16, 
                32
            ],
        }
    ),
    key=["B", "H", "D", "E", "M"],
)
@triton.jit
def _tpa_decode_parallel_bn(
    AQ,  # B N H R
    AK,  # B M H
    AV,  # B M H
    BQ,  # B N R D
    BK,  # B M D
    BV,  # B M E
    O,  # B NUM_BLOCK_M N H E
    LSE,  # B NUM_BLOCK_M H
    CU_SEQLENS,  # L
    SCALE: tl.constexpr,
    SCALE_Q: tl.constexpr,
    SCALE_K: tl.constexpr,
    SCALE_V: tl.constexpr,
    B: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_BLOCK_M: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_m = tl.program_id(2)

    # compute offset
    offset_h = off_h * BLOCK_H
    offset_m = off_m * BLOCK_M
    offset_aq = off_b * N * H * R
    offset_ak = off_b * M * H
    offset_av = off_b * M * H
    offset_bq = off_b * N * R * D
    offset_bk = off_b * M * D
    offset_bv = off_b * M * E
    offset_o = off_b * NUM_BLOCK_M * N * H * E + off_m * N * H * E
    offset_lse = off_b * NUM_BLOCK_M * H + off_m * H

    # compute block ptr and mask
    array_h = offset_h + tl.arange(0, BLOCK_H)
    array_r = tl.arange(0, BLOCK_R)
    array_d = tl.arange(0, BLOCK_D)
    array_e = tl.arange(0, BLOCK_E)
    array_m = offset_m + tl.arange(0, BLOCK)

    mask_h = array_h < H
    mask_r = array_r < R
    mask_d = array_d < D
    mask_e = array_e < E

    aq_block_ptr = AQ + offset_aq + array_h[None, :] * R + array_r[:, None]  # R H
    ak_block_ptr = AK + offset_ak + array_m[:, None] * H + array_h[None, :]  # M H
    av_block_ptr = AV + offset_av + array_m[:, None] * H + array_h[None, :]  # M H
    bq_block_ptr = BQ + offset_bq + array_r[None, :] * D + array_d[:, None]  # D R
    bk_block_ptr = BK + offset_bk + array_m[:, None] * D + array_d[None, :]  # M D
    bv_block_ptr = BV + offset_bv + array_m[None, :] * E + array_e[:, None]  # E M

    cnt = offset_m
    NUM_BLOCKS = tl.cdiv(BLOCK_M, BLOCK)

    o = tl.zeros([BLOCK_E, BLOCK_H], dtype=tl.float32)
    m = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    sse = tl.full([BLOCK_H], 0, dtype=tl.float32)
    c = SCALE * SCALE_Q * SCALE_K

    aq = tl.load(aq_block_ptr, mask=mask_r[:, None] & mask_h[None, :], other=0)
    bq = tl.load(bq_block_ptr, mask=mask_d[:, None] & mask_r[None, :], other=0)

    for i in range(NUM_BLOCKS):
        if cnt < M:
            mask_m = (i * BLOCK + array_m) < M

            ak = tl.load(ak_block_ptr, mask=mask_m[:, None] & mask_h[None, :], other=0)
            av = tl.load(av_block_ptr, mask=mask_m[:, None] & mask_h[None, :], other=0)
            bk = tl.load(bk_block_ptr, mask=mask_m[:, None] & mask_d[None, :], other=0)
            bv = (
                tl.load(bv_block_ptr, mask=mask_e[:, None] & mask_m[None, :], other=0)
                * SCALE_V
            )

            # M D, D R -> M R
            score1 = tl.dot(bk, bq).to(aq.dtype)
            # M R, R H -> M H
            score2 = tl.dot(score1, aq)
            # M H, M H -> N H
            score3 = score2 * ak * c

            # safe softmax
            # local attention
            # M H -> H
            mi = tl.max(score3, axis=0)
            m_ = tl.maximum(m, mi)
            # M H -> H
            sse_local = tl.sum(tl.exp(score3 - m_), axis=0)
            # M H, H -> M H
            p = tl.exp(score3 - m_) / sse_local * av
            # E M, M H -> E H
            o_ = tl.dot(bv.to(p.dtype), p)

            # update
            sse = tl.exp(m - m_) * sse + sse_local
            ratio = sse_local / sse
            o = (1 - ratio) * o + ratio * o_

            ak_block_ptr += BLOCK * H
            av_block_ptr += BLOCK * H
            bk_block_ptr += BLOCK * D
            bv_block_ptr += BLOCK * E
            cnt += BLOCK
            m = m_

    o_block_ptr = O + offset_o + array_h[None, :] * E + array_e[:, None]  # E H

    tl.store(
        o_block_ptr,
        o.to(o_block_ptr.dtype.element_ty),
        mask=mask_e[:, None] & mask_h[None, :],
    )

    lse = tl.log(sse) + m
    lse_block_ptr = LSE + offset_lse + array_h
    tl.store(
        lse_block_ptr,
        lse.to(lse_block_ptr.dtype.element_ty),
        mask=mask_h,
    )


@triton.autotune(
    generate_configs(
        {
            "num_warps": [
                2,
                4,
                8,
            ],
        }
    ),
    key=[
        "B",
        "H",
        "D",
        "E",
    ],
)
@triton.jit
def _tpa_decode_reduce(
    X,  # B NUM_BLOCK_M N H E
    LSE,  # B NUM_BLOCK_M H
    O,  # B N H E
    CU_SEQLENS,  # L
    B: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_NUM_BLOCK_M: tl.constexpr,
    NUM_BLOCK_M: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    # compute offset
    offset_x = off_b * NUM_BLOCK_M * N * H * E + off_h * E
    offset_lse = off_b * NUM_BLOCK_M * H + off_h
    offset_o = off_b * N * H * E + off_h * E

    # compute block ptr and mask
    array_e = tl.arange(0, BLOCK_E)
    array_h = tl.arange(0, BLOCK_H)
    array_num_block_m = tl.arange(0, BLOCK_NUM_BLOCK_M)

    mask_e = array_e < E
    array_h < H
    mask_num_block_m = array_num_block_m < NUM_BLOCK_M

    x_block_ptr = (
        X + offset_x + array_num_block_m[:, None] * N * H * E + array_e[None, :]
    )  # NUM_BLOCK_M E
    lse_block_ptr = LSE + offset_lse + array_num_block_m  # NUM_BLOCK_M
    o_block_ptr = O + offset_o + array_e  # E

    x = tl.load(x_block_ptr, mask=mask_num_block_m[:, None] & mask_e[None, :], other=0)
    lse = tl.load(lse_block_ptr, mask=mask_num_block_m, other=0)
    m = tl.min(lse)
    p = tl.exp(lse - m)
    p = tl.where(mask_num_block_m, p, 0)
    p = p / tl.sum(p)

    o = tl.sum(x * p[:, None], axis=0)
    tl.store(o_block_ptr, o, mask=mask_e)


def tpa_decode_parallel_b_triton(
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
    m = ak.shape[1]
    d = bq.shape[-1]
    e = bv.shape[-1]

    if scale is None:
        scale = d**-0.5
    if scale_q is None:
        scale_q = 1 / r
    if scale_k is None:
        scale_k = 1
    if scale_v is None:
        scale_v = 1

    def grid(meta):
        return (b,)

    o = torch.empty((b, n, h, e), dtype=aq.dtype, device=aq.device)

    BLOCK_H = triton.next_power_of_2(h)
    BLOCK_R = triton.next_power_of_2(r)
    BLOCK_D = triton.next_power_of_2(d)
    BLOCK_E = triton.next_power_of_2(e)

    _tpa_decode_parallel_b[grid](
        AQ=aq,
        AK=ak,
        AV=av,
        BQ=bq,
        BK=bk,
        BV=bv,
        O=o,
        CU_SEQLENS=cu_seqlens,
        SCALE=scale,
        SCALE_Q=scale_q,
        SCALE_K=scale_k,
        SCALE_V=scale_v,
        B=b,
        N=n,
        M=m,
        H=h,
        R=r,
        D=d,
        E=e,
        BLOCK_H=BLOCK_H,
        BLOCK_R=BLOCK_R,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
    )

    return o


def tpa_decode_parallel_bh_triton(
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
    m = ak.shape[1]
    d = bq.shape[-1]
    e = bv.shape[-1]

    if scale is None:
        scale = d**-0.5
    if scale_q is None:
        scale_q = 1 / r
    if scale_k is None:
        scale_k = 1
    if scale_v is None:
        scale_v = 1

    def grid(meta):
        return (b, h)

    o = torch.empty((b, n, h, e), dtype=aq.dtype, device=aq.device)

    BLOCK_H = triton.next_power_of_2(h)
    BLOCK_R = triton.next_power_of_2(r)
    BLOCK_D = triton.next_power_of_2(d)
    BLOCK_E = triton.next_power_of_2(e)

    _tpa_decode_parallel_bh[grid](
        AQ=aq,
        AK=ak,
        AV=av,
        BQ=bq,
        BK=bk,
        BV=bv,
        O=o,
        CU_SEQLENS=cu_seqlens,
        SCALE=scale,
        SCALE_Q=scale_q,
        SCALE_K=scale_k,
        SCALE_V=scale_v,
        B=b,
        N=n,
        M=m,
        H=h,
        R=r,
        D=d,
        E=e,
        BLOCK_H=BLOCK_H,
        BLOCK_R=BLOCK_R,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
    )

    return o


def tpa_decode_parallel_bn_triton_v2(
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
    m = ak.shape[1]
    d = bq.shape[-1]
    e = bv.shape[-1]

    if scale is None:
        scale = d**-0.5
    if scale_q is None:
        scale_q = 1 / r
    if scale_k is None:
        scale_k = 1
    if scale_v is None:
        scale_v = 1

    if b <= 16:
        BLOCK_M = 512
    else:
        BLOCK_M = 1024
    NUM_BLOCK_M = triton.cdiv(m, BLOCK_M)

    def grid(meta):
        NUM_BLOCK_H = triton.cdiv(h, meta["BLOCK_H"])
        return (b, NUM_BLOCK_H, NUM_BLOCK_M)

    o_ = torch.empty((b, NUM_BLOCK_M, n, h, e), dtype=aq.dtype, device=aq.device)
    lse = torch.empty((b, NUM_BLOCK_M, h), dtype=aq.dtype, device=aq.device)

    BLOCK_H = triton.next_power_of_2(h)
    BLOCK_R = triton.next_power_of_2(r)
    BLOCK_D = triton.next_power_of_2(d)
    BLOCK_E = triton.next_power_of_2(e)

    _tpa_decode_parallel_bn[grid](
        AQ=aq,
        AK=ak,
        AV=av,
        BQ=bq,
        BK=bk,
        BV=bv,
        O=o_,
        LSE=lse,
        CU_SEQLENS=cu_seqlens,
        SCALE=scale,
        SCALE_Q=scale_q,
        SCALE_K=scale_k,
        SCALE_V=scale_v,
        B=b,
        N=n,
        M=m,
        H=h,
        R=r,
        D=d,
        E=e,
        BLOCK_R=BLOCK_R,
        BLOCK_D=BLOCK_D,
        BLOCK_E=BLOCK_E,
        BLOCK_M=BLOCK_M,
        NUM_BLOCK_M=NUM_BLOCK_M,
    )

    def grid(meta):
        return (b, h)

    o = torch.empty((b, n, h, e), dtype=aq.dtype, device=aq.device)
    BLOCK_NUM_BLOCK_M = triton.next_power_of_2(NUM_BLOCK_M)

    _tpa_decode_reduce[grid](
        X=o_,
        LSE=lse,
        O=o,
        CU_SEQLENS=cu_seqlens,
        B=b,
        N=n,
        H=h,
        E=e,
        BLOCK_H=BLOCK_H,
        BLOCK_E=BLOCK_E,
        BLOCK_M=BLOCK_M,
        BLOCK_NUM_BLOCK_M=BLOCK_NUM_BLOCK_M,
        NUM_BLOCK_M=NUM_BLOCK_M,
    )

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

def tpa_decode_torch(
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

if __name__ == "__main__":
    torch.manual_seed(2024)
    b = 2
    n = 1
    m = 256
    h = 16
    r = 16
    d = 128
    e = 64
    dtype = torch.bfloat16
    aq = torch.randn((b, n, h, r), dtype=dtype).cuda()
    ak = torch.randn((b, m, h), dtype=dtype).cuda()
    av = torch.randn((b, m, h), dtype=dtype).cuda()
    bq = torch.randn((b, n, r, d), dtype=dtype).cuda()
    bk = torch.randn((b, m, d), dtype=dtype).cuda()
    bv = torch.randn((b, m, e), dtype=dtype).cuda()
    o1 = tpa_decode_parallel_b_triton(aq, ak, av, bq, bk, bv)
    o2 = tpa_decode_parallel_bh_triton(aq, ak, av, bq, bk, bv)
    o3 = tpa_decode_parallel_bn_triton_v2(aq, ak, av, bq, bk, bv)
    o_naive = tpa_decode_naive_torch(aq, ak, av, bq, bk, bv)
    o_decode = tpa_decode_torch(aq, ak, av, bq, bk, bv)
    # print(o_naive[:2, :2, :2, :2])
    # print(o_decode[:2, :2, :2, :2])
    # print(o1[:2, :2, :2, :2])
    # print(o2[:2, :2, :2, :2])
    # print(o3[:2, :2, :2, :2])
    # print(o_naive.shape, o_decode.shape, o1.shape, o2.shape, o3.shape)
    print(f"b: {b}, n: {n}, m: {m}, h: {h}, r: {r}, d: {d}, e: {e}")
    print("naive torch norm:", torch.norm(o_naive).item())
    print("torch norm:", torch.norm(o_decode).item())
    print("triton parallel_b norm:", torch.norm(o1).item())
    print("triton parallel_bh norm:", torch.norm(o2).item())
    print("triton parallel_bn norm:", torch.norm(o3).item())
    print(
        "o diff max (torch vs naive): ",
        torch.abs(o_decode - o_naive).max().item()
    )
    print(
        "o diff max (triton parallel_b vs naive): ",
        torch.abs(o1 - o_naive).max().item()
    )
    print(
        "o diff max (triton parallel_bh vs naive): ",
        torch.abs(o2 - o_naive).max().item()
    )
    print(
        "o diff max (triton parallel_bn vs naive): ",
        torch.abs(o3 - o_naive).max().item()
    )

def get_params():
    shapes = [
        # standard shapes
        (2, 256, 16, 16, 128, 64),
        (2, 2048, 16, 16, 128, 64),
        # special seqlen
        (2, 2049, 16, 16, 128, 64),
        (2, 2047, 16, 16, 128, 64),
        # special rank
        (2, 2049, 17, 16, 128, 64),
        (2, 2047, 31, 16, 128, 64),
        # special num heads
        (2, 2049, 17, 31, 128, 64),
        (2, 2047, 31, 17, 128, 64),
        # special head dim
        (2, 2049, 17, 31, 129, 63),
        (2, 2047, 31, 17, 127, 65),
    ]
    return shapes


@pytest.mark.parametrize("shape", get_params())
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_tpa_decode(shape, dtype):
    # Set random seed for reproducibility
    torch.manual_seed(2024)
    device = torch.device("cuda")

    # Unpack shape parameters
    b, m, h, r, d, e = shape
    n = 1

    # Generate input tensors
    aq = torch.randn((b, n, h, r), dtype=dtype, device=device).requires_grad_()
    ak = torch.randn((b, m, h), dtype=dtype, device=device).requires_grad_()
    av = torch.randn((b, m, h), dtype=dtype, device=device).requires_grad_()
    bq = torch.randn((b, n, r, d), dtype=dtype, device=device).requires_grad_()
    bk = torch.randn((b, m, d), dtype=dtype, device=device).requires_grad_()
    bv = torch.randn((b, m, e), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((b, n, h, e), dtype=dtype, device=device)

    # Optional scale parameters
    scale = d**-0.5
    scale_q = 1 / r
    scale_k = 1.0
    scale_v = 1.0

    ##### Forward pass
    o_naive_torch = tpa_decode_naive_torch(
        aq=aq,
        ak=ak,
        av=av,
        bq=bq,
        bk=bk,
        bv=bv,
        scale=scale,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )

    # PyTorch implementation
    o_torch = tpa_decode_torch(
        aq=aq,
        ak=ak,
        av=av,
        bq=bq,
        bk=bk,
        bv=bv,
        scale=scale,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )

    # Triton implementation
    o_parallel_b_triton = tpa_decode_parallel_b_triton(
        aq=aq,
        ak=ak,
        av=av,
        bq=bq,
        bk=bk,
        bv=bv,
        scale=scale,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )

    o_parallel_bh_triton = tpa_decode_parallel_bh_triton(
        aq=aq,
        ak=ak,
        av=av,
        bq=bq,
        bk=bk,
        bv=bv,
        scale=scale,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )

    o_parallel_bn_triton = tpa_decode_parallel_bn_triton_v2(
        aq=aq,
        ak=ak,
        av=av,
        bq=bq,
        bk=bk,
        bv=bv,
        scale=scale,
        scale_q=scale_q,
        scale_k=scale_k,
        scale_v=scale_v,
    )

    # Get tolerance thresholds based on dtype
    atol, rtol = get_threshold(dtype)
    print("naive torch norm:", torch.norm(o_naive_torch).item())
    print("torch norm:", torch.norm(o_torch).item())
    print("triton parallel_b norm:", torch.norm(o_parallel_b_triton).item())
    print("triton parallel_bh norm:", torch.norm(o_parallel_bh_triton).item())
    print("triton parallel_bn norm:", torch.norm(o_parallel_bn_triton).item())
    print(o_naive_torch[:2, :2, :2, :2])
    print(o_torch[:2, :2, :2, :2])
    print(o_parallel_b_triton[:2, :2, :2, :2])
    print(o_parallel_bh_triton[:2, :2, :2, :2])
    print(o_parallel_bn_triton[:2, :2, :2, :2])
    ##### Check forward pass results
    print(
        "o diff max (torch vs naive): ",
        torch.abs(o_torch - o_naive_torch).max().item(),
    )
    print(
        "o diff norm (torch vs naive): ",
        torch.norm(o_torch - o_naive_torch).item(),
    )
    assert torch.allclose(o_torch, o_naive_torch, atol=atol, rtol=rtol)

    print(
        "o diff max (torch vs triton parallel_b): ",
        torch.abs(o_torch - o_parallel_b_triton).max().item(),
    )
    print(
        "o diff norm (torch vs triton parallel_b): ",
        torch.norm(o_torch - o_parallel_b_triton).item(),
    )
    assert torch.allclose(o_torch, o_parallel_b_triton, atol=atol, rtol=rtol)

    print(
        "o diff max (torch vs triton parallel_bh): ",
        torch.abs(o_torch - o_parallel_bh_triton).max().item(),
    )
    print(
        "o diff norm (torch vs triton parallel_bh): ",
        torch.norm(o_torch - o_parallel_bh_triton).item(),
    )
    assert torch.allclose(o_torch, o_parallel_bh_triton, atol=atol, rtol=rtol)

    print(
        "o diff max (torch vs triton parallel_bn): ",
        torch.abs(o_torch - o_parallel_bn_triton).max().item(),
    )
    print(
        "o diff norm (torch vs triton parallel_bn): ",
        torch.norm(o_torch - o_parallel_bn_triton).item(),
    )
    assert torch.allclose(o_torch, o_parallel_bn_triton, atol=atol, rtol=rtol)
    
# test_tpa_decode((2, 256, 16, 16, 128, 64), torch.bfloat16)