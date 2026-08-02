#!/usr/bin/env python3
"""NVFP4 quantization (torch), bit-identical to ggml-quants.c quantize_row_nvfp4_ref."""
import torch

# ---------------- torch NVFP4 quantization (matches ggml-quants.c) ----------------
KV = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=torch.float32)

def fp32_to_ue4m3(x):
    """x: float32 tensor (>0) -> uint8 ue4m3, matches ggml_fp32_to_ue4m3."""
    x = torch.clamp(x, max=448.0)
    bits = x.view(torch.int32)
    fp32_exp = ((bits >> 23) & 0xFF) - 127
    fp32_man = (bits >> 20) & 0x7
    ue4m3_exp = fp32_exp + 7
    # subnormal: man = round(x * 512), clamp [1, 7]; return 0 if man < 1
    man_sub = (x * 512.0 + 0.5).to(torch.int32)
    man_sub = torch.clamp(man_sub, max=7)
    sub = torch.where(man_sub < 1, torch.zeros_like(man_sub), man_sub)
    # normal: mantissa + round bit, carry into exponent
    round_bit = (bits >> 19) & 1
    m = fp32_man + round_bit
    carry = (m > 7).to(torch.int32)
    m = torch.where(carry > 0, torch.zeros_like(m), m)
    exp = ue4m3_exp + carry
    out = torch.where(exp >= 15, torch.full_like(exp, 0x7E), (exp << 3) | m)
    out = torch.where(ue4m3_exp <= 0, sub, out)
    return out.to(torch.uint8)

def ue4m3_to_fp32(x):
    """uint8 ue4m3 -> float32, matches ggml_ue4m3_to_fp32."""
    xi = x.to(torch.int32)
    exp = (xi >> 3) & 0xF
    man = xi & 0x7
    sub = man.to(torch.float32) * 0.0009765625  # man * 2^-10
    norm = (1.0 + man.to(torch.float32) / 8.0) * torch.pow(2.0, (exp - 8).to(torch.float32))
    d = torch.where(exp == 0, sub, norm)
    d = torch.where((xi == 0) | (xi == 0x7F), torch.zeros_like(d), d)
    return d

def quantize_nvfp4(x):
    """float32 [..., n] -> uint8 NVFP4 bytes, matches quantize_row_nvfp4_ref."""
    xf = x.reshape(-1, 16)
    amax = xf.abs().max(dim=1).values
    ue = fp32_to_ue4m3(amax / 6.0)
    d = ue4m3_to_fp32(ue)
    dist = (xf.unsqueeze(1) - (KV.unsqueeze(0) * d.unsqueeze(1)).unsqueeze(2)).abs()  # [N,16,16]
    idx = dist.argmin(dim=1)  # first minimal index = C's strict-< tie-break
    lo, hi = idx[:, :8], idx[:, 8:]
    qs = (lo | (hi << 4)).to(torch.uint8)  # [N, 8] one sub-block per row
    qs = qs.reshape(-1, 32)                # 4 sub-blocks per 64-value block
    sc = ue.reshape(-1, 4)
    return torch.cat([sc, qs], dim=1).reshape(-1)  # 36 bytes per block

