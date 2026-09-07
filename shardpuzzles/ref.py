"""Single-device PyTorch references for checking outputs and gradients."""

import torch


def gqa_attention(c, p, x) -> torch.Tensor:
    """Causal grouped-query attention with head-major projection columns."""
    b, s, _ = x.shape
    h, hkv, hd = c["h"], c["hkv"], c["hd"]
    q = (x @ p.Wq).view(b, s, h, hd).transpose(1, 2)
    k = (x @ p.Wk).view(b, s, hkv, hd).transpose(1, 2).repeat_interleave(h // hkv, dim=1)
    v = (x @ p.Wv).view(b, s, hkv, hd).transpose(1, 2).repeat_interleave(h // hkv, dim=1)
    positions = torch.arange(s)
    visible = positions[:, None] >= positions[None, :]
    mask = torch.where(visible, 0.0, float("-inf")).to(x.dtype)
    o = torch.softmax(q @ k.transpose(-1, -2) * hd**-0.5 + mask, dim=-1) @ v
    return o.transpose(1, 2).reshape(b, s, h * hd) @ p.Wo
