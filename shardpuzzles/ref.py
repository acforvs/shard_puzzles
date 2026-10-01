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


def capacity_slots(ids: torch.Tensor, num_experts: int, capacity: int) -> torch.Tensor:
    """Assign slots in token/choice order, marking overflow as -1."""
    slots = torch.full_like(ids, -1)
    used = [0] * num_experts
    for token in range(ids.shape[0]):
        for choice in range(ids.shape[1]):
            expert = int(ids[token, choice])
            if used[expert] < capacity:
                slots[token, choice] = used[expert]
                used[expert] += 1
    return slots


def routed_experts(c, x, expert_ids, weights, W1, W2) -> torch.Tensor:
    """Evaluate selected experts directly, applying each source rank's capacity limit."""
    kept = []
    for ids in expert_ids.chunk(c["ep"]):
        flat = ids.flatten()
        counts = torch.nn.functional.one_hot(flat, c["E"]).cumsum(0)
        order = counts.gather(1, flat[:, None]).view_as(ids)
        kept.append(order <= c["C"] // c["ep"])
    hidden = torch.nn.functional.gelu(torch.einsum("td,tkdf->tkf", x, W1[expert_ids]))
    out = torch.einsum("tkf,tkfd->tkd", hidden, W2[expert_ids])
    return (out * (weights * torch.cat(kept)).unsqueeze(-1)).sum(1)


def top_k_softmax_router(c, p, x) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose experts by their logits, then normalize just the selected logits."""
    top, expert_ids = (x @ p.Wr).topk(c["k"], dim=-1)
    return expert_ids, top.softmax(-1)
