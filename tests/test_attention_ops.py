"""Attention layout helpers preserve local head order and enforce their preconditions."""

import pytest
import torch

from shardpuzzles import *
from shardpuzzles.errors import PuzzleError
from shardpuzzles.tensor import to_global


def test_cat_on_a_split_dim_joins_local_slices(mesh):
    mesh(b=2, h=4, hd=2, tp=2)
    q, k = mesh.tensor("b h:tp hd"), mesh.tensor("b h:tp hd")
    joined = cat([q, k], dim=1).expect("b (2*h):tp hd")
    expected = torch.cat([torch.cat([q.shards[r], k.shards[r]], dim=1) for r in mesh.sim.ranks], dim=1)
    assert torch.equal(to_global(joined), expected)
    assert not torch.equal(to_global(joined), torch.cat([to_global(q), to_global(k)], dim=1))


def test_cat_preserves_weight_storage_tracking_from_every_input(mesh):
    mesh(d=2, h=4, tp=2)
    activation, weight = mesh.tensor("d h:tp"), mesh.tensor("d h:tp", is_weight=True)
    assert cat([activation, weight], dim=1).is_weight


def test_splitting_weights_keeps_their_storage_and_projection_tracking(mesh):
    mesh(b=2, d=3, h=4, tp=2)
    weight = mesh.tensor("d (2*h):tp", is_weight=True)
    q, k = weight.split(["h", "h"], dim=1)
    assert q.is_weight and k.is_weight
    (mesh.tensor("b d") @ q).expect("b h:tp")
    assert mesh.sim.weight_matmuls == 1


@pytest.mark.parametrize(
    "project",
    [
        lambda x, w: x @ w,
        lambda x, w: w.transpose(0, 1) @ x.transpose(0, 1),
        lambda x, w: einsum("b d, d f -> b f", x, w),
        lambda x, w: einsum("d f, b d -> b f", w, x),
    ],
)
def test_projection_budget_counts_weights_on_either_side(mesh, project):
    mesh(b=2, d=3, f=4)
    project(mesh.tensor("b d"), mesh.tensor("d f", is_weight=True))
    assert mesh.sim.weight_matmuls == 1


def test_cat_requires_nonempty_compatible_layouts(mesh):
    mesh(b=2, h=4, hd=2, tp=2)
    with pytest.raises(PuzzleError, match="at least one tensor"):
        cat([], dim=1)
    with pytest.raises(PuzzleError, match="split differently"):
        cat([mesh.tensor("b h:tp hd"), mesh.tensor("b h hd")], dim=1)
    with pytest.raises(PuzzleError, match="differ outside"):
        cat([mesh.tensor("b h:tp hd"), mesh.tensor("h:tp hd")], dim=1)


@pytest.mark.parametrize("spec", ["s s", "s:tp s", "s:cp s"])
def test_causal_mask_uses_global_query_positions(mesh, spec):
    mesh(s=6, tp=2, cp=3)
    mask = causal_mask(spec)
    expected = torch.tensor([[0 if k <= q else float("-inf") for k in range(6)] for q in range(6)])
    assert torch.equal(to_global(mask), expected)


def test_causal_mask_requires_whole_key_rows(mesh):
    mesh(s=6, tp=2)
    with pytest.raises(PuzzleError, match="whole keys"):
        causal_mask("s s:tp")


def test_repeating_sharded_heads_matches_global_repeat(mesh):
    mesh(b=2, h=12, hkv=4, s=3, hd=2, tp=2)
    x = mesh.tensor("b hkv:tp s hd")
    repeated = repeat_interleave(x, "h/hkv", dim=1).expect("b h:tp s hd")
    assert torch.equal(to_global(repeated), to_global(x).repeat_interleave(3, dim=1))


@pytest.mark.parametrize("n", [0, -1, "3/2"])
@pytest.mark.parametrize("op", [repeat_interleave, repeat_interleave_backward])
def test_repeat_count_must_be_a_positive_integer(mesh, n, op):
    mesh(h=12, tp=2)
    with pytest.raises(PuzzleError, match="positive|whole number"):
        op(mesh.tensor("h:tp"), n, dim=0)


def test_repeat_backward_sums_each_group_of_query_head_gradients(mesh):
    mesh(b=2, h=12, hkv=4, s=3, hd=2, tp=2)
    x = mesh.tensor("b hkv:tp s hd")
    dy = mesh.tensor("b h:tp s hd")
    dx = repeat_interleave_backward(dy, "h/hkv", dim=1).expect("b hkv:tp s hd")
    saved = to_global(x).requires_grad_(True)
    (expected,) = torch.autograd.grad(saved.repeat_interleave(3, dim=1), saved, to_global(dy))
    assert torch.equal(to_global(dx), expected)


def test_repeat_backward_cannot_group_heads_across_ranks(mesh):
    mesh(h=6, tp=2)
    with pytest.raises(PuzzleError, match="isn't divisible"):
        repeat_interleave_backward(mesh.tensor("h:tp"), 2, dim=0)


def test_softmax_backward_matches_autograd_with_sharded_heads(mesh):
    mesh(b=2, h=4, s=3, tp=2)
    x, dy = mesh.tensor("b h:tp s s"), mesh.tensor("b h:tp s s")
    p = softmax(x + causal_mask("s s"), dim=-1)
    dx = softmax_backward(dy, p, dim=-1)
    saved = to_global(x).requires_grad_(True)
    probabilities = (saved + to_global(causal_mask("s s"))).softmax(-1)
    (expected,) = torch.autograd.grad(probabilities, saved, to_global(dy))
    assert torch.allclose(to_global(dx), expected)


def test_softmax_backward_preserves_partial_gradients(mesh):
    mesh(b=2, s=4, f=6, tp=2)
    p = softmax(mesh.tensor("b s"), dim=-1)
    dy = (mesh.tensor("b f:tp") @ mesh.tensor("f:tp s")).expect("b s", partial=tp)
    partial = softmax_backward(dy, p, dim=-1).expect("b s", partial=tp)
    actual = all_reduce(partial, tp).expect("b s")
    expected = softmax_backward(all_reduce(dy, tp).expect("b s"), p, dim=-1)
    assert torch.allclose(to_global(actual), to_global(expected))


def test_softmax_backward_requires_matching_layouts_and_whole_rows(mesh):
    mesh(b=2, s=4, f=6, tp=2)
    with pytest.raises(PuzzleError, match="must be laid out like"):
        softmax_backward(mesh.tensor("b s"), mesh.tensor("b f"), dim=-1)
    with pytest.raises(PuzzleError, match="needs whole rows"):
        softmax_backward(mesh.tensor("b s:tp"), mesh.tensor("b s:tp"), dim=-1)


def test_softmax_backward_rejects_unreduced_saved_probabilities(mesh):
    mesh(b=2, s=4, f=6, tp=2)
    partial = (mesh.tensor("b f:tp") @ mesh.tensor("f:tp s")).expect("b s", partial=tp)
    with pytest.raises(PuzzleError, match="unreduced partial sum"):
        softmax_backward(mesh.tensor("b s"), partial, dim=-1)


def test_backward_sums_reject_partial_max_gradients(mesh):
    mesh(b=2, s=4, tp=2)
    partial = amax(mesh.tensor("b s:tp"), "s", keepdim=True).expect("b 1", partial_max=tp)
    with pytest.raises(PuzzleError, match="partial max"):
        softmax_backward(partial, mesh.tensor("b 1"), dim=-1)
    with pytest.raises(PuzzleError, match="partial max"):
        repeat_interleave_backward(partial, 1, dim=0)
