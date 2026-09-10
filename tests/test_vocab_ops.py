"""Vocabulary lookups respect row ownership and accumulate repeated token gradients."""

from types import SimpleNamespace

import pytest
import torch

from shardpuzzles import *
from shardpuzzles.errors import PuzzleError
from shardpuzzles.tensor import from_global, to_global
from solutions.p06b_vocab_parallel_cross_entropy_backward import VocabParallelCrossEntropyBackward


@pytest.mark.parametrize("layout", ["v d", "v:tp d", "v d:tp"])
def test_embedding_values_and_repeated_id_gradients_match_autograd(mesh, layout):
    mesh(b=2, s=4, v=6, d=6, tp=3)
    tokens = torch.tensor([[0, 1, 0, 5], [1, 5, 0, 5]])
    ids = from_global(tokens, "b s")
    table = mesh.tensor(layout)
    dy = mesh.tensor("b s d:tp" if layout == "v d:tp" else "b s d")
    lookup = embedding(ids, table)
    if layout == "v:tp d":
        partial = lookup.expect("b s d", partial=tp)
        assert torch.count_nonzero(partial.shards[(1,)]) == 0  # this rank owns unused rows 2 and 3
        y = all_reduce(partial, tp).expect("b s d")
    else:
        y = lookup.expect("b s d:tp" if layout == "v d:tp" else "b s d")
    assert torch.equal(to_global(y), to_global(table)[tokens])

    grad = embedding_backward(ids, dy, layout).expect(layout)
    saved = to_global(table).requires_grad_(True)
    (expected,) = torch.autograd.grad(saved[tokens], saved, to_global(dy))
    assert torch.equal(to_global(grad), expected)
    assert torch.count_nonzero(expected[2:5]) == 0


def test_embedding_gradient_over_split_tokens_needs_a_reduction(mesh):
    mesh(b=4, s=2, v=6, d=3, dp=2, tp=3)
    tokens = torch.tensor([[0, 1], [2, 5], [0, 5], [4, 1]])
    ids = from_global(tokens, "b:dp s")
    dy = mesh.tensor("b:dp s d")
    partial = embedding_backward(ids, dy, "v:tp d").expect("v:tp d", partial=dp)
    grad = all_reduce(partial, dp).expect("v:tp d")
    table = torch.zeros(6, 3, dtype=torch.float64, requires_grad=True)
    (expected,) = torch.autograd.grad(table[tokens], table, to_global(dy))
    assert torch.equal(to_global(grad), expected)


def test_take_and_one_hot_handle_vocab_shard_boundaries(mesh):
    mesh(b=2, s=3, v=6, tp=3)
    tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
    ids = from_global(tokens, "b s")
    logits = mesh.tensor("b s v:tp")
    local = take(logits, ids).expect("b s", partial=tp)
    target = all_reduce(local, tp).expect("b s")
    expected = to_global(logits).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(to_global(target), expected)
    encoded = one_hot(ids, "v:tp").expect("b s v:tp")
    assert torch.equal(to_global(encoded), torch.nn.functional.one_hot(tokens, 6))


def test_unowned_infinite_values_contribute_zero(mesh):
    mesh(b=1, s=2, v=4, d=2, tp=2)
    ids = from_global(torch.tensor([[2, 3]]), "b s")
    logits = from_global(torch.tensor([[[float("-inf"), 1, 2, 3]] * 2]), "b s v:tp")
    partial = take(logits, ids).expect("b s", partial=tp)
    assert torch.equal(partial.shards[(0,)], torch.zeros(1, 2))
    table = from_global(torch.full((4, 2), float("inf")), "v:tp d")
    lookup = embedding(ids, table).expect("b s d", partial=tp)
    assert torch.equal(lookup.shards[(0,)], torch.zeros(1, 2, 2))


@pytest.mark.parametrize("bad", [-1, 6, 1.5])
@pytest.mark.parametrize("op", ["embedding", "embedding_backward", "take", "one_hot"])
def test_vocab_helpers_reject_invalid_ids(mesh, bad, op):
    mesh(b=1, s=2, v=6, d=3, tp=3)
    ids = from_global(torch.tensor([[bad, bad]]), "b s")
    with pytest.raises(PuzzleError, match=r"integer tensor|ids must be in \[0, 6\)"):
        if op == "embedding":
            embedding(ids, mesh.tensor("v:tp d"))
        elif op == "embedding_backward":
            embedding_backward(ids, mesh.tensor("b s d"), "v:tp d")
        elif op == "take":
            take(mesh.tensor("b s v:tp"), ids)
        else:
            one_hot(ids, "v:tp")


def test_vocab_helpers_explain_incompatible_layouts(mesh):
    mesh(b=2, s=4, v=6, d=3, tp=2)
    ids = from_global(torch.zeros(2, 4, dtype=torch.int64), "b s")
    with pytest.raises(PuzzleError, match="2-D"):
        embedding(ids, mesh.tensor("v:tp"))
    with pytest.raises(PuzzleError, match="2-D"):
        embedding_backward(ids, mesh.tensor("b s d"), "v:tp")
    with pytest.raises(PuzzleError, match="one vocabulary dimension"):
        one_hot(ids, "v:tp d")
    with pytest.raises(PuzzleError, match="must be laid out like the lookup's output"):
        embedding_backward(ids, mesh.tensor("b s"), "v:tp d")
    with pytest.raises(PuzzleError, match="without dim"):
        take(mesh.tensor("b s v:tp"), from_global(torch.zeros(2, dtype=torch.int64), "b"))


def test_indices_cannot_be_partial_sums(mesh):
    mesh(b=2, s=4, v=6, tp=2)
    tokens = from_global(torch.zeros(2, 4, 6, dtype=torch.int64), "b s v:tp")
    partial = sum(tokens, "v").expect("b s", partial=tp)
    with pytest.raises(PuzzleError, match="unreduced partial sum"):
        one_hot(partial, "v:tp")


def test_cross_entropy_is_stable_for_large_logits_in_forward_and_backward(mesh):
    mesh(b=1, s=2, d=2, v=6, tp=3)
    x = torch.tensor([[[-1.0, 1.0], [1.0, 1.0]]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor(
        [[1000.0, -1000.0, 1000.0, -1000.0, 0.0, 0.0], [0.0, 3.0, 1.0, 2.0, 1.0, -1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    targets = torch.tensor([[0, 5]])
    model = VocabParallelCrossEntropyBackward()
    model.on_init(SimpleNamespace(W=from_global(weight.detach(), "d v", is_weight=True)))
    loss = model.forward(from_global(x.detach(), "b s d"), from_global(targets, "b s"))
    dy = mesh.tensor("b s")
    grads = model.backward(dy)
    reference = torch.nn.functional.cross_entropy((x @ weight).flatten(0, 1), targets.flatten(), reduction="none").view(
        1, 2
    )
    expected_dx, expected_dw = torch.autograd.grad(reference, (x, weight), to_global(dy))
    assert torch.isfinite(to_global(loss)).all()
    assert torch.allclose(to_global(loss), reference)
    assert torch.allclose(to_global(grads["x"]), expected_dx)
    assert torch.allclose(to_global(grads["W"]), expected_dw)
