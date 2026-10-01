"""Expert routing: source-local capacity, dropped choices, and inverse packing."""

import pytest
import torch

from shardpuzzles import *
from shardpuzzles.errors import PuzzleError
from shardpuzzles.tensor import from_global, to_global


def routing_case(mesh, capacity=2):
    mesh(T=6, d=2, E=4, k=2, C=capacity, ep=2)
    values = torch.arange(12, dtype=torch.float64).view(6, 2)
    ids = torch.tensor([[0, 1], [0, 2], [3, 1], [0, 1], [0, 1], [0, 1]])
    x = from_global(values, "T:ep d")
    packed, plan = dispatch(x, from_global(ids, "T:ep k"), "E", "C")
    return x, packed.expect("E C:ep d"), plan


def test_source_ranks_fill_independent_queues_and_drop_overflow(mesh):
    x, buffers, plan = routing_case(mesh)
    expected_slots = torch.tensor([[0, 0], [-1, 0], [0, -1], [0, 0], [-1, -1], [-1, -1]])
    assert torch.equal(to_global(plan.slots), expected_slots)
    kept = expected_slots >= 0
    choices = combine(buffers, plan).expect("T:ep k d")
    expected = to_global(x).unsqueeze(1).expand(-1, 2, -1) * kept.unsqueeze(-1)
    assert torch.equal(to_global(choices), expected)
    assert not mesh.sim.comms  # packing and unpacking are local


def test_expert_all_to_all_round_trip_preserves_choice_order_and_bytes(mesh):
    _, buffers, plan = routing_case(mesh)
    owned = all_to_all(buffers, "C", "E", ep).expect("E:ep C d", bytes="2*E*C/ep*d*(ep-1)/ep")
    returned = all_to_all(owned, "E", "C", ep).expect("E C:ep d", bytes="2*E*C/ep*d*(ep-1)/ep")
    assert torch.equal(to_global(returned), to_global(buffers))
    assert torch.equal(
        to_global(combine(returned, plan).expect("T:ep k d")),
        to_global(combine(buffers, plan).expect("T:ep k d")),
    )


def test_backward_packing_ignores_dropped_choices_and_sums_kept_choices(mesh):
    _, _, plan = routing_case(mesh)
    dy = full("T:ep k d", 1)
    dbuffers = combine_backward(dy, plan).expect("E C:ep d")
    dx = dispatch_backward(dbuffers, plan).expect("T:ep d")
    expected = torch.tensor([[2, 2], [1, 1], [1, 1], [2, 2], [0, 0], [0, 0]])
    assert torch.equal(to_global(dx), expected)


def test_packing_backward_helpers_are_adjoints(mesh):
    x, buffers, plan = routing_case(mesh)
    dbuffers = mesh.tensor("E C:ep d")
    dx = dispatch_backward(dbuffers, plan).expect("T:ep d")
    assert torch.allclose((to_global(buffers) * to_global(dbuffers)).sum(), (to_global(x) * to_global(dx)).sum())
    choices = combine(buffers, plan).expect("T:ep k d")
    dy = mesh.tensor("T:ep k d")
    packed_grad = combine_backward(dy, plan).expect("E C:ep d")
    assert torch.allclose(
        (to_global(choices) * to_global(dy)).sum(), (to_global(buffers) * to_global(packed_grad)).sum()
    )


def test_dropped_choices_do_not_read_nonfinite_buffer_slots(mesh):
    _, _, plan = routing_case(mesh)
    result = combine(full("E C:ep d", float("inf")), plan).expect("T:ep k d")
    kept = to_global(plan.slots) >= 0
    assert torch.isinf(to_global(result)[kept]).all()
    assert torch.count_nonzero(to_global(result)[~kept]) == 0
    dx = dispatch_backward(full("E C:ep d", float("inf")), plan).expect("T:ep d")
    assert torch.equal(to_global(dx)[4:], torch.zeros(2, 2))


def test_zero_capacity_drops_every_choice_in_both_passes(mesh):
    _, buffers, plan = routing_case(mesh, capacity=0)
    assert (to_global(plan.slots) == -1).all()
    result = combine(buffers, plan).expect("T:ep k d")
    grad = combine_backward(full("T:ep k d", 1), plan).expect("E C:ep d")
    dx = dispatch_backward(grad, plan).expect("T:ep d")
    assert torch.count_nonzero(to_global(result)) == 0
    assert torch.count_nonzero(to_global(dx)) == 0


@pytest.mark.parametrize("invalid", [-1, 4, 1.5])
def test_dispatch_rejects_invalid_expert_ids(mesh, invalid):
    mesh(T=4, d=2, E=4, k=2, C=2, ep=2)
    ids = from_global(torch.full((4, 2), invalid), "T:ep k")
    with pytest.raises(PuzzleError, match=r"integer tensor|ids must be in \[0, 4\)"):
        dispatch(mesh.tensor("T:ep d"), ids, "E", "C")


def test_dispatch_requires_split_tokens_and_integral_per_rank_capacity(mesh):
    mesh(T=4, d=2, E=4, k=2, C=3, ep=2)
    ids = from_global(torch.zeros(4, 2, dtype=torch.int64), "T k")
    with pytest.raises(PuzzleError, match="tokens must be split"):
        dispatch(mesh.tensor("T d"), ids, "E", "C")
    with pytest.raises(PuzzleError, match="isn't divisible"):
        dispatch(mesh.tensor("T:ep d"), shard(ids, "T:ep k"), "E", "C")


def test_combine_requires_the_buffers_to_return_to_the_source_ranks(mesh):
    _, buffers, plan = routing_case(mesh)
    owned = all_to_all(buffers, "C", "E", ep).expect("E:ep C d")
    with pytest.raises(PuzzleError, match="combine expects"):
        combine(owned, plan)


def test_topk_and_gather_preserve_local_token_order(mesh):
    mesh(T=6, E=4, k=2, ep=2)
    scores = mesh.tensor("T:ep E")
    values, ids = topk(scores, "k", dim=-1)
    expected_values, expected_ids = to_global(scores).topk(2, dim=-1)
    assert torch.equal(to_global(values), expected_values)
    assert torch.equal(to_global(ids), expected_ids)
    assert torch.equal(to_global(gather(scores, ids)), expected_values)


def test_scatter_sums_duplicate_indices_like_gather_backward(mesh):
    mesh(T=4, E=4, k=2, ep=2)
    tokens = torch.tensor([[0, 0], [1, 3], [2, 2], [3, 0]])
    ids = from_global(tokens, "T:ep k")
    dy = mesh.tensor("T:ep k")
    grad = scatter(dy, ids, dim=-1, size="E")
    saved = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    (expected,) = torch.autograd.grad(saved.gather(-1, tokens), saved, to_global(dy))
    assert torch.equal(to_global(grad), expected)


def test_scatter_preserves_partial_gradients(mesh):
    mesh(T=4, E=4, k=2, f=6, ep=2, tp=2)
    ids = from_global(torch.tensor([[0, 0], [1, 3], [2, 2], [3, 0]]), "T:ep k")
    partial = (mesh.tensor("T:ep f:tp") @ mesh.tensor("f:tp k")).expect("T:ep k", partial=tp)
    grad = scatter(partial, ids, -1, "E").expect("T:ep E", partial=tp)
    actual = all_reduce(grad, tp).expect("T:ep E")
    complete = all_reduce(partial, tp).expect("T:ep k")
    expected = scatter(complete, ids, -1, "E")
    assert torch.allclose(to_global(actual), to_global(expected))


@pytest.mark.parametrize("k", [-1, 5, "3/2"])
def test_topk_validates_candidate_count(mesh, k):
    mesh(T=4, E=4, ep=2)
    with pytest.raises(PuzzleError, match="k must be|whole number"):
        topk(mesh.tensor("T:ep E"), k)


def test_topk_requires_whole_candidate_rows(mesh):
    mesh(T=4, E=4, ep=2)
    with pytest.raises(PuzzleError, match="some of the candidates"):
        topk(mesh.tensor("T E:ep"), 1)


@pytest.mark.parametrize("operation", ["gather", "scatter"])
def test_routing_index_ops_validate_indices(mesh, operation):
    mesh(T=4, E=4, k=2, ep=2)
    ids = from_global(torch.full((4, 2), 4), "T:ep k")
    with pytest.raises(PuzzleError, match=r"ids must be in \[0, 4\)"):
        if operation == "gather":
            gather(mesh.tensor("T:ep E"), ids)
        else:
            scatter(mesh.tensor("T:ep k"), ids, -1, "E")


def test_sigmoid_backward_matches_autograd(mesh):
    mesh(T=4, E=4, ep=2)
    x, dy = mesh.tensor("T:ep E"), mesh.tensor("T:ep E")
    actual = sigmoid_backward(dy, sigmoid(x))
    saved = to_global(x).requires_grad_(True)
    (expected,) = torch.autograd.grad(saved.sigmoid(), saved, to_global(dy))
    assert torch.allclose(to_global(actual), expected)
