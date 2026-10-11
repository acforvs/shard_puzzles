"""The sharding rules, one behaviour per test."""

import pytest
import torch

from shardpuzzles import *
from shardpuzzles.errors import PuzzleError, Unfinished
from shardpuzzles.tensor import to_global


def test_spec_strings_parse_and_print(mesh):
    mesh(b=2, s=4, h=2, hd=3, tp=2)
    t = mesh.tensor("b s:tp (h*hd)")
    assert t.spec == "b s:tp h*hd" and t.local_shape == (2, 2, 6)


def test_local_sizes_are_rejected(mesh):
    mesh(b=2, s=4, tp=2)
    with pytest.raises(PuzzleError, match="not `s/tp`"):
        mesh.tensor("b s/tp")


def test_matmul_result_is_pending_until_expected(mesh):
    mesh(b=2, d=4, f=6, tp=2)
    y = mesh.tensor("b d") @ mesh.tensor("d f:tp", is_weight=True)
    with pytest.raises(PuzzleError, match="hasn't been checked"):
        gelu(y)
    with pytest.raises(PuzzleError, match="hasn't been checked"):
        y.transpose(0, 1)
    assert y.expect("b f:tp").spec == "b f:tp"


def test_expect_reports_both_layouts(mesh):
    mesh(b=2, d=4, f=6, tp=2)
    y = mesh.tensor("b d") @ mesh.tensor("d f:tp")
    with pytest.raises(PuzzleError, match=r"expected: b f\n  actual:   b f:tp"):
        y.expect("b f")


def test_contracting_a_split_dim_gives_a_partial_sum(mesh):
    mesh(b=2, d=4, f=6, tp=2)
    y = mesh.tensor("b f:tp") @ mesh.tensor("f:tp d")
    with pytest.raises(PuzzleError, match="this is a partial sum over tp.*Say partial=tp"):
        y.expect("b d")
    assert y.expect("b d", partial=tp).partial == {"tp"}


def test_mismatched_contraction_sharding_is_explained(mesh):
    mesh(b=2, d=4, f=6, tp=2)
    with pytest.raises(PuzzleError, match="contracted dim is d:tp on the left and d on the right"):
        _ = mesh.tensor("b d:tp") @ mesh.tensor("d f")


def test_show_prints_each_rank_and_the_real_value_of_a_partial_sum(mesh, capsys):
    mesh(b=1, d=2, f=4, tp=2)
    x, w = mesh.tensor("b f:tp"), mesh.tensor("f:tp d")
    y = (x @ w).expect("b d", partial=tp)
    assert show(y) is y and capsys.readouterr().out == ""  # silent unless the checker asks for it
    mesh.sim.show_values = True
    show(x)
    show(y)
    out = capsys.readouterr().out
    assert "f[0:2]" in out and "f[2:4]" in out
    real = to_global(all_reduce(y, tp).expect("b d")).flatten().tolist()
    assert "sum over tp" in out and " ".join(f"{v:+.3f}" for v in real) in out


def test_nonlinear_ops_need_a_reduced_input(mesh):
    mesh(b=2, d=4, f=6, tp=2)
    y = (mesh.tensor("b f:tp") @ mesh.tensor("f:tp d")).expect("b d", partial=tp)
    with pytest.raises(PuzzleError, match="unreduced partial sum"):
        gelu(y)


def test_adding_a_full_tensor_to_a_partial_one_is_rejected(mesh):
    mesh(b=2, d=4, f=6, tp=2)
    y = (mesh.tensor("b f:tp") @ mesh.tensor("f:tp d")).expect("b d", partial=tp)
    with pytest.raises(PuzzleError, match="partial-ness differs"):
        _ = y + mesh.tensor("b d")


def test_pending_results_add_up_and_are_checked_once(mesh):
    mesh(b=2, d=4, f=6, tp=2)
    a = mesh.tensor("b f:tp")
    total = a @ mesh.tensor("f:tp d") + a @ mesh.tensor("f:tp d")
    assert total.expect("b d", partial=tp).spec == "b d  partial(tp)"


def test_view_cannot_move_sharding_inward(mesh):
    mesh(b=2, h=2, hd=3, tp=2)
    t = mesh.tensor("b (h*hd):tp")
    assert t.view("b h:tp hd").spec == "b h:tp hd"
    with pytest.raises(PuzzleError, match="outermost"):
        t.view("b h hd:tp")


def test_view_ignores_size_one_dims(mesh):
    mesh(b=2, d=4, tp=2)
    assert mesh.tensor("b d:tp").view("b 1 d:tp").spec == "b 1 d:tp"


def test_softmax_over_a_split_dim_is_rejected(mesh):
    mesh(b=2, s=4, tp=2)
    with pytest.raises(PuzzleError, match="only sees part of the row"):
        softmax(mesh.tensor("b s:tp"), dim=-1)


def test_collectives_move_the_right_data_and_bytes(mesh):
    mesh(b=2, s=4, d=6, tp=2)
    x = mesh.tensor("b s:tp d")
    gathered = all_gather(x, "s", tp).expect("b s d", bytes="2*b*s*d*(tp-1)/tp")
    assert torch.equal(to_global(gathered), to_global(x))
    with pytest.raises(PuzzleError, match="incorrect byte count for all_gather"):
        all_gather(x, "s", tp).expect("b s d", bytes="2*b*s*d")


def test_reduce_scatter_needs_a_partial_input(mesh):
    mesh(b=2, s=4, d=6, tp=2)
    with pytest.raises(PuzzleError, match="nothing to reduce"):
        reduce_scatter(mesh.tensor("b s d"), "s", tp)


def test_all_reduce_max_for_partial_max(mesh):
    mesh(b=2, v=4, tp=2)
    m = amax(mesh.tensor("b v:tp"), -1).expect("b", partial_max=tp)
    with pytest.raises(PuzzleError, match="use op='max'"):
        all_reduce(m, tp)
    assert all_reduce(m, tp, op="max").expect("b", bytes="4*b*(tp-1)/tp").spec == "b"


def test_all_to_all_swaps_which_dim_is_split(mesh):
    mesh(b=2, s=4, h=2, tp=2)
    x = mesh.tensor("b s:tp h")
    y = all_to_all(x, "s", "h", tp).expect("b s h:tp")
    assert torch.equal(to_global(y), to_global(x))


def test_context_parallel_all_to_all_preserves_other_mesh_slices(mesh):
    mesh(b=1, s=6, h=6, hd=4, tp=2, cp=3)
    x = mesh.tensor("b s:cp h hd:tp")
    y = all_to_all(x, "s", "h", cp).expect("b s h:cp hd:tp")
    assert torch.equal(to_global(y), to_global(x))
    returned = all_to_all(y, "h", "s", cp).expect("b s:cp h hd:tp")
    assert all(torch.equal(returned.shards[r], x.shards[r]) for r in mesh.sim.ranks)


def test_all_to_all_reports_heads_that_cannot_be_divided_between_ranks(mesh):
    mesh(b=1, s=4, hkv=1, hd=2, cp=2)
    with pytest.raises(PuzzleError, match="isn't divisible by cp"):
        all_to_all(mesh.tensor("b s:cp hkv hd"), "s", "hkv", cp)
    assert not mesh.sim.comms


def test_einsum_sum_over_split_dim_is_partial(mesh):
    mesh(b=4, d=3, f=2, dp=2)
    x, dy = mesh.tensor("b:dp d"), mesh.tensor("b:dp f")
    dw = einsum("b d, b f -> d f", x, dy).expect("d f", partial=dp)
    assert torch.allclose(to_global(all_reduce(dw, dp).expect("d f")), to_global(x).T @ to_global(dy))


def test_todo_points_at_the_blank(mesh):
    mesh(b=2, d=4)
    with pytest.raises(Unfinished, match="still a TODO"):
        gelu(TODO)
