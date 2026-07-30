"""Regressions for layout validity and global tensor reconstruction."""

import pytest
import torch

from shardpuzzles import *
from shardpuzzles.errors import PuzzleError
from shardpuzzles.layout import Layout
from shardpuzzles.tensor import to_global


def test_one_mesh_axis_cannot_split_two_dimensions(mesh):
    mesh(a=4, b=4, tp=2)
    with pytest.raises(PuzzleError, match="diagonal blocks"):
        shard(mesh.tensor("a b"), "a:tp b:tp")


@pytest.mark.parametrize("spec", ["a:tp"])
@pytest.mark.parametrize("kind", ["partial", "partial_max"])
def test_a_group_cannot_split_entries_and_reduce_the_same_entries(mesh, spec, kind):
    mesh(a=4, tp=2)
    with pytest.raises(PuzzleError, match="different entries"):
        Layout.parse(spec, **{kind: tp})


def test_partial_groups_must_exist_on_the_mesh(mesh):
    mesh(a=4, tp=2)
    with pytest.raises(PuzzleError, match="group dp isn't part"):
        Layout.parse("a", partial=dp)


def test_mixed_sum_and_max_require_an_explicit_order(mesh):
    mesh(a=4, tp=2, dp=2)
    with pytest.raises(PuzzleError, match="finish one reduction first"):
        Layout.parse("a", partial=tp, partial_max=dp)


def test_negative_dimensions_are_reported_as_puzzle_errors(mesh):
    mesh(a=4)
    with pytest.raises(PuzzleError, match="negative size"):
        full("-1 a", 0)


def test_named_singleton_axes_do_not_break_a_valid_reshape(mesh):
    mesh(b=2, s=3, h=4, hd=1, tp=2)
    x = mesh.tensor("b s (h*hd):tp")
    y = x.view("b s h:tp hd")
    assert torch.equal(to_global(y), to_global(x).reshape(2, 3, 4, 1))


def test_an_all_nan_replica_is_not_treated_as_unwritten_memory(mesh):
    mesh(a=4, tp=2)
    t = mesh.tensor("a")
    t.shards[(0,)].fill_(float("nan"))
    with pytest.raises(PuzzleError, match="ranks disagree"):
        to_global(t)


def test_missing_ranks_cannot_construct_a_tensor(mesh):
    mesh(a=4, tp=2)
    t = mesh.tensor("a:tp")
    with pytest.raises(PuzzleError, match="every simulated rank"):
        ShardedTensor(t.layout, {(0,): t.shards[(0,)]})
