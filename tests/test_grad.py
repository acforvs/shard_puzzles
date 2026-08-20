"""Backward helpers preserve gradient layouts and reject invalid partial states."""

import pytest
import torch

from shardpuzzles import *
from shardpuzzles.errors import PuzzleError
from shardpuzzles.tensor import to_global


def test_gelu_backward_preserves_partial_sums(mesh):
    mesh(b=2, d=3, f=4, tp=2)
    x = mesh.tensor("b d")
    dy = (mesh.tensor("b f:tp") @ mesh.tensor("f:tp d")).expect("b d", partial=tp)
    partial = gelu_backward(dy, x).expect("b d", partial=tp)
    actual = all_reduce(partial, tp).expect("b d")
    expected = gelu_backward(all_reduce(dy, tp).expect("b d"), x)
    assert torch.allclose(to_global(actual), to_global(expected))


def test_gelu_backward_requires_matching_layouts(mesh):
    mesh(b=2, d=3, f=4)
    with pytest.raises(PuzzleError, match="must be laid out like the saved tensor"):
        gelu_backward(mesh.tensor("b d"), mesh.tensor("b f"))


def test_gelu_backward_requires_a_reduced_saved_input(mesh):
    mesh(b=2, d=3, f=4, tp=2)
    x = (mesh.tensor("b f:tp") @ mesh.tensor("f:tp d")).expect("b d", partial=tp)
    with pytest.raises(PuzzleError, match="unreduced partial sum"):
        gelu_backward(mesh.tensor("b d"), x)


def test_gelu_backward_rejects_a_partial_max_gradient(mesh):
    mesh(b=2, f=4, tp=2)
    dy = amax(mesh.tensor("b f:tp"), "f", keepdim=True).expect("b 1", partial_max=tp)
    with pytest.raises(PuzzleError, match="partial max"):
        gelu_backward(dy, mesh.tensor("b 1"))
