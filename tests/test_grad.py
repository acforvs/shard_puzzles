"""Backward helpers preserve gradient layouts and reject invalid partial states."""

import pytest
import torch

from shardpuzzles import *
from shardpuzzles.errors import PuzzleError
from shardpuzzles.tensor import to_global


@pytest.mark.parametrize("backward", [gelu_backward, silu_backward, sigmoid_backward])
def test_activation_backward_preserves_partial_sums(mesh, backward):
    mesh(b=2, d=3, f=4, tp=2)
    x = mesh.tensor("b d")
    dy = (mesh.tensor("b f:tp") @ mesh.tensor("f:tp d")).expect("b d", partial=tp)
    partial = backward(dy, x).expect("b d", partial=tp)
    actual = all_reduce(partial, tp).expect("b d")
    expected = backward(all_reduce(dy, tp).expect("b d"), x)
    assert torch.allclose(to_global(actual), to_global(expected))


@pytest.mark.parametrize("backward", [gelu_backward, silu_backward, sigmoid_backward])
def test_activation_backward_requires_matching_layouts(mesh, backward):
    mesh(b=2, d=3, f=4)
    with pytest.raises(PuzzleError, match="must be laid out like the saved tensor"):
        backward(mesh.tensor("b d"), mesh.tensor("b f"))


@pytest.mark.parametrize("backward", [gelu_backward, silu_backward, sigmoid_backward])
def test_activation_backward_requires_a_reduced_saved_input(mesh, backward):
    mesh(b=2, d=3, f=4, tp=2)
    x = (mesh.tensor("b f:tp") @ mesh.tensor("f:tp d")).expect("b d", partial=tp)
    with pytest.raises(PuzzleError, match="unreduced partial sum"):
        backward(mesh.tensor("b d"), x)


@pytest.mark.parametrize("backward", [gelu_backward, silu_backward, sigmoid_backward])
def test_activation_backward_rejects_a_partial_max_gradient(mesh, backward):
    mesh(b=2, f=4, tp=2)
    dy = amax(mesh.tensor("b f:tp"), "f", keepdim=True).expect("b 1", partial_max=tp)
    with pytest.raises(PuzzleError, match="partial max"):
        backward(dy, mesh.tensor("b 1"))
