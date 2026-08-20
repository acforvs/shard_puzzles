"""Local backward helpers; tensor contractions and communication stay in the puzzle."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from . import rules
from .errors import PuzzleError
from .tensor import ShardedTensor, require_tensor


def gelu_backward(dy: ShardedTensor, x: ShardedTensor) -> ShardedTensor:
    """Gradient through GELU, given its saved input x.

    x must be fully reduced and laid out like dy. The operation is linear in dy, so an
    incoming partial-sum gradient stays partial over the same groups.
    """
    return _activation_backward(dy, x, F.gelu, "gelu_backward")


def silu_backward(dy: ShardedTensor, x: ShardedTensor) -> ShardedTensor:
    """Gradient through SiLU, given its saved input x.

    x must be fully reduced and laid out like dy. Incoming partial-sum gradients
    stay partial over the same groups.
    """
    return _activation_backward(dy, x, F.silu, "silu_backward")


def _activation_backward(
    dy: ShardedTensor, x: ShardedTensor, activation: Callable[[torch.Tensor], torch.Tensor], op: str
) -> ShardedTensor:
    dy, x = require_tensor(dy, op), require_tensor(x, op)
    if not dy.layout.same_dims(x.layout):
        raise PuzzleError(f"{op}: the gradient [{dy.spec}] must be laid out like the saved tensor [{x.spec}]")
    rules.require_reduced(x.layout, op)
    rules.require_no_partial_max(dy.layout, op)
    shards = {}
    for rank, saved in x.shards.items():
        saved = saved.clone().requires_grad_(True)
        (shards[rank],) = torch.autograd.grad(activation(saved), saved, dy.shards[rank])
    return dy.with_data(shards)
