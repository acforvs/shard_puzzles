"""Local backward helpers; tensor contractions and communication stay in the puzzle."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from . import rules
from .errors import PuzzleError
from .layout import DimRef, Size, parse_size
from .simulation import current
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
    dy, x = _gradient_and_saved(dy, x, op)
    shards = {}
    for rank, saved in x.shards.items():
        saved = saved.clone().requires_grad_(True)
        (shards[rank],) = torch.autograd.grad(activation(saved), saved, dy.shards[rank])
    return dy.with_data(shards)


def _gradient_and_saved(dy: ShardedTensor, saved: ShardedTensor, op: str) -> tuple[ShardedTensor, ShardedTensor]:
    dy, saved = require_tensor(dy, op), require_tensor(saved, op)
    if not dy.layout.same_dims(saved.layout):
        raise PuzzleError(f"{op}: the gradient [{dy.spec}] must be laid out like the saved tensor [{saved.spec}]")
    rules.require_reduced(saved.layout, op)
    rules.require_no_partial_max(dy.layout, op)
    return dy, saved


def softmax_backward(dp: ShardedTensor, p: ShardedTensor, dim: DimRef) -> ShardedTensor:
    """Gradient through softmax, given its saved output p. Requires whole rows.

    Incoming partial-sum gradients remain partial over the same groups.
    """
    dp, p = _gradient_and_saved(dp, p, "softmax_backward")
    i = p.dim_index(dim)
    if p.dims[i].split_over:
        raise PuzzleError(f"softmax_backward over {p.dims[i]}: needs whole rows")
    shards = {r: p.shards[r] * (dp.shards[r] - (dp.shards[r] * p.shards[r]).sum(i, keepdim=True)) for r in dp.shards}
    return dp.with_data(shards)


def repeat_interleave_backward(t: ShardedTensor, n: Size, dim: DimRef) -> ShardedTensor:
    """Undo a repeat by summing each group of n consecutive slices, locally on every rank."""
    t = require_tensor(t, "repeat_interleave_backward")
    rules.require_no_partial_max(t.layout, "repeat_interleave_backward")
    i = t.dim_index(dim)
    factor = parse_size(n)
    times = current().evaluate(factor)
    if times <= 0:
        raise PuzzleError("repeat_interleave_backward: repeat count must be positive")
    result_dim = t.dims[i].resized(t.dims[i].size / factor)
    count = result_dim.local_size()
    dims = t.layout.with_dim(i, result_dim).dims
    return t.with_data(t.map_shards(lambda x: x.unflatten(i, (count, times)).sum(i + 1)), dims)
