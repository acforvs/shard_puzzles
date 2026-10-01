"""Local backward helpers; tensor contractions and communication stay in the puzzle."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
import torch.nn.functional as F

from . import rules
from .errors import PuzzleError
from .layout import Dim, DimRef, Layout, Size, parse_dims, parse_size
from .ops import _check_ids_range, _owned_rows, _require_ids, _row_offset
from .simulation import current
from .tensor import Pending, ShardedTensor, require_tensor


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


def sigmoid_backward(dy: ShardedTensor, s: ShardedTensor) -> ShardedTensor:
    """Gradient through sigmoid given its saved output s; incoming partial sums stay partial."""
    dy, s = _gradient_and_saved(dy, s, "sigmoid_backward")
    return dy.with_data({r: dy.shards[r] * s.shards[r] * (1 - s.shards[r]) for r in dy.shards})


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


def one_hot(ids: ShardedTensor, spec: str) -> ShardedTensor:
    """One-hot rows; with a split vocab dim, each rank fills only the columns it owns."""
    ids = _require_ids(ids, "one_hot")
    dims = parse_dims(spec)
    if len(dims) != 1:
        raise PuzzleError("one_hot: specify one vocabulary dimension, e.g. 'v:tp'")
    (dim,) = dims
    n = dim.local_size()
    _check_ids_range(ids, current().evaluate(dim.size), "one_hot")
    layout = Layout(ids.dims + dims)
    shards = {}
    for rank in current().ranks:
        local, owned = _owned_rows(ids.shards[rank], n, _row_offset(dim, n, rank))
        shards[rank] = F.one_hot(local, n).to(torch.float64) * owned.unsqueeze(-1)
    return ShardedTensor(layout, shards, is_weight=False)


def embedding_backward(ids: ShardedTensor, dy: ShardedTensor, spec: str) -> Pending:
    """Sum token gradients into the table rows this rank owns, including repeated ids.

    Split token dimensions and incoming partial gradients leave partial weight gradients.
    """
    ids, dy = _require_ids(ids, "embedding_backward"), require_tensor(dy, "embedding_backward")
    rules.require_no_partial_max(dy.layout, "embedding_backward")
    table = Layout(parse_dims(spec))
    if table.ndim != 2:
        raise PuzzleError("embedding_backward: specify a 2-D table layout, e.g. 'v:tp d'")
    if not Layout(ids.dims + table.dims[1:]).same_dims(dy.layout):
        raise PuzzleError(f"embedding_backward: dy [{dy.spec}] must be laid out like the lookup's output")
    rows, width = table.local_shape()
    _check_ids_range(ids, current().evaluate(table.dims[0].size), "embedding_backward")
    partial = dy.partial | {d.split_over for d in ids.dims if d.split_over}
    layout = dataclasses.replace(table, partial=frozenset(partial))
    shards = {}
    for rank in current().ranks:
        local, owned = _owned_rows(ids.shards[rank].reshape(-1), rows, _row_offset(table.dims[0], rows, rank))
        grad = torch.zeros(rows, width, dtype=torch.float64)
        shards[rank] = grad.index_add(0, local[owned], dy.shards[rank].reshape(-1, width)[owned])
    return Pending(ShardedTensor(layout, shards, is_weight=False), "embedding_backward")


def scatter(src: ShardedTensor, ids: ShardedTensor, dim: DimRef, size: Size) -> ShardedTensor:
    """Add gathered/top-k gradients back into their original positions, summing repeated indices."""
    src, ids = require_tensor(src, "scatter"), _require_ids(ids, "scatter")
    rules.require_no_partial_max(src.layout, "scatter")
    i = src.dim_index(dim)
    if not ids.layout.same_dims(src.layout):
        raise PuzzleError("scatter: ids must be laid out like src")
    output_dim = Dim(parse_size(size))
    _check_ids_range(ids, output_dim.local_size(), "scatter")
    layout = src.layout.with_dim(i, output_dim)
    if src.dims[i].split_over:
        layout = dataclasses.replace(layout, partial=src.partial | {src.dims[i].split_over})
    shape = layout.local_shape()
    shards = {r: x.new_zeros(shape).scatter_add(i, ids.shards[r], x) for r, x in src.shards.items()}
    return ShardedTensor(layout, shards, is_weight=src.is_weight)
