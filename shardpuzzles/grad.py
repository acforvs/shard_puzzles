"""Local backward helpers; tensor contractions and communication stay in the puzzle."""

from __future__ import annotations

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
    dy, x = require_tensor(dy, "gelu_backward"), require_tensor(x, "gelu_backward")
    if not dy.layout.same_dims(x.layout):
        raise PuzzleError(f"gelu_backward: the gradient [{dy.spec}] must be laid out like the saved tensor [{x.spec}]")
    rules.require_reduced(x.layout, "gelu_backward")
    rules.require_no_partial_max(dy.layout, "gelu_backward")
    shards = {}
    for rank, saved in x.shards.items():
        saved = saved.clone().requires_grad_(True)
        (shards[rank],) = torch.autograd.grad(F.gelu(saved), saved, dy.shards[rank])
    return dy.with_data(shards)
