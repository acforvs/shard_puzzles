"""Local MoE packing. Learners move the expert buffers between ranks with collectives.

C slots per expert are divided equally among source ranks. Choices fill slots in
token/choice order; overflow contributes zero in both forward and backward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import ref, rules
from .errors import PuzzleError
from .layout import Dim, Layout, Size, format_dims, parse_size
from .ops import _check_ids_range, _require_ids
from .tensor import Pending, ShardedTensor, require_tensor


@dataclass(frozen=True)
class DispatchPlan:
    """Saved routing metadata. Puzzle tensors keep it visible to the state checker."""

    token_dim: Dim
    choice_dim: Dim
    expert_dim: Dim
    slot_dim: Dim
    expert_ids: ShardedTensor
    slots: ShardedTensor  # -1 marks a dropped choice

    @property
    def buffer_dims(self) -> tuple[Dim, Dim]:
        return self.expert_dim, self.slot_dim

    @property
    def choice_dims(self) -> tuple[Dim, Dim]:
        return self.token_dim, self.choice_dim

    def check(self, t: ShardedTensor, leading: tuple[Dim, Dim], op: str) -> None:
        if t.ndim != 3 or not all(a.same_as(b) for a, b in zip(t.dims, leading)):
            raise PuzzleError(f"{op} expects [{format_dims(leading)} *], got [{t.spec}]")


def dispatch(
    x: ShardedTensor, expert_ids: ShardedTensor, experts: Size, capacity: Size
) -> tuple[Pending, DispatchPlan]:
    """Pack local tokens [T:g d] into [E C:g d] buffers, without communication."""
    x, ids = require_tensor(x, "dispatch"), _require_ids(expert_ids, "dispatch")
    rules.require_reduced(x.layout, "dispatch")
    if x.ndim != 2 or ids.ndim != 2:
        raise PuzzleError(f"dispatch expects x [T d] and expert ids [T k]; got [{x.spec}] and [{ids.spec}]")
    tokens = x.dims[0]
    if not tokens.same_as(ids.dims[0]):
        raise PuzzleError(f"dispatch: token dims differ: [{x.spec}] vs [{ids.spec}]")
    if not tokens.split_over or x.dims[1].group or ids.dims[1].group:
        raise PuzzleError("dispatch: tokens must be split over a group, with whole hidden and choice dimensions")
    expert_dim, slot_dim = Dim(parse_size(experts)), Dim(parse_size(capacity), tokens.group)
    n_experts, slots_per_rank = expert_dim.local_size(), slot_dim.local_size()
    _check_ids_range(ids, n_experts, "dispatch")
    shards, saved_ids, slots = {}, {}, {}
    for rank, values in x.shards.items():
        chosen = ids.shards[rank].clone()
        slot = ref.capacity_slots(chosen, n_experts, slots_per_rank)
        kept = slot >= 0
        token, _ = torch.nonzero(kept, as_tuple=True)
        buffer = values.new_zeros(n_experts, slots_per_rank, x.local_shape[1])
        buffer[chosen[kept], slot[kept]] = values[token]
        shards[rank], saved_ids[rank], slots[rank] = buffer, chosen, slot
    plan = DispatchPlan(
        tokens,
        ids.dims[1],
        expert_dim,
        slot_dim,
        ShardedTensor(ids.layout, saved_ids, is_weight=False),
        ShardedTensor(ids.layout, slots, is_weight=False),
    )
    return Pending(ShardedTensor(Layout(plan.buffer_dims + x.dims[1:]), shards, is_weight=False), "dispatch"), plan


def combine(y: ShardedTensor, plan: DispatchPlan) -> Pending:
    """Unpack [E C:g d] expert results into [T:g k d]; dropped choices stay zero."""
    y = require_tensor(y, "combine")
    rules.require_reduced(y.layout, "combine")
    plan.check(y, plan.buffer_dims, "combine")
    layout = Layout(plan.choice_dims + y.dims[2:])
    shards = {}
    for rank, values in y.shards.items():
        ids, slots = plan.expert_ids.shards[rank], plan.slots.shards[rank]
        kept = slots >= 0
        token, choice = torch.nonzero(kept, as_tuple=True)
        output = values.new_zeros(layout.local_shape())
        output[token, choice] = values[ids[kept], slots[kept]]
        shards[rank] = output
    return Pending(ShardedTensor(layout, shards, is_weight=False), "combine")


def combine_backward(dy: ShardedTensor, plan: DispatchPlan) -> Pending:
    """Put each kept per-choice gradient back into its expert buffer slot."""
    dy = require_tensor(dy, "combine_backward")
    rules.require_no_partial_max(dy.layout, "combine_backward")
    plan.check(dy, plan.choice_dims, "combine_backward")
    layout = dy.layout.with_dims(plan.buffer_dims + dy.dims[2:])
    shards = {}
    for rank, values in dy.shards.items():
        ids, slots = plan.expert_ids.shards[rank], plan.slots.shards[rank]
        kept = slots >= 0
        token, choice = torch.nonzero(kept, as_tuple=True)
        buffer = values.new_zeros(layout.local_shape())
        buffer[ids[kept], slots[kept]] = values[token, choice]
        shards[rank] = buffer
    return Pending(ShardedTensor(layout, shards, is_weight=False), "combine_backward")


def dispatch_backward(dbuffers: ShardedTensor, plan: DispatchPlan) -> Pending:
    """Sum each token's kept buffer-slot gradients over its choices."""
    dbuffers = require_tensor(dbuffers, "dispatch_backward")
    rules.require_no_partial_max(dbuffers.layout, "dispatch_backward")
    plan.check(dbuffers, plan.buffer_dims, "dispatch_backward")
    layout = dbuffers.layout.with_dims((plan.token_dim, dbuffers.dims[2]))
    shards = {}
    for rank, values in dbuffers.shards.items():
        ids, slots = plan.expert_ids.shards[rank], plan.slots.shards[rank]
        kept = slots >= 0
        token, _ = torch.nonzero(kept, as_tuple=True)
        output = values.new_zeros(layout.local_shape())
        shards[rank] = output.index_add(0, token, values[ids[kept], slots[kept]])
    return Pending(ShardedTensor(layout, shards, is_weight=False), "dispatch_backward")
