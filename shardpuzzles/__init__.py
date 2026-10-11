"""Tensor-layout puzzles with numerical CPU checks."""

from .collectives import all_gather, all_reduce, all_to_all, reduce_scatter
from .definition import Ids, Module, Spec, Weight
from .grad import (
    embedding_backward,
    gelu_backward,
    one_hot,
    repeat_interleave_backward,
    scatter,
    sigmoid_backward,
    silu_backward,
    softmax_backward,
)
from .moe import combine, combine_backward, dispatch, dispatch_backward
from .ops import (
    amax,
    cat,
    causal_mask,
    einsum,
    embedding,
    exp,
    full,
    gather,
    gelu,
    log,
    relu,
    repeat_interleave,
    shard,
    sigmoid,
    silu,
    size,
    softmax,
    sum,
    take,
    topk,
)
from .show import show
from .simulation import cp, dp, ep, tp
from .tensor import TODO, Pending, ShardedTensor

__all__ = [
    "tp", "dp", "ep", "cp", "Module", "ShardedTensor", "Pending", "TODO", "Spec", "Weight", "Ids",
    "shard", "all_gather", "reduce_scatter", "all_reduce", "all_to_all",
    "einsum", "gelu", "silu", "relu", "sigmoid", "exp", "log", "softmax", "sum", "amax",
    "size", "full", "show", "cat", "repeat_interleave", "causal_mask",
    "gelu_backward", "silu_backward", "softmax_backward", "repeat_interleave_backward",
    "embedding", "embedding_backward", "take", "one_hot",
    "topk", "gather", "scatter", "sigmoid_backward",
    "dispatch", "combine", "dispatch_backward", "combine_backward",
]  # fmt: skip
