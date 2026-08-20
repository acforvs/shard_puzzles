"""Tensor-layout puzzles with numerical CPU checks."""

from .collectives import all_gather, all_reduce, all_to_all, reduce_scatter
from .definition import Ids, Module, Spec, Weight
from .grad import gelu_backward
from .ops import amax, einsum, exp, full, gelu, log, relu, shard, sigmoid, silu, size, softmax, sum
from .show import show
from .simulation import dp, tp
from .tensor import TODO, Pending, ShardedTensor

__all__ = [
    "tp", "dp", "Module", "ShardedTensor", "Pending", "TODO", "Spec", "Weight", "Ids",
    "shard", "all_gather", "reduce_scatter", "all_reduce", "all_to_all",
    "einsum", "gelu", "silu", "relu", "sigmoid", "exp", "log", "softmax", "sum", "amax",
    "size", "full", "show", "gelu_backward",
]  # fmt: skip
