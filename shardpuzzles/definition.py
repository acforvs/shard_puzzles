"""What a puzzle file defines: a Module (the code to check) and a Spec (how to check it)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import torch

from .errors import Unfinished


class Module:
    """Puzzle code: shard the weights in on_init, compute in forward (and backward).

    on_init receives the global (replicated) parameters as attributes of `p`: p.W, p.g, ..."""

    def on_init(self, p: SimpleNamespace) -> None:
        raise Unfinished("on_init isn't written yet")

    def forward(self, *inputs: Any, **_: Any) -> Any:
        raise Unfinished("forward isn't written yet")

    def backward(self, *dy: Any, **_: Any) -> Any:
        """Backward puzzles: return {name: gradient} for the input and every trained parameter."""
        raise Unfinished("backward isn't written yet")


@dataclass(frozen=True)
class Ids:
    """An integer input laid out as `spec`, with values in [0, high)."""

    spec: str
    high: str


@dataclass(frozen=True)
class Weight:
    """A parameter. "randn" is scaled by 1/sqrt(fan_in); "norm" is 1 + 0.1*randn (norm scales)."""

    spec: str
    init: Literal["randn", "norm"] = "randn"


Reference = Callable[..., torch.Tensor]  # reference(config, params, **inputs) -> the global output


@dataclass(frozen=True)
class Spec:
    dims: str  # the size names, e.g. "b s d f"
    params: Mapping[str, str | Weight]  # global (unsharded) parameters
    inputs: Mapping[str, str | Ids]  # name -> layout the input arrives in
    output: str  # layout forward must return
    configs: Sequence[dict[str, int]]  # sizes, including mesh groups (tp=2, ...)
    reference: Reference  # single-device forward; autograd on it gives the reference gradients
    grads: Mapping[str, str] | None = None  # backward puzzles: gradient name -> expected layout
    require_bytes: bool = False  # every collective's .expect(...) must state bytes=
    max_comm_bytes: str | None = None  # budget for the total bytes per rank
    max_weight_matmuls: int | None = None  # forward may multiply by weights at most this often
    max_weight_elements: str | None = None  # stored elements per rank; default: one copy of all parameters
    max_forward_collectives: int | str | None = None  # nonempty payload collectives; integer metadata is free

    def replace(self, **changes: Any) -> Spec:
        """A copy with some fields changed (backward puzzles reuse their forward's Spec)."""
        return dataclasses.replace(self, **changes)
