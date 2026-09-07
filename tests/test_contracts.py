"""Correct values must not bypass an exercise's storage or communication contract."""

from types import SimpleNamespace

import pytest

from shardpuzzles import *
from shardpuzzles.catalog import discover
from shardpuzzles.errors import WrongResult
from shardpuzzles.runner import run
from solutions.p00b_partial_sums import RowParallelLinear
from solutions.p00c_data_parallel import Linear
from solutions.p01a_mlp_tp import MLP
from solutions.p02a_swiglu_tp_sp import SwiGLU

PUZZLES = discover()


def test_replicated_weights_fail_the_storage_budget():
    class Replicated(RowParallelLinear):
        def on_init(self, p):
            self.W = p.W

    puzzle = PUZZLES["00b"]
    for config in puzzle.spec.configs:
        with pytest.raises(WrongResult, match="stored weights.*storage budget"):
            run(puzzle.spec, Replicated, config)


@pytest.mark.parametrize(
    "wrap", [lambda p: p, lambda p: [p], lambda p: {"originals": p}, lambda p: SimpleNamespace(p=p)]
)
def test_hidden_full_weights_are_counted(wrap):
    class Hidden(RowParallelLinear):
        def on_init(self, p):
            super().on_init(p)
            self.hidden = wrap(p)
            self.cycle = self

    puzzle = PUZZLES["00b"]
    with pytest.raises(WrongResult, match="stored weights.*storage budget"):
        run(puzzle.spec, Hidden, puzzle.spec.configs[0])


def test_aliases_do_not_count_twice():
    class Aliased(RowParallelLinear):
        def on_init(self, p):
            super().on_init(p)
            self.alias = self.W

    puzzle = PUZZLES["00b"]
    run(puzzle.spec, Aliased, puzzle.spec.configs[0])


def test_raw_shards_cannot_hide_weight_storage():
    class Raw(RowParallelLinear):
        def on_init(self, p):
            super().on_init(p)
            self.raw = p.W.shards

    puzzle = PUZZLES["00b"]
    with pytest.raises(WrongResult, match="raw torch tensor"):
        run(puzzle.spec, Raw, puzzle.spec.configs[0])


def test_cached_full_weights_are_checked_after_forward():
    class Cached(RowParallelLinear):
        def forward(self, x):
            y = super().forward(x)
            self.cached = all_gather(self.W, "d", tp).expect("d f", bytes="2*d*f*(tp-1)/tp")
            return y

    puzzle = PUZZLES["00b"]
    with pytest.raises(WrongResult, match="after forward, stored weights"):
        run(puzzle.spec, Cached, puzzle.spec.configs[0])


def test_weight_copies_are_checked_after_backward():
    class Cached(Linear):
        def backward(self, dy):
            grads = super().backward(dy)
            self.cached = self.W * 1
            return grads

    puzzle = PUZZLES["00c"]
    with pytest.raises(WrongResult, match="after backward, stored weights"):
        run(puzzle.spec, Cached, puzzle.spec.configs[0])


def test_redundant_collectives_fail_even_with_correct_values():
    class ExtraCollective(RowParallelLinear):
        def forward(self, x):
            y = super().forward(x)
            all_gather(self.W, "d", tp).expect("d f", bytes="2*d*f*(tp-1)/tp")
            return y

    puzzle = PUZZLES["00b"]
    with pytest.raises(WrongResult, match="forward uses 2 payload collectives; at most 1"):
        run(puzzle.spec, ExtraCollective, puzzle.spec.configs[0])


def test_an_alternative_resharding_algorithm_is_accepted():
    class Rescale(Module):
        def on_init(self, p):
            self.g = p.g

        def forward(self, x):
            return all_to_all(x * self.g, "s", "d", tp).expect("b s d:tp", bytes="2*b*s/tp*d*(tp-1)/tp")

    puzzle = PUZZLES["00a"]
    for config in puzzle.spec.configs:
        run(puzzle.spec, Rescale, config)


def test_replicated_mlp_cannot_bypass_tensor_parallelism():
    class ReplicatedMLP(Module):
        def on_init(self, p):
            self.W1, self.W2 = p.W1, p.W2

        def forward(self, x):
            h = (x @ self.W1).expect("b s f")
            return (gelu(h) @ self.W2).expect("b s d")

    puzzle = PUZZLES["01a"]
    for config in puzzle.spec.configs:
        with pytest.raises(WrongResult, match="stored weights.*storage budget"):
            run(puzzle.spec, ReplicatedMLP, config)


def test_mlp_redundant_communication_fails_the_collective_limit():
    class ExtraCollective(MLP):
        def forward(self, x):
            y = super().forward(x)
            all_gather(self.W1, "f", tp).expect("d f", bytes="2*d*f*(tp-1)/tp")
            return y

    puzzle = PUZZLES["01a"]
    for config in puzzle.spec.configs:
        with pytest.raises(WrongResult, match="forward uses 2 payload collectives; at most 1"):
            run(puzzle.spec, ExtraCollective, config)


def test_replicated_swiglu_cannot_bypass_tensor_parallelism():
    class ReplicatedSwiGLU(Module):
        def on_init(self, p):
            self.Wgate, self.Wup, self.Wdown = p.Wgate, p.Wup, p.Wdown

        def forward(self, x):
            gate = (x @ self.Wgate).expect("b s:tp f")
            up = (x @ self.Wup).expect("b s:tp f")
            return ((silu(gate) * up) @ self.Wdown).expect("b s:tp d")

    puzzle = PUZZLES["02a"]
    for config in puzzle.spec.configs:
        with pytest.raises(WrongResult, match="stored weights.*storage budget"):
            run(puzzle.spec, ReplicatedSwiGLU, config)


def test_swiglu_redundant_communication_fails_the_collective_limit():
    class ExtraCollective(SwiGLU):
        def forward(self, x):
            y = super().forward(x)
            all_gather(self.Wgate, "f", tp).expect("d f", bytes="2*d*f*(tp-1)/tp")
            return y

    puzzle = PUZZLES["02a"]
    for config in puzzle.spec.configs:
        with pytest.raises(WrongResult, match="forward uses 3 payload collectives; at most 2"):
            run(puzzle.spec, ExtraCollective, config)


@pytest.mark.parametrize("pid", ["03a", "04a"])
def test_attention_full_weight_copies_fail_the_storage_budget(pid):
    puzzle = PUZZLES[pid]
    base = puzzle.module_class(solution=True)

    class Cached(base):
        def on_init(self, p):
            super().on_init(p)
            self.originals = p

    for config in puzzle.spec.configs:
        with pytest.raises(WrongResult, match="stored weights.*storage budget"):
            run(puzzle.spec, Cached, config)


@pytest.mark.parametrize("pid", ["03a", "04a"])
def test_attention_extra_communication_fails_the_collective_limit(pid):
    puzzle = PUZZLES[pid]
    base = puzzle.module_class(solution=True)

    class ExtraCollective(base):
        def forward(self, x):
            y = super().forward(x)
            all_gather(x, "s", tp).expect("b s d")
            return y

    for config in puzzle.spec.configs:
        with pytest.raises(WrongResult, match="forward uses 3 payload collectives; at most 2"):
            run(puzzle.spec, ExtraCollective, config)


def test_unfused_attention_cannot_bypass_the_fused_projection_requirement():
    puzzle = PUZZLES["04a"]
    unfused = PUZZLES["03a"].module_class(solution=True)
    for config in puzzle.spec.configs:
        run(puzzle.spec.replace(max_weight_matmuls=None), unfused, config)
        with pytest.raises(WrongResult, match="forward multiplies by weights 4 times; at most 2"):
            run(puzzle.spec, unfused, config)
