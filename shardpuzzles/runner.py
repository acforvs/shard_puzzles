"""Running puzzle code on simulated ranks and comparing it with the single-device reference."""

from __future__ import annotations

import enum
from collections.abc import Iterator
from dataclasses import dataclass, fields, is_dataclass
from types import SimpleNamespace

import sympy as sp
import torch

from . import simulation
from .definition import Ids, Module, Spec, Weight
from .errors import Unfinished, WrongResult
from .layout import Layout, parse_dims, parse_size
from .simulation import Phase, Simulation
from .tensor import TODO, Pending, ShardedTensor, from_global, to_global


class Outcome(enum.Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNFINISHED = "unfinished"  # a TODO or ??? is left


@dataclass
class Data:
    """Global tensors (for the reference) and their per-rank versions (for the puzzle)."""

    global_params: dict[str, torch.Tensor]
    params: dict[str, ShardedTensor]
    global_inputs: dict[str, torch.Tensor]
    inputs: dict[str, ShardedTensor]
    global_dy: torch.Tensor  # the gradient fed to backward


def global_shape(spec: str) -> list[int]:
    return [simulation.current().evaluate(d.size) for d in parse_dims(spec)]


def make_data(spec: Spec, seed: int) -> Data:
    generator = torch.Generator().manual_seed(seed)
    trained = set(spec.grads or ())

    def randn(shape: list[int]) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float64)

    global_params, params = {}, {}
    for name, weight in spec.params.items():
        weight = weight if isinstance(weight, Weight) else Weight(weight)
        shape = global_shape(weight.spec)
        value = 1 + 0.1 * randn(shape) if weight.init == "norm" else randn(shape) / shape[0] ** 0.5
        global_params[name] = value.requires_grad_(name in trained)
        params[name] = from_global(value.detach(), weight.spec, is_weight=True)

    global_inputs, inputs = {}, {}
    for name, layout in spec.inputs.items():
        if isinstance(layout, Ids):
            high = simulation.current().evaluate(parse_size(layout.high))
            value = torch.randint(0, high, global_shape(layout.spec), generator=generator)
            layout = layout.spec
        else:
            value = randn(global_shape(layout)).requires_grad_(name in trained)
        global_inputs[name] = value
        inputs[name] = from_global(value.detach(), layout)
    return Data(global_params, params, global_inputs, inputs, randn(global_shape(spec.output)))


def returned_tensor(value: object, spec: str, what: str) -> ShardedTensor:
    """What forward/backward returned must be a checked tensor laid out exactly as `spec`."""
    if value is TODO:
        raise Unfinished(f"{what} is still a TODO")
    if isinstance(value, Pending):
        raise WrongResult(f"{what} is an unchecked {value.made_by} result: add .expect(...) before returning it")
    if not isinstance(value, ShardedTensor):
        raise WrongResult(f"{what} is a {type(value).__name__}, not a tensor")
    expected = Layout(parse_dims(spec))
    if not expected.matches(value.layout):
        raise WrongResult(f"{what} is [{value.spec}], expected [{expected}]")
    return value


def compare(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    if not torch.allclose(got, want, rtol=1e-6, atol=1e-8):
        error = (got - want).abs().max().item()
        raise WrongResult(f"{what} has the right layout but the wrong numbers (max error {error:.3g})")


def check_budgets(spec: Spec, sim: Simulation) -> None:
    if spec.max_forward_collectives is not None:
        count = sum(c.phase == Phase.FORWARD and c.nbytes > 0 for c in sim.comms)
        limit = sim.evaluate(parse_size(spec.max_forward_collectives, allow_group_sizes=True))
        if count > limit:
            raise WrongResult(f"forward uses {count} payload collectives; at most {limit}")
    if spec.max_weight_matmuls is not None and sim.weight_matmuls > spec.max_weight_matmuls:
        raise WrongResult(
            f"forward multiplies by weights {sim.weight_matmuls} times; at most {spec.max_weight_matmuls}"
        )
    if spec.max_comm_bytes is not None:
        budget = sim.evaluate_float(parse_size(spec.max_comm_bytes, allow_group_sizes=True))
        if sim.bytes_sent() > budget + 1e-9:
            raise WrongResult(
                f"moved {sim.bytes_sent():g} B per rank; the budget is {spec.max_comm_bytes} = {budget:g} B"
            )


def stored_tensors(
    value: object, path: str = "self", ancestors: frozenset[int] = frozenset()
) -> Iterator[tuple[str, ShardedTensor]]:
    """Inspect module state, including containers, without counting aliases or following cycles forever.

    Puzzle tensors are leaves: their per-rank torch arrays are simulator internals. Raw torch
    tensors in learner state are rejected, so unpacking .shards cannot evade the storage budget.
    This checks ordinary puzzle code; it is not a sandbox for arbitrary Python programs.
    """
    if value is TODO:
        raise Unfinished(f"{path} is still a TODO")
    if isinstance(value, Pending):
        raise WrongResult(f"{path} is an unchecked {value.made_by} result; call .expect(...) before storing it")
    if isinstance(value, ShardedTensor):
        yield path, value
        return
    if isinstance(value, torch.Tensor):
        raise WrongResult(f"{path} stores a raw torch tensor; keep weights as puzzle tensors so storage can be checked")
    if id(value) in ancestors:
        return
    ancestors = ancestors | {id(value)}
    children: list[tuple[str, object]] = []
    if isinstance(value, dict):
        children = [(f"{path}[{key!r}]", item) for key, item in value.items()]
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = [(f"{path}[{i}]", item) for i, item in enumerate(value)]
    elif isinstance(value, (Module, SimpleNamespace)):
        children = [(f"{path}.{name}", item) for name, item in vars(value).items()]
    elif is_dataclass(value) and not isinstance(value, type):
        children = [(f"{path}.{f.name}", getattr(value, f.name)) for f in fields(value)]
    elif hasattr(value, "__dict__") and not isinstance(value, (type, type(lambda: None))):
        children = [(f"{path}.{name}", item) for name, item in vars(value).items()]
    for child_path, child in children:
        yield from stored_tensors(child, child_path, ancestors)


def check_weight_storage(spec: Spec, module: Module, phase: str) -> None:
    """Bound persistent weights, including full copies hidden beside the shards or cached later."""
    default = sp.Add(
        *(sp.Mul(*(d.size for d in parse_dims(w.spec if isinstance(w, Weight) else w))) for w in spec.params.values())
    )
    formula = (
        parse_size(spec.max_weight_elements, allow_group_sizes=True)
        if spec.max_weight_elements is not None
        else default
    )
    budget = simulation.current().evaluate(formula)
    tensors: dict[int, tuple[str, ShardedTensor]] = {}
    for path, tensor in stored_tensors(module):
        if phase == "on_init" or tensor.is_weight:
            tensors.setdefault(id(tensor), (path, tensor))
    count = sum(t.local_numel for _, t in tensors.values())
    if count > budget:
        held = ", ".join(f"{path}: [{t.spec}] ({t.local_numel})" for path, t in tensors.values())
        raise WrongResult(
            f"after {phase}, stored weights use {count} elements per rank; the storage budget is "
            f"{formula} = {budget}. Shard the weights and release full copies.\n  stored: {held}"
        )


def new_simulation(spec: Spec, config: dict[str, int]) -> Simulation:
    return Simulation(config, spec.dims.split(), spec.require_bytes)


def execute(spec: Spec, cls: type[Module], sim: Simulation, seed: int = 0) -> None:
    """Run one config in `sim`. Raises a PuzzleError (Unfinished, WrongResult, ...) on failure."""
    with simulation.running(sim):
        data = make_data(spec, seed)
        module = cls()
        sim.phase = Phase.INIT
        module.on_init(SimpleNamespace(**data.params))
        check_weight_storage(spec, module, "on_init")
        sim.phase = Phase.FORWARD
        output = returned_tensor(module.forward(*data.inputs.values()), spec.output, "forward's result")
        check_weight_storage(spec, module, "forward")

        grads = {}
        if spec.grads:
            sim.phase = Phase.SETUP
            dy = from_global(data.global_dy, spec.output)
            sim.phase = Phase.BACKWARD
            returned = module.backward(dy)
            check_weight_storage(spec, module, "backward")
            if not isinstance(returned, dict) or set(returned) != set(spec.grads):
                got = sorted(returned) if isinstance(returned, dict) else type(returned).__name__
                raise WrongResult(f"backward must return a dict with keys {sorted(spec.grads)}, got {got}")
            for name, layout in spec.grads.items():
                grads[name] = returned_tensor(returned[name], layout, f"grad '{name}'")

        check_budgets(spec, sim)
        reference = spec.reference(sim.sizes, SimpleNamespace(**data.global_params), **data.global_inputs)
        compare(to_global(output), reference.detach(), "forward's result")
        if grads:
            leaves = [(data.global_params | data.global_inputs)[name] for name in grads]
            for (name, grad), want in zip(grads.items(), torch.autograd.grad(reference, leaves, data.global_dy)):
                compare(to_global(grad), want, f"grad '{name}'")


def run(spec: Spec, cls: type[Module], config: dict[str, int], seed: int = 0) -> Simulation:
    """Run one config in a fresh simulation and return it (for its trace and comms)."""
    sim = new_simulation(spec, config)
    execute(spec, cls, sim, seed)
    return sim
