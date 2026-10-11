# shard-puzzles

Small puzzles for reasoning about tensor shapes and how tensors are distributed across ranks.
Write short pieces of PyTorch-style code, state each result's layout, and check the result numerically
on a simulated CPU mesh. No GPU or `torch.distributed` setup is needed.

## Quick start

```bash
uv sync
uv run check.py 0a
```

Open [puzzles/p00a_layouts.py](puzzles/p00a_layouts.py), replace the `TODO` blanks, and run the
checker again. Continue through the numbered files in `puzzles/`.

Run `uv run check.py 1` to check a whole section, or `uv run check.py` to check every exercise.
Hints, scaffolds, and worked answers are described under [Getting help](#getting-help).

## Layouts

A layout describes one global tensor and what each rank holds. Dimension sizes are always global.
For example, with `s=4` and `tp=2`, `s:tp` gives rank 0 positions `[0:2]` and rank 1 positions `[2:4]`.

| Layout | What each rank holds | How to reconstruct the global tensor |
|---|---|---|
| `"b s d"` | the whole tensor | use any replica |
| `"b s:tp d"` | its slice of the sequence, shape `[b, s/tp, d]` | concatenate the slices |
| `"b s d", partial=tp` | a full-size tensor containing one contribution | sum the contributions |

A partial sum can have exactly the same local shape as a replicated tensor. The values mean
different things: if a matmul contracts an axis split across ranks, each rank sums only its own
slice. A reduction is needed to obtain the complete value.

## Writing an answer

Each exercise defines a `Module`. `on_init(p)` receives replicated parameters, `forward(...)`
computes the output, and an exercise with a backward pass returns gradients from `backward(dy)`.

The result of a matmul, `einsum`, reduction or collective is pending until you state its layout:

```python
result = operation(...).expect("b s d")
```

For a partial sum, add a mesh group such as `partial=tp`, `partial=dp`, `partial=ep`, or `partial=cp`.
For a collective, the exercise may also require
`bytes="..."`. The engine derives the layout independently and checks your statement. Elementwise
operations preserve the layout and do not need another `.expect(...)`.

The checker also compares output values with a single-device reference and gradients with autograd,
using several tensor and mesh sizes. Storage budgets enforce weight-sharding requirements, and
collective limits enforce the exercise's communication constraints. Storage checks cover persistent
module weights, including copies in containers; they do not model peak temporary memory.

Put `show(tensor)` in your code to inspect each rank's local shape, slice and values. A passing run
prints a trace of layouts and communication. Failed and incomplete runs hide computed byte formulas;
a byte mismatch shows only your submitted formula.

## Getting help

```bash
uv run check.py 0b --hint
uv run check.py 0b --hint 2
uv run check.py 0b --scaffold
uv run check.py 0b --solution
```

Hints are progressive. A scaffold adds layout checkpoints to the blanks, while keeping byte answers
hidden. `--solution` runs the worked answer from `solutions/`. Add `-q` to omit the trace.
In a quiz, the operations are given: fill in the `"???"` layouts and byte counts.

You can stop an attempt with `raise NotImplementedError` to inspect the trace so far.

## Communication model

The byte model counts payload sent per rank, assuming bf16 elements and ring/pairwise algorithms.
Let `g` be the group size.

| Collective | Bytes sent per rank | `n` counts |
|---|---|---|
| `all_gather` | `2*n*(g-1)/g` | elements in the gathered output |
| `reduce_scatter` | `2*n*(g-1)/g` | elements in the unreduced input |
| `all_reduce` | `4*n*(g-1)/g` | elements in the tensor |
| `all_to_all` | `2*n*(g-1)/g` | elements in the input |

`shard(t, layout)` keeps a local slice of a replicated tensor and sends no data. A local reshape
or transpose also sends no data; a reshape cannot move sharding between axes.

## Weight gradients

For `y = x @ W`, every token uses the same `W`, so its gradient sums every token's contribution:

```text
dW[d, f] = sum over b and s of x[b, s, d] * dy[b, s, f]
```

`einsum("b s d, b s f -> d f", x, dy)` expresses that contraction. When the batch is split over
`dp`, each rank holds only its own contribution to `dW`.

The checker treats replicated `dy` as one logical gradient copied across ranks, rather than
independent contributions from each replica. Keep that convention in mind when reasoning backward
through a collective.

## Checks

```bash
uv run pytest
uvx ruff check .
uvx pyright
```

Tests cover reference values and gradients, unfinished stubs, generated scaffolds, storage and
communication contracts, and error reporting. Learners can use `check.py` to validate their answers.
