# shard-puzzles

Small puzzles for reasoning about tensor shapes and how tensors are distributed across ranks.
Write short pieces of PyTorch-style code, state each result's layout, and check the result numerically
on a simulated CPU mesh. No GPU or `torch.distributed` setup is needed.

## Quick start

```bash
uv sync
uv run check.py 0a
uv run check.py 0b
uv run check.py 0c
uv run check.py 1a
uv run check.py 1b
uv run check.py 2a
uv run check.py 2b
uv run check.py 3a
uv run check.py 3b
uv run check.py 4a
uv run check.py 4b
uv run check.py 5a
uv run check.py 5b
uv run check.py 6a
uv run check.py 6b
uv run check.py 7a
uv run check.py 7b
uv run check.py 8a
uv run check.py 8b
uv run check.py 9a
uv run check.py 9b
uv run check.py 10a
uv run check.py 10b
uv run check.py 11a
uv run check.py 11b
uv run check.py 12a
uv run check.py 12b
uv run check.py 12q
uv run check.py 13a
uv run check.py 13b
```

Start with `puzzles/p00a_layouts.py`. Replace the `TODO` blanks and run the checker again.
The introductory exercises cover replicated and split tensors, partial sums, and a data-parallel
linear layer with a manual backward pass. `01a` combines two linear projections into a tensor-parallel
MLP, and `01b` works backward through it. `02a` and `02b` combine tensor and sequence parallelism
in a SwiGLU MLP, covering both forward and backward passes.
`03a` and `03b` apply those layouts to causal grouped-query attention; `04a` and `04b` fuse
the QKV projections and work through the corresponding gradients.
`05a` and `05b` split an embedding table by vocabulary rows; `06a` and `06b` compute
cross-entropy and its gradients while keeping the vocabulary logits split across ranks.
`07a` and `07b` gather sharded weights for FSDP and scatter their gradients; `08a` and `08b`
combine FSDP with tensor parallelism on a two-dimensional mesh.
`09a` and `09b` route tokens to experts with expert parallelism; `10a` and `10b` add
selection bias, sigmoid routing weights, and a shared expert.
`11a` and `11b` split each expert across tensor-parallel ranks and keep the return
buffers narrow while moving them between expert ranks.
`12a` and `12b` compress tokens before dispatch, run routed experts in the latent space,
and expand the combined output while keeping a full-width shared expert.
`12q` combines the latent and shared experts with sigmoid routing in a quiz on layouts and communication.
`13a` and `13b` use Ulysses all-to-all exchanges to switch between sequence and head splits
for attention. Here `cp` names the mesh group that partitions the sequence; the weights are replicated.
`uv run check.py` checks all available exercises.

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
