"""Puzzle 00b - Partial sums

    y = x @ W        x: [b s d:tp] (each rank holds a slice of the features)    W: [d f]

A layout has a third state besides replicated and split: a *partial sum*.

    "b s f", partial=tp     every tp rank holds a full-size [b s f], but only one term of a sum;
                            the real tensor is the sum of the tp ranks' tensors

Where it comes from. y[b,s,f] = sum over d of x[b,s,d] * W[d,f]. With tp=2 and d split in halves:

    y  =  (sum over rank 0's half of d)  +  (sum over rank 1's half of d)
          computed by rank 0                 computed by rank 1

Each rank's x_r @ W_r already has y's shape, yet neither is y: you need the sum. A partial tensor
looks exactly like a replicated one (same local shape), so the mistake is invisible in the shapes:
that is why you must state it. The rule: an @ or einsum that sums over a dim split across a group
gives a partial sum over that group.

Getting out of it:
    all_reduce(t, group)            partial -> replicated   bytes 2 * 2 * (elements) * (g-1)/g
    reduce_scatter(t, dim, group)   partial -> split        bytes 2 * (elements in) * (g-1)/g

Your task. Return y [b s f], replicated.
  * In on_init, keep the part of W this rank needs (shard is free). What if you keep all of W?
    Try it and read the error.
  * Put show(y) between your two lines: each rank prints different numbers, and the "sum over tp"
    line is the real value.

Read: Megatron-LM (arXiv 1909.08053) Sec. 3 (this is its "row-parallel" linear layer).
"""

from shardpuzzles import *


# ---- your code ---------------------------------------------------------------
class RowParallelLinear(Module):
    def on_init(self, p):
        # TODO: keep the rows of W [d f] that meet this rank's slice of d
        self.W = TODO

    def forward(self, x):
        x = x.expect("b s d:tp")  # the input: each rank holds a slice of the features
        # TODO: y = x @ W, replicated: what does each rank hold after its local matmul? (state the bytes)
        y = TODO
        return y


# ---- puzzle definition -------------------------------------------------------
SPEC = Spec(
    dims="b s d f",
    params={"W": "d f"},
    inputs={"x": "b s d:tp"},
    output="b s f",
    configs=[
        {"b": 2, "s": 3, "d": 4, "f": 5, "tp": 2},
        {"b": 1, "s": 2, "d": 6, "f": 3, "tp": 3},
        {"b": 2, "s": 2, "d": 8, "f": 4, "tp": 4},
    ],
    reference=lambda c, p, x: x @ p.W,
    require_bytes=True,
    max_weight_elements='d*f/tp',
    max_forward_collectives=1,
)
