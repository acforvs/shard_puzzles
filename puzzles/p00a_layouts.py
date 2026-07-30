"""Puzzle 00a - Layouts: replicated and split tensors

Section 00 teaches the notation. Nothing to design yet: 00a split tensors, 00b partial sums,
00c a whole forward and backward.

The setup. A mesh group (here tp) is a set of ranks: simulated processes, each with its own memory.
A *layout* says how the ranks hold one global tensor. Sizes are always the global ones:

    "b s d"       replicated: every rank holds all of it
    "b s:tp d"    split along s: tp rank r holds the r-th of tp equal slices of s
                  (s=4, tp=2: rank 0 holds s[0:2], rank 1 holds s[2:4]; local shape [b, s/tp, d])

What you write. The code reads like single-device PyTorch, but the result of every @, einsum,
reduction (sum, amax, ...) and collective is *pending*: it can't be used until you state its layout.

    y = all_gather(x, "s", tp).expect("b s d")

That statement is the contract. The engine works out the layout itself, and if yours differs the
check fails on that line and says why. Elementwise ops (x * g, gelu(x), ...) can't change a layout,
so their results need no .expect. The given first line of forward states what you receive.

What gets checked: every .expect as it runs; the bytes each rank sent, where a puzzle asks for
bytes= (this one does); and at the end your output's numbers against a single-device reference, for
several sizes and mesh sizes (tp=3 too, so don't hard-code).

Your task. x arrives split along s: each rank holds its own tokens. x: [b s:tp d], g: [d].
Return y = x * g laid out as [b s d:tp]: every rank holds all the tokens, but only its slice of d.
  * all_gather(t, dim, group) makes a split dim whole: [s:tp] -> [s]. Every rank receives the
    tp-1 slices it lacks, so it sends 2 * (elements of the output) * (tp-1)/tp bytes (bf16).
  * shard(t, layout) keeps only this rank's slice of a whole dim. It is free.
  * show(t) prints what every rank holds: its local shape, which slice, and the first values.
    Try show(x) and show(y).

Afterwards: all_to_all(t, "s", "d", tp) does the same re-split in one step. How many bytes would it send?

    uv run check.py 0a      (--hint for hints, --scaffold for the layout of every line)
"""

from shardpuzzles import *


# ---- your code ---------------------------------------------------------------
class Rescale(Module):
    def on_init(self, p):
        self.g = p.g  # replicated: every rank holds all of g

    def forward(self, x):
        x = x.expect("b s:tp d")  # the input: each rank holds its own tokens
        # TODO: y = x * g, then re-split it from s to d (state each collective's layout and bytes)
        y = TODO
        return y


# ---- puzzle definition -------------------------------------------------------
SPEC = Spec(
    dims="b s d",
    params={"g": Weight("d", init="norm")},
    inputs={"x": "b s:tp d"},
    output="b s d:tp",
    configs=[
        {"b": 2, "s": 4, "d": 6, "tp": 2},
        {"b": 1, "s": 6, "d": 3, "tp": 3},
        {"b": 2, "s": 8, "d": 4, "tp": 4},
    ],
    reference=lambda c, p, x: x * p.g,
    require_bytes=True,
    max_forward_collectives=1,
)
