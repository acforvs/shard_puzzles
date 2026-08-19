"""Puzzle 01a - MLP with tensor parallelism

    y = gelu(x @ W1) @ W2          W1: [d f]   W2: [f d]

Mesh tp.  x: [b s d], replicated on every rank.  Return y: [b s d], replicated.

Split the weights so each rank stores 1/tp of them and the layer needs a single collective.
Which weight is split by columns, which by rows, and where does a partial sum appear?

Read: Megatron-LM (arXiv 1909.08053) Sec. 3, Fig. 3a; Ultra-Scale Playbook, "Tensor Parallelism".
"""

import torch.nn.functional as F

from shardpuzzles import *


# ---- your code ---------------------------------------------------------------
class MLP(Module):
    def on_init(self, p):
        # TODO: keep this rank's piece of W1 [d f] and W2 [f d]
        self.W1 = TODO
        self.W2 = TODO

    def forward(self, x):
        x = x.expect("b s d")
        # TODO: compute y with one collective (state each layout, and the collective's bytes)
        h = TODO
        y = TODO
        return y


# ---- puzzle definition -------------------------------------------------------
SPEC = Spec(
    dims="b s d f",
    params={"W1": "d f", "W2": "f d"},
    inputs={"x": "b s d"},
    output="b s d",
    configs=[
        {"b": 2, "s": 4, "d": 8, "f": 16, "tp": 2},
        {"b": 1, "s": 3, "d": 6, "f": 24, "tp": 4},
        {"b": 2, "s": 5, "d": 4, "f": 12, "tp": 3},
    ],
    reference=lambda c, p, x: F.gelu(x @ p.W1) @ p.W2,
    require_bytes=True,
    max_weight_elements="2*d*f/tp",
    max_forward_collectives=1,
)
