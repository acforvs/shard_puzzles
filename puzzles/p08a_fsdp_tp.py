"""Puzzle 08a - 2-D parallelism: FSDP x tensor parallel

    y = gelu(x @ W1) @ W2          W1: [d f]   W2: [f d]

Mesh tp x dp (every rank has a tp coordinate and a dp coordinate).
x: [b:dp s d], replicated across tp.  Return [b:dp s d], replicated across tp.

Combine 01a (Megatron TP inside a node) with 07a (FSDP across nodes): every rank stores
1/(tp*dp) of each weight. Which group splits which dim? For each collective, which group does
it run over, and does its size depend on the batch?

Read: Ultra-Scale Playbook, "5D parallelism"; torchtitan's FSDP2 + TP recipe.
"""

import torch.nn.functional as F

from shardpuzzles import *


# ---- your code ---------------------------------------------------------------
class FSDPTPMLP(Module):
    def on_init(self, p):
        # TODO: two split dims per weight
        self.W1 = TODO
        self.W2 = TODO

    def forward(self, x):
        x = x.expect("b:dp s d")
        # TODO: gather over dp, then puzzle 01a over tp
        W1 = TODO
        h = TODO
        W2 = TODO
        y = TODO
        self.x, self.h = x, h  # saved for the backward (08b)
        return y


# ---- puzzle definition -------------------------------------------------------
SPEC = Spec(
    dims="b s d f",
    params={"W1": "d f", "W2": "f d"},
    inputs={"x": "b:dp s d"},
    output="b:dp s d",
    configs=[
        {"b": 4, "s": 3, "d": 8, "f": 16, "tp": 2, "dp": 2},
        {"b": 2, "s": 2, "d": 6, "f": 24, "tp": 4, "dp": 2},
        {"b": 6, "s": 2, "d": 6, "f": 12, "tp": 2, "dp": 3},
    ],
    reference=lambda c, p, x: F.gelu(x @ p.W1) @ p.W2,
    require_bytes=True,
    max_weight_elements='2*d*f/(dp*tp)',
    max_forward_collectives=3,
)
