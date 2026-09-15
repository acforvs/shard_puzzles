"""Puzzle 07a - Fully sharded data parallel (ZeRO-3), forward

    y = gelu(x @ W1) @ W2          W1: [d f]   W2: [f d]

Mesh dp.  x: [b:dp s d] (each rank has its own micro-batch).  Return [b:dp s d].

Plain data parallelism replicates the weights (puzzle 00). FSDP stores only 1/dp of every weight
and gathers each one just before it's used. How does the traffic scale with the batch size, and with dp?

Read: ZeRO (arXiv 1910.02054) Sec. 5; PyTorch FSDP (arXiv 2304.11277); Ultra-Scale Playbook, "ZeRO-3".
"""

import torch.nn.functional as F

from shardpuzzles import *


# ---- your code ---------------------------------------------------------------
class FSDPMLP(Module):
    def on_init(self, p):
        # TODO: store 1/dp of each weight
        self.W1 = TODO
        self.W2 = TODO

    def forward(self, x):
        x = x.expect("b:dp s d")
        # TODO: gather each weight right before you use it
        W1 = TODO
        h = TODO
        W2 = TODO
        y = TODO
        self.x, self.h = x, h  # saved for the backward (07b); the gathered weights are freed
        return y


# ---- puzzle definition -------------------------------------------------------
SPEC = Spec(
    dims="b s d f",
    params={"W1": "d f", "W2": "f d"},
    inputs={"x": "b:dp s d"},
    output="b:dp s d",
    configs=[
        {"b": 4, "s": 3, "d": 8, "f": 16, "dp": 2},
        {"b": 4, "s": 2, "d": 8, "f": 12, "dp": 4},
        {"b": 6, "s": 2, "d": 6, "f": 9, "dp": 3},
    ],
    reference=lambda c, p, x: F.gelu(x @ p.W1) @ p.W2,
    require_bytes=True,
    max_weight_elements='2*d*f/dp',
    max_forward_collectives=2,
)
