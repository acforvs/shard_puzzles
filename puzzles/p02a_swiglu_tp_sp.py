"""Puzzle 02a - SwiGLU MLP with tensor + sequence parallelism

    y = (silu(x @ Wgate) * (x @ Wup)) @ Wdown          Wgate, Wup: [d f]   Wdown: [f d]

Mesh tp.  x arrives sequence-parallel: [b s:tp d].  Return [b s:tp d].

With sequence parallelism, the activations between blocks are split along s, so norms and
dropout only touch 1/tp of the tokens. 01a's all-reduce turns into two collectives. Which two,
and where do they go? Compare the bytes with 01a.

Read: Korthikanti et al., "Reducing Activation Recomputation" (arXiv 2205.05198) Sec. 4.2.2.
"""

import torch.nn.functional as F

from shardpuzzles import *


# ---- your code ---------------------------------------------------------------
class SwiGLU(Module):
    def on_init(self, p):
        # TODO: shard Wgate, Wup [d f] and Wdown [f d]
        self.Wgate = TODO
        self.Wup = TODO
        self.Wdown = TODO

    def forward(self, x):
        x = x.expect("b s:tp d")
        # TODO: every rank needs every token for its slice of f: gather, then the gate and up projections
        x = TODO
        gate = TODO
        up = TODO
        act = silu(gate) * up
        # TODO: down-projection, and back to sequence-parallel
        y = TODO
        self.x, self.gate, self.up, self.act = x, gate, up, act  # saved for the backward (02b)
        return y


# ---- puzzle definition -------------------------------------------------------
SPEC = Spec(
    dims="b s d f",
    params={"Wgate": "d f", "Wup": "d f", "Wdown": "f d"},
    inputs={"x": "b s:tp d"},
    output="b s:tp d",
    configs=[
        {"b": 2, "s": 4, "d": 8, "f": 16, "tp": 2},
        {"b": 1, "s": 8, "d": 6, "f": 24, "tp": 4},
        {"b": 2, "s": 6, "d": 4, "f": 12, "tp": 3},
    ],
    reference=lambda c, p, x: (F.silu(x @ p.Wgate) * (x @ p.Wup)) @ p.Wdown,
    require_bytes=True,
    max_weight_elements='3*d*f/tp',
    max_forward_collectives=2,
)
