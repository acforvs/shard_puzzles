"""Puzzle 10a - DeepSeek-V3-style routing + a shared expert, under expert parallelism

09a with two changes:
    routing:  s = sigmoid(x Wr) [T E]; choose the top-k experts by s + bias (bias: [E], aux-loss-free
              balancing, it only steers SELECTION); weights = s[chosen] / sum(s[chosen]) (no bias)
    shared:   y = sum_j weights_j * routed_j(x)  +  Ws2 gelu(Ws1 x)          Ws1: [d fs]   Ws2: [fs d]

Mesh ep.  x: [T:ep d].  Return [T:ep d].  Routed experts and capacity as in 09a.

Does the shared expert need any communication? It doesn't depend on the all_to_all, so what
could it overlap with?

Ops: sigmoid, gather(t, ids, dim) (torch.gather), sum(t, dim, keepdim).

Read: DeepSeek-V3 (arXiv 2412.19437) Sec. 2.1.2; DeepSeekMoE (arXiv 2401.06066) Sec. 3.2.
"""

import torch
import torch.nn.functional as F

from puzzles.p09a_moe_expert_parallel import SPEC as PLAIN_MOE
from shardpuzzles import *
from shardpuzzles.ref import routed_experts


# ---- your code ---------------------------------------------------------------
class SharedExpertMoE(Module):
    def on_init(self, p):
        self.Wr, self.bias = p.Wr, p.bias
        self.W1 = shard(p.W1, "E:ep d f")
        self.W2 = shard(p.W2, "E:ep f d")
        # TODO: the shared expert's weights
        self.Ws1, self.Ws2 = TODO

    def forward(self, x):
        x = x.expect("T:ep d")
        # TODO: sigmoid scores, selection with the bias, weights without it
        s = TODO
        _, expert_ids = TODO
        g = TODO
        g_sum = TODO
        weights = TODO
        buffers, plan = dispatch(x, expert_ids, "E", "C")
        buffers = all_to_all(buffers.expect("E C:ep d"), "C", "E", ep).expect(
            "E:ep C d", bytes="2*E*C/ep*d * (ep-1)/ep"
        )
        # TODO: the shared expert (no dependency on the all_to_all, so it could overlap with it)
        hs = TODO
        shared = TODO
        h = (buffers @ self.W1).expect("E:ep C f")
        y = all_to_all((gelu(h) @ self.W2).expect("E:ep C d"), "E", "C", ep).expect(
            "E C:ep d", bytes="2*E*C/ep*d * (ep-1)/ep"
        )
        y = combine(y, plan).expect("T:ep k d")
        self.x, self.s, self.expert_ids, self.g_sum, self.weights = x, s, expert_ids, g_sum, weights  # for 10b
        self.plan, self.buffers, self.h, self.y, self.hs = plan, buffers, h, y, hs
        return sum(y * weights.view("T:ep k 1"), "k").expect("T:ep d") + shared


# ---- puzzle definition -------------------------------------------------------
def reference(c, p, x):
    s = torch.sigmoid(x @ p.Wr)
    expert_ids = (s + p.bias).topk(c["k"], dim=-1).indices
    g = s.gather(-1, expert_ids)
    return routed_experts(c, x, expert_ids, g / g.sum(-1, keepdim=True), p.W1, p.W2) + F.gelu(x @ p.Ws1) @ p.Ws2


SPEC = PLAIN_MOE.replace(
    dims="T d E k C f fs",
    params={"Wr": "d E", "bias": "E", "W1": "E d f", "W2": "E f d", "Ws1": "d fs", "Ws2": "fs d"},
    configs=[
        {"T": 8, "d": 6, "E": 4, "k": 2, "C": 4, "f": 8, "fs": 10, "ep": 2},
        {"T": 12, "d": 4, "E": 8, "k": 2, "C": 8, "f": 6, "fs": 8, "ep": 4},
        {"T": 9, "d": 5, "E": 6, "k": 3, "C": 6, "f": 4, "fs": 6, "ep": 3},
        {"T": 4, "d": 4, "E": 4, "k": 1, "C": 2, "f": 3, "fs": 5, "ep": 2},
        {"T": 3, "d": 3, "E": 2, "k": 2, "C": 2, "f": 4, "fs": 5, "ep": 1},
    ],
    reference=reference,
    max_weight_elements='d*E + E + 2*E*d*f/ep + 2*d*fs',
)
