"""Puzzle 12q - Review quiz: DeepSeek-V3 routing + LatentMoE + a shared expert

10a's routing (sigmoid scores, bias for selection only, normalized weights) with 12a's latent
experts and the full-width shared expert, under expert parallelism.
Mesh ep.  x: [T:ep d].  Return [T:ep d].

The code is complete; replace every "???" with the layout (and the bytes, for collectives).
"""

import torch
import torch.nn.functional as F

from puzzles.p12a_latent_moe import SPEC as LATENT_MOE
from shardpuzzles import *
from shardpuzzles.ref import routed_experts


# ---- your code ---------------------------------------------------------------
class MoEBlock(Module):
    def on_init(self, p):
        self.Wr, self.bias, self.Wdown, self.Wup, self.Ws1, self.Ws2 = p.Wr, p.bias, p.Wdown, p.Wup, p.Ws1, p.Ws2
        self.W1 = shard(p.W1, "E:ep l f")
        self.W2 = shard(p.W2, "E:ep f l")

    def forward(self, x):
        x = x.expect("T:ep d")
        s = sigmoid((x @ self.Wr).expect("???"))
        _, expert_ids = topk(s + self.bias, "k", dim=-1)
        g = gather(s, expert_ids, dim=-1)
        weights = g / sum(g, -1, keepdim=True).expect("???")

        buffers, plan = dispatch((x @ self.Wdown).expect("???"), expert_ids, "E", "C")
        buffers = all_to_all(buffers.expect("E C:ep l"), "C", "E", ep).expect(
            "???", bytes="???"
        )
        y = (gelu((buffers @ self.W1).expect("???")) @ self.W2).expect("???")
        y = all_to_all(y, "E", "C", ep).expect("???", bytes="???")
        routed = (
            sum(combine(y, plan).expect("???") * weights.view("T:ep k 1"), "k").expect("???") @ self.Wup
        ).expect("???")

        shared = (gelu((x @ self.Ws1).expect("???")) @ self.Ws2).expect("???")
        return routed + shared


# ---- puzzle definition -------------------------------------------------------
def reference(c, p, x):
    s = torch.sigmoid(x @ p.Wr)
    expert_ids = (s + p.bias).topk(c["k"], dim=-1).indices
    g = s.gather(-1, expert_ids)
    routed = routed_experts(c, x @ p.Wdown, expert_ids, g / g.sum(-1, keepdim=True), p.W1, p.W2)
    return routed @ p.Wup + F.gelu(x @ p.Ws1) @ p.Ws2


SPEC = LATENT_MOE.replace(
    params={**LATENT_MOE.params, "bias": "E"}, configs=LATENT_MOE.configs[:2], reference=reference,
    max_weight_elements='d*E+E+2*d*l+2*E*l*f/ep+2*d*fs',
)
