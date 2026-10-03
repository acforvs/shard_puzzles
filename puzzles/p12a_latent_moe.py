"""Puzzle 12a - LatentMoE: routed experts in a compressed latent space

    router:  top-k on x Wr, weights = softmax over the chosen logits (full width, as in 09a)
    z = x Wdown                                         Wdown: [d l], l = d/alpha
    routed = sum_j weight_j * W2[e_j] gelu(W1[e_j] z)      W1: [E l f]   W2: [E f l]   (experts live in l)
    y = routed Wup + Ws2 gelu(Ws1 x)                     Wup: [l d]; the shared expert stays full width

Mesh ep.  x: [T:ep d].  Return [T:ep d].  Capacity, dispatch and combine as in 09a.

Where do the projections go relative to dispatch and combine so that the all_to_alls carry l-wide
vectors? Then compare with 09a: with E' = alpha*E experts, k' = alpha*k, l = d/alpha and the same capacity factor
(so C is unchanged), what happens to the dispatch volume, the expert FLOPs and the parameters?
(Megatron-Core can also apply a norm before Wup.)

Read: LatentMoE, Elango et al. (arXiv 2601.18089); Megatron-Core `MoELayer` (fc1_latent_proj / fc2_latent_proj).
"""

import torch.nn.functional as F

from puzzles.p09a_moe_expert_parallel import SPEC as PLAIN_MOE
from shardpuzzles import *
from shardpuzzles.ref import routed_experts, top_k_softmax_router


# ---- your code ---------------------------------------------------------------
class LatentMoE(Module):
    def on_init(self, p):
        self.Wr, self.Wdown, self.Wup, self.Ws1, self.Ws2 = p.Wr, p.Wdown, p.Wup, p.Ws1, p.Ws2  # replicated
        self.W1 = shard(p.W1, "E:ep l f")
        self.W2 = shard(p.W2, "E:ep f l")

    def forward(self, x):
        x = x.expect("T:ep d")
        top, expert_ids = topk((x @ self.Wr).expect("T:ep E"), "k", dim=-1)  # route on the full-width token
        weights = softmax(top, dim=-1)
        # TODO: compress, then dispatch l-wide vectors to the experts' ranks
        z = TODO
        buffers, plan = TODO
        buffers = TODO
        h = (buffers @ self.W1).expect("E:ep C f")
        y = (gelu(h) @ self.W2).expect("E:ep C l")
        # TODO: back home, combine in the latent space, then expand once
        y = TODO
        routed_l = TODO
        routed = TODO
        hs = (x @ self.Ws1).expect("T:ep fs")
        shared = (gelu(hs) @ self.Ws2).expect("T:ep d")
        self.x, self.expert_ids, self.weights, self.plan = x, expert_ids, weights, plan  # saved for 12b
        self.buffers, self.h, self.y, self.routed_l, self.hs = buffers, h, y, routed_l, hs
        return routed + shared


# ---- puzzle definition -------------------------------------------------------
def reference(c, p, x):
    expert_ids, weights = top_k_softmax_router(c, p, x)
    return routed_experts(c, x @ p.Wdown, expert_ids, weights, p.W1, p.W2) @ p.Wup + F.gelu(x @ p.Ws1) @ p.Ws2


SPEC = PLAIN_MOE.replace(
    dims="T d l E k C f fs",
    params={"Wr": "d E", "Wdown": "d l", "W1": "E l f", "W2": "E f l", "Wup": "l d", "Ws1": "d fs", "Ws2": "fs d"},
    configs=[  # 09a's configs at alpha = 2: E and k doubled, l = d/2, the same capacity factor
        {"T": 8, "d": 6, "l": 3, "E": 8, "k": 4, "C": 4, "f": 8, "fs": 10, "ep": 2},
        {"T": 12, "d": 4, "l": 2, "E": 16, "k": 4, "C": 8, "f": 6, "fs": 8, "ep": 4},
        {"T": 9, "d": 6, "l": 3, "E": 12, "k": 6, "C": 6, "f": 4, "fs": 6, "ep": 3},
        {"T": 4, "d": 4, "l": 1, "E": 4, "k": 1, "C": 2, "f": 3, "fs": 5, "ep": 2},
        {"T": 3, "d": 4, "l": 2, "E": 2, "k": 2, "C": 2, "f": 3, "fs": 5, "ep": 1},
    ],
    reference=reference,
    max_weight_elements='d*E+2*d*l+2*E*l*f/ep+2*d*fs',
    max_comm_bytes="4*E*C*l/ep*(ep-1)/ep",
)
