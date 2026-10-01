"""Puzzle 09a - Mixture of experts with expert parallelism

    logits = x Wr  [T E];   pick the top-k experts per token; weights = softmax over those k logits
    y[t] = sum_j weight[t, j] * W2[e_j] gelu(W1[e_j] x[t])          W1: [E d f]   W2: [E f d]

Mesh ep.  x: [T:ep d] (each rank holds T/ep tokens).  Return [T:ep d].  Each rank stores E/ep experts.

Expert buffers have a fixed capacity: C slots per expert in total, of which each source rank fills
C/ep (first come, first served; overflowing choices are dropped and contribute 0). So the buffer
layout is [E C:ep d]: every rank holds a slice of every expert's queue.
    dispatch(x, ids, "E", "C") -> (buffers [E C:ep d], plan)    combine(y, plan) -> [T:ep k d]
Both are local; moving the buffers to the ranks that own the experts, and back, is yours.

Write each all_to_all's bytes in terms of E, C, d, ep. How do they change if you double k and
keep the capacity factor C*E/(T*k) fixed?

Read: GShard (arXiv 2006.16668) Sec. 2-3; Switch Transformer (arXiv 2101.03961) Sec. 2.2; Ultra-Scale Playbook, "Expert Parallelism".
"""

from shardpuzzles import *
from shardpuzzles.ref import routed_experts, top_k_softmax_router


# ---- your code ---------------------------------------------------------------
class ExpertParallelMoE(Module):
    def on_init(self, p):
        self.Wr = p.Wr  # the router is small and replicated
        # TODO: give each rank E/ep experts
        self.W1 = TODO
        self.W2 = TODO

    def forward(self, x):
        x = x.expect("T:ep d")  # this rank's tokens
        top, expert_ids = topk((x @ self.Wr).expect("T:ep E"), "k", dim=-1)
        weights = softmax(top, dim=-1)
        buffers, plan = dispatch(x, expert_ids, "E", "C")
        # TODO: send every buffer slot to the rank that owns its expert, run the experts, send results back
        buffers = TODO
        h = TODO
        y = TODO
        # TODO: unpack the choices and mix them with the router weights
        y = TODO
        out = TODO
        self.x, self.expert_ids, self.weights, self.plan = x, expert_ids, weights, plan  # saved for 09b
        self.buffers, self.h, self.y = buffers, h, y
        return out


# ---- puzzle definition -------------------------------------------------------
def reference(c, p, x):
    expert_ids, weights = top_k_softmax_router(c, p, x)
    return routed_experts(c, x, expert_ids, weights, p.W1, p.W2)


SPEC = Spec(
    dims="T d E k C f",
    params={"Wr": "d E", "W1": "E d f", "W2": "E f d"},
    inputs={"x": "T:ep d"},
    output="T:ep d",
    configs=[
        {"T": 8, "d": 6, "E": 4, "k": 2, "C": 4, "f": 8, "ep": 2},
        {"T": 12, "d": 4, "E": 8, "k": 2, "C": 8, "f": 6, "ep": 4},
        {"T": 9, "d": 5, "E": 6, "k": 3, "C": 6, "f": 4, "ep": 3},
        {"T": 4, "d": 4, "E": 4, "k": 1, "C": 2, "f": 3, "ep": 2},
        {"T": 3, "d": 3, "E": 2, "k": 2, "C": 2, "f": 4, "ep": 1},
    ],
    reference=reference,
    require_bytes=True,
    max_weight_elements='d*E + 2*E*d*f/ep',
    max_forward_collectives=2,
)
