"""Puzzle 11a - Expert tensor parallelism: MoE on a tp x ep mesh

09a, but every expert is itself split Megatron-style across tp (W1 by columns, W2 by rows):
each rank stores E/ep experts x 1/tp of each.

Mesh tp x ep.  x: [T:ep d], replicated across tp.  Return [T:ep d], replicated across tp.

After the row-parallel W2 the expert output is partial over tp. The obvious fix is an all_reduce
before sending results back. Can you reduce_scatter instead, so the return all_to_all carries
d/tp-wide vectors, and gather only at the very end? Compare the total bytes.
Megatron-Core recommends ETP=1 for fine-grained MoE. Looking at the GEMM shapes and the extra
collectives, why?

Read: Megatron-Core MoE README ("expert tensor parallelism", "Parallel Folding"); Megatron-LM Sec. 3.
"""

from puzzles.p09a_moe_expert_parallel import SPEC as PLAIN_MOE
from shardpuzzles import *


# ---- your code ---------------------------------------------------------------
class ExpertTensorParallelMoE(Module):
    def on_init(self, p):
        self.Wr = p.Wr
        # TODO: experts split over ep, and each expert split Megatron-style over tp
        self.W1 = TODO
        self.W2 = TODO

    def forward(self, x):
        x = x.expect("T:ep d")
        top, expert_ids = topk((x @ self.Wr).expect("T:ep E"), "k", dim=-1)
        weights = softmax(top, dim=-1)
        buffers, plan = dispatch(x, expert_ids, "E", "C")
        buffers = all_to_all(buffers.expect("E C:ep d"), "C", "E", ep).expect(
            "E:ep C d", bytes="2*E*C/ep*d * (ep-1)/ep"
        )
        # TODO: the tp-split experts, then the return trip with d/tp-wide vectors
        h = TODO
        y = TODO
        # TODO: combine on d:tp, gather d at the very end
        y = TODO
        out = TODO
        self.x, self.expert_ids, self.weights, self.plan = x, expert_ids, weights, plan  # saved for 11b
        self.buffers, self.h, self.y = buffers, h, y
        return out


# ---- puzzle definition -------------------------------------------------------
SPEC = PLAIN_MOE.replace(
    configs=[
        {"T": 8, "d": 6, "E": 4, "k": 2, "C": 4, "f": 8, "tp": 2, "ep": 2},
        {"T": 12, "d": 4, "E": 6, "k": 2, "C": 6, "f": 6, "tp": 2, "ep": 3},
        {"T": 8, "d": 8, "E": 4, "k": 2, "C": 4, "f": 12, "tp": 4, "ep": 2},
        {"T": 4, "d": 4, "E": 4, "k": 1, "C": 2, "f": 6, "tp": 2, "ep": 2},
        {"T": 3, "d": 4, "E": 2, "k": 2, "C": 2, "f": 6, "tp": 2, "ep": 1},
        {"T": 4, "d": 3, "E": 4, "k": 2, "C": 2, "f": 5, "tp": 1, "ep": 2},
    ],
    max_weight_elements='d*E+2*E*d*f/(ep*tp)',
    max_forward_collectives=4,
    max_comm_bytes=(
        "2*E*C*d/ep*(ep-1)/ep*(1+1/tp)"
        "+2*(E*C+T)*d/ep*(tp-1)/tp"
    ),
)
