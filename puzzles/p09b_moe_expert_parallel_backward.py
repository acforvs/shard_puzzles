"""Puzzle 09b - Backward: mixture of experts with expert parallelism

The forward is 09a's reference solution, imported below. It saves self.x, self.expert_ids,
self.weights [T:ep k], self.plan, the expert inputs self.buffers [E:ep C d], self.h [E:ep C f]
(pre-gelu) and self.y [T:ep k d] (per-choice outputs). dy: [T:ep d].
Return {"x": [T:ep d], "Wr": [d E], "W1": [E:ep d f], "W2": [E:ep f d]}.

Walk the forward in reverse. The weighted sum over k gives two gradients (into the expert outputs
and into the router weights). combine_backward / dispatch_backward undo the local packing; every
all_to_all is mirrored. The router's gradient reaches the logits only through the k chosen
entries: scatter them back into [T E]. Dropped choices contribute no expert-output gradient.

Helpers: gelu_backward, softmax_backward, scatter(src, ids, dim, "E"), combine_backward(dy, plan),
dispatch_backward(dbuf, plan).
"""

from puzzles.p09a_moe_expert_parallel import SPEC as FORWARD
from shardpuzzles import *
from solutions.p09a_moe_expert_parallel import ExpertParallelMoE  # 09a's forward; saves the tensors listed above


# ---- your code ---------------------------------------------------------------
class ExpertParallelMoEBackward(ExpertParallelMoE):
    def backward(self, dy):
        dy = dy.expect("T:ep d")
        # TODO: out = sum_k w_k * y_k: the gradient into the router weights, and into each choice's output
        dweights = TODO
        dy_choices = TODO
        # TODO: mirror the return trip, run the experts backward, mirror the dispatch trip
        dout = TODO
        dW2 = TODO
        dh = TODO
        dW1 = TODO
        dbuffers = TODO
        dx = TODO
        # TODO: the router: softmax over the chosen logits, scattered back into [T E]
        dlogits = TODO
        dWr = TODO
        dx = TODO
        return {"x": dx, "Wr": dWr, "W1": dW1, "W2": dW2}


# ---- puzzle definition -------------------------------------------------------
SPEC = FORWARD.replace(grads={"x": "T:ep d", "Wr": "d E", "W1": "E:ep d f", "W2": "E:ep f d"}, require_bytes=False)
