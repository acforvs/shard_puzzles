"""Puzzle 11b - Backward: expert tensor parallelism (tp x ep)

The forward is 11a's reference solution, imported below; it ends with an all_gather over tp of the
d-split output. It saves self.x, self.expert_ids, self.weights, self.plan, self.buffers [E:ep C d],
self.h [E:ep C f:tp] and self.y [T:ep k d:tp] (per-choice outputs, still d-split).
dy: [T:ep d], replicated across tp. Return
    {"x": [T:ep d], "Wr": [d E], "W1": [E:ep d f:tp], "W2": [E:ep f:tp d]}.

What's the backward of an all_gather when the incoming gradient is already the same on every rank
(not a partial sum)? Mirror the forward: keep the all_to_alls d/tp wide. The router runs
redundantly on every tp rank, so which group does its gradient reduce over?

Helpers: gelu_backward, softmax_backward, scatter, combine_backward, dispatch_backward.
"""

from puzzles.p11a_expert_tensor_parallel import SPEC as FORWARD
from shardpuzzles import *
from solutions.p11a_expert_tensor_parallel import (
    ExpertTensorParallelMoE,  # 11a's forward; saves the tensors listed above
)


# ---- your code ---------------------------------------------------------------
class ExpertTensorParallelMoEBackward(ExpertTensorParallelMoE):
    def backward(self, dy):
        dy = dy.expect("T:ep d")
        # TODO: the final all_gather's backward (dy is replicated), then the weighted sum's
        dy = TODO
        dweights = TODO
        dy_choices = TODO
        # TODO: back through the return trip: d/tp wide over ep, then whole d for the experts
        dout = TODO
        dW2 = einsum("e c f, e c d -> e f d", gelu(self.h), dout).expect("E:ep f:tp d")
        dh = gelu_backward((dout @ self.W2.transpose(-1, -2)).expect("E:ep C f:tp"), self.h)
        dW1 = einsum("e c d, e c f -> e d f", self.buffers, dh).expect("E:ep d f:tp")
        # TODO: the expert input's gradient is partial over tp: reduce it so the trip home is d/tp wide
        dbuffers = TODO
        dx = TODO
        dlogits = scatter(softmax_backward(dweights, self.weights, dim=-1), self.expert_ids, dim=-1, size="E")
        dWr = all_reduce(einsum("t d, t e -> d e", self.x, dlogits).expect("d E", partial=ep), ep).expect("d E")
        dx = dx + (dlogits @ self.Wr.transpose(0, 1)).expect("T:ep d")
        return {"x": dx, "Wr": dWr, "W1": dW1, "W2": dW2}


# ---- puzzle definition -------------------------------------------------------
SPEC = FORWARD.replace(
    grads={"x": "T:ep d", "Wr": "d E", "W1": "E:ep d f:tp", "W2": "E:ep f:tp d"},
    require_bytes=False,
    max_comm_bytes=(
        "2*E*C*d/ep*(ep-1)/ep*(1+3/tp)"
        "+(6*E*C*d/ep+4*T*d/ep+4*T*k/ep)*(tp-1)/tp"
        "+4*d*E*(ep-1)/ep"
    ),
)
