"""Puzzle 10b - Backward: DeepSeek-V3-style routing + a shared expert

The forward is 10a's reference solution, imported below. It saves self.x, self.s (sigmoid scores),
self.expert_ids, self.g_sum [T:ep 1], self.weights, self.plan, self.buffers [E:ep C d], self.h
[E:ep C f], self.y [T:ep k d] and self.hs [T:ep fs] (the shared expert's pre-gelu). dy: [T:ep d]. Return
    {"x": [T:ep d], "Wr": [d E], "W1": [E:ep d f], "W2": [E:ep f d], "Ws1": [d fs], "Ws2": [fs d]}
(the routing bias is updated by a balancing rule, not by gradients).

The routing weights are w = g / sum(g) with g = sigmoid(logits) at the chosen experts:
    dg = (dw - sum_j dw_j w_j) / sum(g)      then scatter dg into [T E] and apply sigmoid_backward.
The shared expert needs no communication in forward. What about its weight gradients?

Helpers: gelu_backward, sigmoid_backward(dy, s), scatter, combine_backward, dispatch_backward.
"""

from puzzles.p10a_moe_shared_expert import SPEC as FORWARD
from shardpuzzles import *
from solutions.p10a_moe_shared_expert import SharedExpertMoE  # 10a's forward; saves the tensors listed above


# ---- your code ---------------------------------------------------------------
class SharedExpertMoEBackward(SharedExpertMoE):
    def backward(self, dy):
        dy = dy.expect("T:ep d")
        # TODO: the shared expert: local tokens, replicated weights
        dWs2 = TODO
        dhs = TODO
        dWs1 = TODO
        dx_shared = TODO
        dweights = einsum("t k d, t d -> t k", self.y, dy).expect("T:ep k")
        dout = combine_backward(dy.view("T:ep 1 d") * self.weights.view("T:ep k 1"), self.plan).expect("E C:ep d")
        dout = all_to_all(dout, "C", "E", ep).expect("E:ep C d")
        dW2 = einsum("e c f, e c d -> e f d", gelu(self.h), dout).expect("E:ep f d")
        dh = gelu_backward((dout @ self.W2.transpose(-1, -2)).expect("E:ep C f"), self.h)
        dW1 = einsum("e c d, e c f -> e d f", self.buffers, dh).expect("E:ep d f")
        dbuffers = all_to_all((dh @ self.W1.transpose(-1, -2)).expect("E:ep C d"), "E", "C", ep).expect("E C:ep d")
        dx = dispatch_backward(dbuffers, self.plan).expect("T:ep d")
        # TODO: the router: w = g / sum(g), g = sigmoid(logits)[ids] (the bias only picked the ids)
        dg = TODO
        dlogits = TODO
        dWr = TODO
        dx_router = TODO
        return {"x": dx + dx_router + dx_shared, "Wr": dWr, "W1": dW1, "W2": dW2, "Ws1": dWs1, "Ws2": dWs2}


# ---- puzzle definition -------------------------------------------------------
SPEC = FORWARD.replace(
    grads={"x": "T:ep d", "Wr": "d E", "W1": "E:ep d f", "W2": "E:ep f d", "Ws1": "d fs", "Ws2": "fs d"},
    require_bytes=False,
)
