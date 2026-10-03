"""Puzzle 12b - Backward: LatentMoE

The forward is 12a's reference solution, imported below. It saves self.x, self.expert_ids,
self.weights, self.plan, self.buffers [E:ep C l], self.h [E:ep C f], self.y [T:ep k l],
self.routed_l [T:ep l] (the latent before Wup) and self.hs [T:ep fs]. dy: [T:ep d]. Return
    {"x": [T:ep d], "Wr": [d E], "Wdown": [d l], "W1": [E:ep l f], "W2": [E:ep f l], "Wup": [l d],
     "Ws1": [d fs], "Ws2": [fs d]}.

The routed path is 09b's, run in the latent space between Wdown and Wup. Do the backward
all_to_alls also carry l-wide vectors? Which gradients come from replicated weights applied to
ep-split tokens?

Helpers: gelu_backward, softmax_backward, scatter, combine_backward, dispatch_backward.
"""

from puzzles.p12a_latent_moe import SPEC as FORWARD
from shardpuzzles import *
from solutions.p12a_latent_moe import LatentMoE  # 12a's forward; saves the tensors listed above


# ---- your code ---------------------------------------------------------------
class LatentMoEBackward(LatentMoE):
    def backward(self, dy):
        dy = dy.expect("T:ep d")
        dWs2 = all_reduce(einsum("t f, t d -> f d", gelu(self.hs), dy).expect("fs d", partial=ep), ep).expect("fs d")
        dhs = gelu_backward((dy @ self.Ws2.transpose(0, 1)).expect("T:ep fs"), self.hs)
        dWs1 = all_reduce(einsum("t d, t f -> d f", self.x, dhs).expect("d fs", partial=ep), ep).expect("d fs")
        dx_shared = (dhs @ self.Ws1.transpose(0, 1)).expect("T:ep d")
        # TODO: back through Wup into the latent space
        dWup = TODO
        droute = TODO
        # TODO: the routed experts, l wide throughout (09b's pattern)
        dweights = TODO
        dout = TODO
        dW2 = TODO
        dh = TODO
        dW1 = TODO
        dbuffers = TODO
        dz = TODO
        # TODO: Wdown and the router (both see the full-width x)
        dWdown = TODO
        dlogits = TODO
        dWr = TODO
        dx = TODO
        return {
            "x": dx + dx_shared,
            "Wr": dWr,
            "Wdown": dWdown,
            "W1": dW1,
            "W2": dW2,
            "Wup": dWup,
            "Ws1": dWs1,
            "Ws2": dWs2,
        }


# ---- puzzle definition -------------------------------------------------------
SPEC = FORWARD.replace(
    grads={
        "x": "T:ep d",
        "Wr": "d E",
        "Wdown": "d l",
        "W1": "E:ep l f",
        "W2": "E:ep f l",
        "Wup": "l d",
        "Ws1": "d fs",
        "Ws2": "fs d",
    },
    require_bytes=False,
    max_comm_bytes="(8*E*C*l/ep+8*d*(l+fs)+4*d*E)*(ep-1)/ep",
)
