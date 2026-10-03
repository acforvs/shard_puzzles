from shardpuzzles import *
from solutions.p12a_latent_moe import LatentMoE


class LatentMoEBackward(LatentMoE):
    def backward(self, dy):
        dy = dy.expect("T:ep d")
        dWs2 = all_reduce(einsum("t f, t d -> f d", gelu(self.hs), dy).expect("fs d", partial=ep), ep).expect("fs d")
        dhs = gelu_backward((dy @ self.Ws2.transpose(0, 1)).expect("T:ep fs"), self.hs)
        dWs1 = all_reduce(einsum("t d, t f -> d f", self.x, dhs).expect("d fs", partial=ep), ep).expect("d fs")
        dx_shared = (dhs @ self.Ws1.transpose(0, 1)).expect("T:ep d")
        # TODO: back through Wup into the latent space
        dWup = all_reduce(einsum("t l, t d -> l d", self.routed_l, dy).expect("l d", partial=ep), ep).expect("l d")
        droute = (dy @ self.Wup.transpose(0, 1)).expect("T:ep l")
        # END TODO
        # TODO: the routed experts, l wide throughout (09b's pattern)
        dweights = einsum("t k l, t l -> t k", self.y, droute).expect("T:ep k")
        dout = combine_backward(droute.view("T:ep 1 l") * self.weights.view("T:ep k 1"), self.plan).expect("E C:ep l")
        dout = all_to_all(dout, "C", "E", ep).expect("E:ep C l")
        dW2 = einsum("e c f, e c l -> e f l", gelu(self.h), dout).expect("E:ep f l")
        dh = gelu_backward((dout @ self.W2.transpose(-1, -2)).expect("E:ep C f"), self.h)
        dW1 = einsum("e c l, e c f -> e l f", self.buffers, dh).expect("E:ep l f")
        dbuffers = all_to_all((dh @ self.W1.transpose(-1, -2)).expect("E:ep C l"), "E", "C", ep).expect("E C:ep l")
        dz = dispatch_backward(dbuffers, self.plan).expect("T:ep l")
        # END TODO
        # TODO: Wdown and the router (both see the full-width x)
        dWdown = all_reduce(einsum("t d, t l -> d l", self.x, dz).expect("d l", partial=ep), ep).expect("d l")
        dlogits = scatter(softmax_backward(dweights, self.weights, dim=-1), self.expert_ids, dim=-1, size="E")
        dWr = all_reduce(einsum("t d, t e -> d e", self.x, dlogits).expect("d E", partial=ep), ep).expect("d E")
        dx = (dz @ self.Wdown.transpose(0, 1) + dlogits @ self.Wr.transpose(0, 1)).expect("T:ep d")
        # END TODO
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


HINTS = [
    "dWup = einsum(routed_l, dy) (partial over ep -> all_reduce) and droute = dy @ Wup^T [T:ep l]; then 09b's routed backward, in l.",
    "Wr, Wdown, Wup and the shared expert are replicated over ep-split tokens: all_reduce their gradients over ep.",
]
