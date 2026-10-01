from shardpuzzles import *
from solutions.p10a_moe_shared_expert import SharedExpertMoE


class SharedExpertMoEBackward(SharedExpertMoE):
    def backward(self, dy):
        dy = dy.expect("T:ep d")
        # TODO: the shared expert: local tokens, replicated weights
        dWs2 = all_reduce(einsum("t f, t d -> f d", gelu(self.hs), dy).expect("fs d", partial=ep), ep).expect("fs d")
        dhs = gelu_backward((dy @ self.Ws2.transpose(0, 1)).expect("T:ep fs"), self.hs)
        dWs1 = all_reduce(einsum("t d, t f -> d f", self.x, dhs).expect("d fs", partial=ep), ep).expect("d fs")
        dx_shared = (dhs @ self.Ws1.transpose(0, 1)).expect("T:ep d")
        # END TODO
        dweights = einsum("t k d, t d -> t k", self.y, dy).expect("T:ep k")
        dout = combine_backward(dy.view("T:ep 1 d") * self.weights.view("T:ep k 1"), self.plan).expect("E C:ep d")
        dout = all_to_all(dout, "C", "E", ep).expect("E:ep C d")
        dW2 = einsum("e c f, e c d -> e f d", gelu(self.h), dout).expect("E:ep f d")
        dh = gelu_backward((dout @ self.W2.transpose(-1, -2)).expect("E:ep C f"), self.h)
        dW1 = einsum("e c d, e c f -> e d f", self.buffers, dh).expect("E:ep d f")
        dbuffers = all_to_all((dh @ self.W1.transpose(-1, -2)).expect("E:ep C d"), "E", "C", ep).expect("E C:ep d")
        dx = dispatch_backward(dbuffers, self.plan).expect("T:ep d")
        # TODO: the router: w = g / sum(g), g = sigmoid(logits)[ids] (the bias only picked the ids)
        dg = (dweights - sum(dweights * self.weights, -1, keepdim=True).expect("T:ep 1")) / self.g_sum
        dlogits = sigmoid_backward(scatter(dg, self.expert_ids, dim=-1, size="E"), self.s)
        dWr = all_reduce(einsum("t d, t e -> d e", self.x, dlogits).expect("d E", partial=ep), ep).expect("d E")
        dx_router = (dlogits @ self.Wr.transpose(0, 1)).expect("T:ep d")
        # END TODO
        return {"x": dx + dx_router + dx_shared, "Wr": dWr, "W1": dW1, "W2": dW2, "Ws1": dWs1, "Ws2": dWs2}


HINTS = [
    "The shared expert's weights are replicated over ep-split tokens, so their gradients are partial over ep: all_reduce.",
    "dg = (dw - sum(dw * w)) / sum(g); dlogits = sigmoid_backward(scatter(dg, ids, -1, 'E'), s).",
]
