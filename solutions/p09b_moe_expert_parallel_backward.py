from shardpuzzles import *
from solutions.p09a_moe_expert_parallel import ExpertParallelMoE


class ExpertParallelMoEBackward(ExpertParallelMoE):
    def backward(self, dy):
        dy = dy.expect("T:ep d")
        # TODO: out = sum_k w_k * y_k: the gradient into the router weights, and into each choice's output
        dweights = einsum("t k d, t d -> t k", self.y, dy).expect("T:ep k")
        dy_choices = dy.view("T:ep 1 d") * self.weights.view("T:ep k 1")
        # END TODO
        # TODO: mirror the return trip, run the experts backward, mirror the dispatch trip
        dout = combine_backward(dy_choices, self.plan).expect("E C:ep d")
        dout = all_to_all(dout, "C", "E", ep).expect("E:ep C d")
        dW2 = einsum("e c f, e c d -> e f d", gelu(self.h), dout).expect("E:ep f d")
        dh = gelu_backward((dout @ self.W2.transpose(-1, -2)).expect("E:ep C f"), self.h)
        dW1 = einsum("e c d, e c f -> e d f", self.buffers, dh).expect("E:ep d f")
        dbuffers = (dh @ self.W1.transpose(-1, -2)).expect("E:ep C d")
        dbuffers = all_to_all(dbuffers, "E", "C", ep).expect("E C:ep d")
        dx = dispatch_backward(dbuffers, self.plan).expect("T:ep d")
        # END TODO
        # TODO: the router: softmax over the chosen logits, scattered back into [T E]
        dlogits = scatter(softmax_backward(dweights, self.weights, dim=-1), self.expert_ids, dim=-1, size="E")
        dWr = all_reduce(einsum("t d, t e -> d e", self.x, dlogits).expect("d E", partial=ep), ep).expect("d E")
        dx = dx + (dlogits @ self.Wr.transpose(0, 1)).expect("T:ep d")
        # END TODO
        return {"x": dx, "Wr": dWr, "W1": dW1, "W2": dW2}


HINTS = [
    "out = sum_k w_k y_k: dw = einsum('t k d, t d -> t k', y, dy), and the gradient into y_k is dy * w_k.",
    "combine_backward -> all_to_all(gather 'C', scatter 'E') -> expert backward -> all_to_all(gather 'E', scatter 'C') -> dispatch_backward.",
    "Router: scatter(softmax_backward(dw, w), ids, -1, 'E') gives dlogits; dWr sums over ep-split tokens, so all_reduce it over ep.",
]
