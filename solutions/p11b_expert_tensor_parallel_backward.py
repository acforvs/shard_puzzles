from shardpuzzles import *
from solutions.p11a_expert_tensor_parallel import ExpertTensorParallelMoE


class ExpertTensorParallelMoEBackward(ExpertTensorParallelMoE):
    def backward(self, dy):
        dy = dy.expect("T:ep d")
        # TODO: the final all_gather's backward (dy is replicated), then the weighted sum's
        dy = shard(dy, "T:ep d:tp")  # free: each rank keeps its slice
        dweights = einsum("t k d, t d -> t k", self.y, dy).expect("T:ep k", partial=tp)
        dweights = all_reduce(dweights, tp).expect("T:ep k")
        dy_choices = dy.view("T:ep 1 d:tp") * self.weights.view("T:ep k 1")
        # END TODO
        # TODO: back through the return trip: d/tp wide over ep, then whole d for the experts
        dout = all_to_all(combine_backward(dy_choices, self.plan).expect("E C:ep d:tp"), "C", "E", ep).expect(
            "E:ep C d:tp"
        )
        dout = all_gather(dout, "d", tp).expect("E:ep C d")
        # END TODO
        dW2 = einsum("e c f, e c d -> e f d", gelu(self.h), dout).expect("E:ep f:tp d")
        dh = gelu_backward((dout @ self.W2.transpose(-1, -2)).expect("E:ep C f:tp"), self.h)
        dW1 = einsum("e c d, e c f -> e d f", self.buffers, dh).expect("E:ep d f:tp")
        # TODO: the expert input's gradient is partial over tp: reduce it so the trip home is d/tp wide
        dbuffers = (dh @ self.W1.transpose(-1, -2)).expect("E:ep C d", partial=tp)
        dbuffers = reduce_scatter(dbuffers, "d", tp).expect("E:ep C d:tp")
        dbuffers = all_to_all(dbuffers, "E", "C", ep).expect("E C:ep d:tp")
        dx = all_gather(dispatch_backward(dbuffers, self.plan).expect("T:ep d:tp"), "d", tp).expect("T:ep d")
        # END TODO
        dlogits = scatter(softmax_backward(dweights, self.weights, dim=-1), self.expert_ids, dim=-1, size="E")
        dWr = all_reduce(einsum("t d, t e -> d e", self.x, dlogits).expect("d E", partial=ep), ep).expect("d E")
        dx = dx + (dlogits @ self.Wr.transpose(0, 1)).expect("T:ep d")
        return {"x": dx, "Wr": dWr, "W1": dW1, "W2": dW2}


HINTS = [
    "With a replicated dy, the final all_gather's backward is shard(dy, 'T:ep d:tp'): free.",
    "dweights contracts d:tp: all_reduce over tp. Mirror the forward: all_gather over tp after the incoming all_to_all, reduce_scatter before the outgoing one, all_gather dx at the end.",
]
