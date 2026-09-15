from shardpuzzles import *
from solutions.p07a_fsdp import FSDPMLP


class FSDPMLPBackward(FSDPMLP):
    def backward(self, dy):
        dy = dy.expect("b:dp s d")
        # TODO: W2 is needed again; then W2's gradient, sharded like the stored W2
        W2 = all_gather(self.W2, "f", dp).expect("f d", bytes="2*f*d * (dp-1)/dp")
        da = (dy @ W2.transpose(0, 1)).expect("b:dp s f")
        dW2 = einsum("b s f, b s d -> f d", gelu(self.h), dy).expect("f d", partial=dp)
        dW2 = reduce_scatter(dW2, "f", dp).expect("f:dp d", bytes="2*f*d * (dp-1)/dp")
        # END TODO
        dh = gelu_backward(da, self.h)
        # TODO: the same for W1, plus dx
        W1 = all_gather(self.W1, "d", dp).expect("d f", bytes="2*d*f * (dp-1)/dp")
        dx = (dh @ W1.transpose(0, 1)).expect("b:dp s d")
        dW1 = einsum("b s d, b s f -> d f", self.x, dh).expect("d f", partial=dp)
        dW1 = reduce_scatter(dW1, "d", dp).expect("d:dp f", bytes="2*d*f * (dp-1)/dp")
        # END TODO
        return {"x": dx, "W1": dW1, "W2": dW2}


HINTS = [
    "The forward freed the gathered weights: all_gather them again in backward.",
    "Weight gradients sum this rank's micro-batch only, so they are partial over dp. reduce_scatter them along the stored shard dim (ZeRO-3's gradient step).",
]
