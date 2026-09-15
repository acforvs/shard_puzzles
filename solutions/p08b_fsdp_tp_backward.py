from shardpuzzles import *
from solutions.p08a_fsdp_tp import FSDPTPMLP


class FSDPTPMLPBackward(FSDPTPMLP):
    def backward(self, dy):
        dy = dy.expect("b:dp s d")
        # TODO: W2: re-gather, then its gradient (back to the stored layout)
        W2 = all_gather(self.W2, "d", dp).expect("f:tp d", bytes="2*f/tp*d * (dp-1)/dp")
        da = (dy @ W2.transpose(0, 1)).expect("b:dp s f:tp")
        dW2 = einsum("b s f, b s d -> f d", gelu(self.h), dy).expect("f:tp d", partial=dp)
        dW2 = reduce_scatter(dW2, "d", dp).expect("f:tp d:dp", bytes="2*f/tp*d * (dp-1)/dp")
        # END TODO
        dh = gelu_backward(da, self.h)
        # TODO: W1 and dx
        W1 = all_gather(self.W1, "d", dp).expect("d f:tp", bytes="2*d*f/tp * (dp-1)/dp")
        dx = (dh @ W1.transpose(0, 1)).expect("b:dp s d", partial=tp)
        dx = all_reduce(dx, tp).expect("b:dp s d", bytes="4*b/dp*s*d * (tp-1)/tp")
        dW1 = einsum("b s d, b s f -> d f", self.x, dh).expect("d f:tp", partial=dp)
        dW1 = reduce_scatter(dW1, "d", dp).expect("d:dp f:tp", bytes="2*d*f/tp * (dp-1)/dp")
        # END TODO
        return {"x": dx, "W1": dW1, "W2": dW2}


HINTS = [
    "dp: re-gather the weights and reduce_scatter their gradients. tp: the forward all_reduce's backward is the identity.",
    "dx = dh @ W1^T contracts f:tp: all_reduce over tp.",
]
