from shardpuzzles import *
from solutions.p01a_mlp_tp import MLP


class MLPBackward(MLP):
    def backward(self, dy):
        dy = dy.expect("b s d")
        # TODO: the gradient into gelu's output, and W2's gradient
        da = (dy @ self.W2.transpose(0, 1)).expect("b s f:tp")
        dW2 = einsum("b s f, b s d -> f d", gelu(self.h), dy).expect("f:tp d")
        # END TODO
        dh = gelu_backward(da, self.h)
        # TODO: W1's gradient, and x's gradient (x was replicated: what does that require?)
        dW1 = einsum("b s d, b s f -> d f", self.x, dh).expect("d f:tp")
        dx = (dh @ self.W1.transpose(0, 1)).expect("b s d", partial=tp)
        dx = all_reduce(dx, tp).expect("b s d")
        # END TODO
        return {"x": dx, "W1": dW1, "W2": dW2}


HINTS = [
    "The forward all_reduce's backward is the identity: dy is already complete on every rank. da = dy @ W2^T -> [b s f:tp].",
    "Weight gradients are einsums over the token dims (b, s); here they need no communication.",
    "dx = dh @ W1^T contracts f:tp, so it is partial over tp -> all_reduce.",
]
