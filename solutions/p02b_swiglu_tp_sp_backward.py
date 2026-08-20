from shardpuzzles import *
from solutions.p02a_swiglu_tp_sp import SwiGLU


class SwiGLUBackward(SwiGLU):
    def backward(self, dy):
        dy = dy.expect("b s:tp d")
        # TODO: undo the forward's last collective, then the down-projection
        dy = all_gather(dy, "s", tp).expect("b s d")
        dact = (dy @ self.Wdown.transpose(0, 1)).expect("b s f:tp")
        dWdown = einsum("b s f, b s d -> f d", self.act, dy).expect("f:tp d")
        # END TODO
        dgate = silu_backward(dact * self.up, self.gate)
        dup = dact * silu(self.gate)
        # TODO: the gate/up weight gradients, and dx (laid out like the forward's input)
        dWgate = einsum("b s d, b s f -> d f", self.x, dgate).expect("d f:tp")
        dWup = einsum("b s d, b s f -> d f", self.x, dup).expect("d f:tp")
        dx = (dgate @ self.Wgate.transpose(0, 1) + dup @ self.Wup.transpose(0, 1)).expect("b s d", partial=tp)
        dx = reduce_scatter(dx, "s", tp).expect("b s:tp d")
        # END TODO
        return {"x": dx, "Wgate": dWgate, "Wup": dWup, "Wdown": dWdown}


HINTS = [
    "The backward of the forward's reduce_scatter is an all_gather of dy along s.",
    "dx sums two matmuls that contract f:tp: partial over tp. The backward of the forward's all_gather is a reduce_scatter along s.",
]
