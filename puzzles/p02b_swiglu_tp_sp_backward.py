"""Puzzle 02b - Backward: SwiGLU MLP with tensor + sequence parallelism

The forward is 02a's reference solution, imported below. It saves self.x (the gathered [b s d]),
self.gate, self.up and self.act = silu(gate) * up. dy: [b s:tp d]. Return
    {"x": [b s:tp d], "Wgate": [d f:tp], "Wup": [d f:tp], "Wdown": [f:tp d]}.

all_gather and reduce_scatter are each other's backward. Which one comes first here?

Helpers: silu_backward(dy, x).
"""

from puzzles.p02a_swiglu_tp_sp import SPEC as FORWARD
from shardpuzzles import *
from solutions.p02a_swiglu_tp_sp import SwiGLU  # 02a's forward; saves the tensors listed above


# ---- your code ---------------------------------------------------------------
class SwiGLUBackward(SwiGLU):
    def backward(self, dy):
        dy = dy.expect("b s:tp d")
        # TODO: undo the forward's last collective, then the down-projection
        dy = TODO
        dact = TODO
        dWdown = TODO
        dgate = silu_backward(dact * self.up, self.gate)
        dup = dact * silu(self.gate)
        # TODO: the gate/up weight gradients, and dx (laid out like the forward's input)
        dWgate = TODO
        dWup = TODO
        dx = TODO
        return {"x": dx, "Wgate": dWgate, "Wup": dWup, "Wdown": dWdown}


# ---- puzzle definition -------------------------------------------------------
SPEC = FORWARD.replace(
    grads={"x": "b s:tp d", "Wgate": "d f:tp", "Wup": "d f:tp", "Wdown": "f:tp d"}, require_bytes=False
)
