"""Puzzle 08b - Backward: FSDP x tensor parallel

The forward is 08a's reference solution, imported below (gathered weights are freed after use). It
saves self.x and self.h = x @ W1. dy: [b:dp s d], replicated across tp. Return
    {"x": [b:dp s d], "W1": [d:dp f:tp], "W2": [f:tp d:dp]}.

Combine 01b and 07b. Which collectives run over dp and which over tp?

Helpers: gelu_backward.
"""

from puzzles.p08a_fsdp_tp import SPEC as FORWARD
from shardpuzzles import *
from solutions.p08a_fsdp_tp import FSDPTPMLP  # 08a's forward; saves the tensors listed above


# ---- your code ---------------------------------------------------------------
class FSDPTPMLPBackward(FSDPTPMLP):
    def backward(self, dy):
        dy = dy.expect("b:dp s d")
        # TODO: W2: re-gather, then its gradient (back to the stored layout)
        W2 = TODO
        da = TODO
        dW2 = TODO
        dh = gelu_backward(da, self.h)
        # TODO: W1 and dx
        W1 = TODO
        dx = TODO
        dW1 = TODO
        return {"x": dx, "W1": dW1, "W2": dW2}


# ---- puzzle definition -------------------------------------------------------
SPEC = FORWARD.replace(grads={"x": "b:dp s d", "W1": "d:dp f:tp", "W2": "f:tp d:dp"})
