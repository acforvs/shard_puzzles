"""Puzzle 01b - Backward: MLP with tensor parallelism

The forward is 01a's reference solution, imported below. It saves self.x [b s d] and
self.h = x @ W1 [b s f:tp]. dy: [b s d] (replicated, like y). Return
    {"x": [b s d], "W1": [d f:tp], "W2": [f:tp d]}  (each gradient laid out like its tensor).

Every collective has a conjugate in backward (Megatron calls the pair f / g): the forward's
all-reduce becomes nothing at all, and the replicated x needs an all-reduce of its gradient. Why?

Helpers: gelu_backward(dy, x) gives the gradient through gelu (the parallelism is yours).
"""

from puzzles.p01a_mlp_tp import SPEC as FORWARD
from shardpuzzles import *
from solutions.p01a_mlp_tp import MLP  # 01a's forward; saves the tensors listed above


# ---- your code ---------------------------------------------------------------
class MLPBackward(MLP):
    def backward(self, dy):
        dy = dy.expect("b s d")
        # TODO: the gradient into gelu's output, and W2's gradient
        da = TODO
        dW2 = TODO
        dh = gelu_backward(da, self.h)
        # TODO: W1's gradient, and x's gradient (x was replicated: what does that require?)
        dW1 = TODO
        dx = TODO
        return {"x": dx, "W1": dW1, "W2": dW2}


# ---- puzzle definition -------------------------------------------------------
SPEC = FORWARD.replace(grads={"x": "b s d", "W1": "d f:tp", "W2": "f:tp d"}, require_bytes=False)
