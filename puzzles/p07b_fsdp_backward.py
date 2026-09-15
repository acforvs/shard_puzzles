"""Puzzle 07b - Backward: fully sharded data parallel (ZeRO-3)

The forward is 07a's reference solution, imported below. It saves self.x and self.h = x @ W1, but
not the gathered weights: they were freed after use.
dy: [b:dp s d]. Return {"x": [b:dp s d], "W1": [d:dp f], "W2": [f:dp d]}: gradients sharded like the
stored weights. Which collectives run, in which order, and how many bytes each? Compare with DDP.

Helpers: gelu_backward.

Read: ZeRO (arXiv 1910.02054) Sec. 5.3; PyTorch FSDP (arXiv 2304.11277) Sec. 3.
"""

from puzzles.p07a_fsdp import SPEC as FORWARD
from shardpuzzles import *
from solutions.p07a_fsdp import FSDPMLP  # 07a's forward; saves the tensors listed above


# ---- your code ---------------------------------------------------------------
class FSDPMLPBackward(FSDPMLP):
    def backward(self, dy):
        dy = dy.expect("b:dp s d")
        # TODO: W2 is needed again; then W2's gradient, sharded like the stored W2
        W2 = TODO
        da = TODO
        dW2 = TODO
        dh = gelu_backward(da, self.h)
        # TODO: the same for W1, plus dx
        W1 = TODO
        dx = TODO
        dW1 = TODO
        return {"x": dx, "W1": dW1, "W2": dW2}


# ---- puzzle definition -------------------------------------------------------
SPEC = FORWARD.replace(grads={"x": "b:dp s d", "W1": "d:dp f", "W2": "f:dp d"})
