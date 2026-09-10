"""Puzzle 06b - Backward: vocab-parallel cross-entropy

The forward is 06a's reference solution, imported below. It saves self.x, self.targets, self.logits
[b s v:tp], and the global self.m and self.sumexp ([b s 1]). dy = dL/dloss: [b s], replicated.
Return {"x": [b s d], "W": [d v:tp]}.

d loss / d logits = softmax(logits) - one_hot(target). The forward already reduced what you need
(the max and the sum of exps). Which collectives are left?

Helpers: one_hot(ids, "v:tp") (each rank fills only its vocab columns).
"""

from puzzles.p06a_vocab_parallel_cross_entropy import SPEC as FORWARD
from shardpuzzles import *
from solutions.p06a_vocab_parallel_cross_entropy import (
    VocabParallelCrossEntropy,  # 06a's forward; saves the tensors listed above
)


# ---- your code ---------------------------------------------------------------
class VocabParallelCrossEntropyBackward(VocabParallelCrossEntropy):
    def backward(self, dy):
        dy = dy.expect("b s")
        # TODO: d loss / d logits for this rank's vocab slice
        probs = TODO
        dlogits = TODO
        # TODO: dW and dx
        dW = TODO
        dx = TODO
        return {"x": dx, "W": dW}


# ---- puzzle definition -------------------------------------------------------
SPEC = FORWARD.replace(grads={"x": "b s d", "W": "d v:tp"}, max_comm_bytes="4*b*s*(3+d)*(tp-1)/tp")
