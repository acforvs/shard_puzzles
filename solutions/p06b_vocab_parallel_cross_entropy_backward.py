from shardpuzzles import *
from solutions.p06a_vocab_parallel_cross_entropy import VocabParallelCrossEntropy


class VocabParallelCrossEntropyBackward(VocabParallelCrossEntropy):
    def backward(self, dy):
        dy = dy.expect("b s")
        # TODO: d loss / d logits for this rank's vocab slice
        probs = exp(self.logits - self.m) / self.sumexp
        dlogits = (probs - one_hot(self.targets, "v:tp")) * dy.view("b s 1")
        # END TODO
        # TODO: dW and dx
        dW = einsum("b s d, b s v -> d v", self.x, dlogits).expect("d v:tp")
        dx = (dlogits @ self.W.transpose(0, 1)).expect("b s d", partial=tp)
        dx = all_reduce(dx, tp).expect("b s d")
        # END TODO
        return {"x": dx, "W": dW}


HINTS = [
    "exp(logits - m) / sumexp is this rank's slice of the global softmax; subtract one_hot(targets, 'v:tp') and scale by dy.",
    "dW = einsum(x, dlogits) is local; dx = dlogits @ W^T contracts v:tp -> all_reduce.",
]
