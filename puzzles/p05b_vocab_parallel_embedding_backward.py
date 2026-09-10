"""Puzzle 05b - Backward: vocab-parallel embedding

The forward is 05a's reference solution, imported below. It saves self.ids. dy: [b s:tp d].
Return {"E": [v:tp d]} (ids are integers; they have no gradient).

Which rows of dE does a rank own, and which tokens' gradients does it need for them?

Helpers: embedding_backward(ids, dy, "v:tp d") adds dy rows into the table rows this rank owns.
"""

from puzzles.p05a_vocab_parallel_embedding import SPEC as FORWARD
from shardpuzzles import *
from solutions.p05a_vocab_parallel_embedding import (
    VocabParallelEmbedding,  # 05a's forward; saves the tensors listed above
)


# ---- your code ---------------------------------------------------------------
class VocabParallelEmbeddingBackward(VocabParallelEmbedding):
    def backward(self, dy):
        dy = dy.expect("b s:tp d")
        # TODO: every rank needs every token's gradient for the rows it owns
        dy = TODO
        dE = TODO
        return {"E": dE}


# ---- puzzle definition -------------------------------------------------------
SPEC = FORWARD.replace(grads={"E": "v:tp d"}, max_comm_bytes="4*b*s*d*(tp-1)/tp")
