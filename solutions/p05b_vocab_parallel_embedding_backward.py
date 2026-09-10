from shardpuzzles import *
from solutions.p05a_vocab_parallel_embedding import VocabParallelEmbedding


class VocabParallelEmbeddingBackward(VocabParallelEmbedding):
    def backward(self, dy):
        dy = dy.expect("b s:tp d")
        # TODO: every rank needs every token's gradient for the rows it owns
        dy = all_gather(dy, "s", tp).expect("b s d")
        dE = embedding_backward(self.ids, dy, "v:tp d").expect("v:tp d")
        # END TODO
        return {"E": dE}


HINTS = ["The reduce_scatter's backward is all_gather(dy) along s; then embedding_backward(ids, dy, 'v:tp d')."]
