from shardpuzzles import *


class VocabParallelEmbedding(Module):
    def on_init(self, p):
        # TODO: split the table
        self.E = shard(p.E, "v:tp d")
        # END TODO

    def forward(self, ids):
        ids = ids.expect("b s")  # token ids, replicated
        # TODO: look up what this rank owns, then one collective
        y = embedding(ids, self.E).expect("b s d", partial=tp)  # rows owned elsewhere are zeros
        y = reduce_scatter(y, "s", tp).expect("b s:tp d")
        # END TODO
        self.ids = ids  # saved for the backward (05b)
        return y


HINTS = [
    "shard(E, 'v:tp d'); embedding(ids, E) returns zeros for ids this rank doesn't own, so the result is a partial sum over tp.",
    "reduce_scatter along s sums the partials and splits the sequence in one step.",
]
