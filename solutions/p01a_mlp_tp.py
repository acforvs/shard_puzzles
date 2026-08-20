from shardpuzzles import *


class MLP(Module):
    def on_init(self, p):
        # TODO: keep this rank's piece of W1 [d f] and W2 [f d]
        self.W1 = shard(p.W1, "d f:tp")  # column-parallel
        self.W2 = shard(p.W2, "f:tp d")  # row-parallel
        # END TODO

    def forward(self, x):
        x = x.expect("b s d")
        # TODO: compute y with one collective (state each layout, and the collective's bytes)
        h = (x @ self.W1).expect("b s f:tp")
        y = (gelu(h) @ self.W2).expect("b s d", partial=tp)
        y = all_reduce(y, tp).expect("b s d", bytes="2 * 2*b*s*d * (tp-1)/tp")
        # END TODO
        self.x, self.h = x, h  # saved for backward
        return y


HINTS = [
    "W1 is split by columns ('d f:tp') and W2 by rows ('f:tp d'). Then x @ W1 needs no communication: [b s f:tp].",
    "[b s f:tp] @ [f:tp d] contracts the split dim, so each rank holds a partial sum: [b s d] partial over tp.",
    "all_reduce(y, tp) makes it whole. A ring all-reduce sends 2*(2 bytes*b*s*d)*(tp-1)/tp per rank.",
]
