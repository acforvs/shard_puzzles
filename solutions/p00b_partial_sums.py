from shardpuzzles import *


class RowParallelLinear(Module):
    def on_init(self, p):
        # TODO: keep the rows of W [d f] that meet this rank's slice of d
        self.W = shard(p.W, "d:tp f")
        # END TODO

    def forward(self, x):
        x = x.expect("b s d:tp")  # the input: each rank holds a slice of the features
        # TODO: y = x @ W, replicated: what does each rank hold after its local matmul? (state the bytes)
        y = (x @ self.W).expect("b s f", partial=tp)
        y = all_reduce(y, tp).expect("b s f", bytes="4*b*s*f * (tp-1)/tp")
        # END TODO
        return y


HINTS = [
    "x is split along d, the dim that x @ W sums over. Keep the matching rows of W: shard(p.W, 'd:tp f').",
    "Each rank's x_r @ W_r is a full [b s f], but it sums only over its own d slice: .expect('b s f', partial=tp).",
    "all_reduce(y, tp) adds the pieces up on every rank; bytes = 2*2*b*s*f*(tp-1)/tp.",
]
