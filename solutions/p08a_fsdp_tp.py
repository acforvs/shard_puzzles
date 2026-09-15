from shardpuzzles import *


class FSDPTPMLP(Module):
    def on_init(self, p):
        # TODO: two split dims per weight
        self.W1 = shard(p.W1, "d:dp f:tp")  # tp splits what Megatron splits, dp the other dim
        self.W2 = shard(p.W2, "f:tp d:dp")
        # END TODO

    def forward(self, x):
        x = x.expect("b:dp s d")
        # TODO: gather over dp, then puzzle 01a over tp
        W1 = all_gather(self.W1, "d", dp).expect("d f:tp", bytes="2*d*f/tp * (dp-1)/dp")
        h = (x @ W1).expect("b:dp s f:tp")
        W2 = all_gather(self.W2, "d", dp).expect("f:tp d", bytes="2*f/tp*d * (dp-1)/dp")
        y = (gelu(h) @ W2).expect("b:dp s d", partial=tp)
        y = all_reduce(y, tp).expect("b:dp s d", bytes="4*b/dp*s*d * (tp-1)/tp")
        # END TODO
        self.x, self.h = x, h  # saved for the backward (08b)
        return y


HINTS = [
    "Each weight has two split dims. tp splits what Megatron splits (W1's columns, W2's rows); dp splits the other one: 'd:dp f:tp' and 'f:tp d:dp'.",
    "Gather each weight over dp, then it's 01a: x @ W1 -> [b:dp s f:tp], @ W2 -> partial over tp, all_reduce over tp.",
]
