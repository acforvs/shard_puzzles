from shardpuzzles import *


class FSDPMLP(Module):
    def on_init(self, p):
        # TODO: store 1/dp of each weight
        self.W1 = shard(p.W1, "d:dp f")
        self.W2 = shard(p.W2, "f:dp d")
        # END TODO

    def forward(self, x):
        x = x.expect("b:dp s d")
        # TODO: gather each weight right before you use it
        W1 = all_gather(self.W1, "d", dp).expect("d f", bytes="2*d*f * (dp-1)/dp")
        h = (x @ W1).expect("b:dp s f")
        W2 = all_gather(self.W2, "f", dp).expect("f d", bytes="2*f*d * (dp-1)/dp")
        y = (gelu(h) @ W2).expect("b:dp s d")
        # END TODO
        self.x, self.h = x, h  # saved for the backward (07b); the gathered weights are freed
        return y


HINTS = [
    "Store W1 as 'd:dp f' and W2 as 'f:dp d' (any split dim works). In forward, all_gather each weight just before its matmul.",
    "Activations never move. Each gather sends 2*(weight elements)*(dp-1)/dp bytes per rank, independent of b.",
]
