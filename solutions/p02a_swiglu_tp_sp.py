from shardpuzzles import *


class SwiGLU(Module):
    def on_init(self, p):
        # TODO: shard Wgate, Wup [d f] and Wdown [f d]
        self.Wgate = shard(p.Wgate, "d f:tp")
        self.Wup = shard(p.Wup, "d f:tp")
        self.Wdown = shard(p.Wdown, "f:tp d")
        # END TODO

    def forward(self, x):
        x = x.expect("b s:tp d")
        # TODO: every rank needs every token for its slice of f: gather, then the gate and up projections
        x = all_gather(x, "s", tp).expect("b s d", bytes="2*b*s*d * (tp-1)/tp")
        gate = (x @ self.Wgate).expect("b s f:tp")
        up = (x @ self.Wup).expect("b s f:tp")
        # END TODO
        act = silu(gate) * up
        # TODO: down-projection, and back to sequence-parallel
        y = (act @ self.Wdown).expect("b s d", partial=tp)
        y = reduce_scatter(y, "s", tp).expect("b s:tp d", bytes="2*b*s*d * (tp-1)/tp")
        # END TODO
        self.x, self.gate, self.up, self.act = x, gate, up, act  # saved for the backward (02b)
        return y


HINTS = [
    "Same weight sharding as 01a. x arrives as [b s:tp d], so all_gather it along s first.",
    "Instead of all_reduce, reduce_scatter the partial output along s: one collective returns [b s:tp d]. AG + RS move the same bytes as one all-reduce.",
]
