from shardpuzzles import *


class Rescale(Module):
    def on_init(self, p):
        self.g = p.g  # replicated: every rank holds all of g

    def forward(self, x):
        x = x.expect("b s:tp d")  # the input: each rank holds its own tokens
        # TODO: y = x * g, then re-split it from s to d (state each collective's layout and bytes)
        y = x * self.g
        y = all_gather(y, "s", tp).expect("b s d", bytes="2*b*s*d * (tp-1)/tp")
        y = shard(y, "b s d:tp")
        # END TODO
        return y


HINTS = [
    "x * self.g is elementwise: each rank scales its own tokens and the layout stays [b s:tp d]. No .expect needed.",
    "all_gather(y, 's', tp) makes s whole. Its output has b*s*d elements per rank; each rank receives all but "
    "its own 1/tp of them, so bytes = 2*b*s*d*(tp-1)/tp.",
    "Now every rank holds all of y, so shard(y, 'b s d:tp') just keeps this rank's slice of d, for free.",
]
