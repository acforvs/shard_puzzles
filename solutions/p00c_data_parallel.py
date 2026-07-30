from shardpuzzles import *


class Linear(Module):
    def on_init(self, p):
        self.W = p.W  # replicated: every dp rank holds all of W

    def forward(self, x):
        x = x.expect("b:dp s d")  # the input: this rank's share of the batch
        self.x = x  # saved for backward
        # TODO: compute y and state its layout
        y = (x @ self.W).expect("b:dp s f")
        # END TODO
        return y

    def backward(self, dy):
        dy = dy.expect("b:dp s f")  # the gradient arrives laid out like forward's output
        # TODO: dx needs no communication
        dx = (dy @ self.W.transpose(0, 1)).expect("b:dp s d")
        # END TODO
        # TODO: dW: an einsum over this rank's tokens, then make it whole (bytes: an all-reduce sends 2*2*numel*(g-1)/g)
        dW = einsum("b s d, b s f -> d f", self.x, dy).expect("d f", partial=dp)
        dW = all_reduce(dW, dp).expect("d f", bytes="4*d*f * (dp-1)/dp")
        # END TODO
        return {"x": dx, "W": dW}


HINTS = [  # shown one at a time by: uv run check.py N --hint [k]
    "forward: y = (x @ self.W).expect('b:dp s f').",
    "backward: dx = (dy @ self.W.transpose(0, 1)).expect('b:dp s d').",
    "dW = einsum('b s d, b s f -> d f', self.x, dy) is partial over dp; all_reduce(dW, dp) and state bytes=4*d*f*(dp-1)/dp.",
]
