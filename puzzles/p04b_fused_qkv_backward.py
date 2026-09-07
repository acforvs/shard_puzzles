"""Puzzle 04b - Backward: fused QKV projection under tensor parallelism

The forward is 04a's reference solution, imported below. It saves self.x (gathered), self.q / self.k /
self.v ([b h:tp s hd]; k, v repeated), self.probs and self.o ([b s (h*hd):tp]). dy: [b s:tp d].
Return gradients for the ORIGINAL weights:
    {"x": [b s:tp d], "Wq": [d (h*hd):tp], "Wk": [d (hkv*hd):tp], "Wv": [d (hkv*hd):tp], "Wo": [(h*hd):tp d]}

Rebuild the fused gradient dqkv in the forward's fused layout, then un-fuse dWqkv into
dWq / dWk / dWv. Does any shard have to move to another rank?
"""

from puzzles.p03b_attention_tp_sp_backward import SPEC as UNFUSED_BACKWARD
from puzzles.p04a_fused_qkv import SPEC as FORWARD
from shardpuzzles import *
from solutions.p04a_fused_qkv import FusedAttention  # 04a's forward; saves the tensors listed above


# ---- your code ---------------------------------------------------------------
class FusedAttentionBackward(FusedAttention):
    def backward(self, dy):
        dy = dy.expect("b s:tp d")
        dy = all_gather(dy, "s", tp).expect("b s d")
        dWo = einsum("b s e, b s d -> e d", self.o, dy).expect("(h*hd):tp d")
        do = (dy @ self.Wo.transpose(0, 1)).expect("b s (h*hd):tp").view("b s h:tp hd").transpose(1, 2)
        dv = (self.probs.transpose(-1, -2) @ do).expect("b h:tp s hd")
        dprobs = (do @ self.v.transpose(-1, -2)).expect("b h:tp s s")
        ds = softmax_backward(dprobs, self.probs, dim=-1) * size("hd") ** -0.5
        dq = (ds @ self.k).expect("b h:tp s hd")
        dk = repeat_interleave_backward((ds.transpose(-1, -2) @ self.q).expect("b h:tp s hd"), "h/hkv", dim=1)
        dv = repeat_interleave_backward(dv, "h/hkv", dim=1)
        # TODO: assemble dqkv in the forward's fused layout, [b s (hkv*(h/hkv+2)*hd):tp]
        dq = TODO
        dk = TODO
        dv = TODO
        dqkv = TODO
        # TODO: the fused weight's gradient and dx
        dWqkv = TODO
        dx = TODO
        # TODO: un-fuse dWqkv into the original weights' layouts
        dWq, dWk, dWv = TODO
        return {"x": dx, "Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo}


# ---- puzzle definition -------------------------------------------------------
SPEC = FORWARD.replace(grads=UNFUSED_BACKWARD.grads)
