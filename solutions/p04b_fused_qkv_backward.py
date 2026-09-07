from shardpuzzles import *
from solutions.p04a_fused_qkv import FusedAttention


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
        dq = dq.transpose(1, 2).view("b s hkv:tp h/hkv hd")
        dk = dk.transpose(1, 2).view("b s hkv:tp 1 hd")
        dv = dv.transpose(1, 2).view("b s hkv:tp 1 hd")
        dqkv = cat([dq, dk, dv], dim=3).view("b s (hkv*(h/hkv+2)*hd):tp")
        # END TODO
        # TODO: the fused weight's gradient and dx
        dWqkv = einsum("b s d, b s e -> d e", self.x, dqkv).expect("d (hkv*(h/hkv+2)*hd):tp")
        dx = (dqkv @ self.Wqkv.transpose(0, 1)).expect("b s d", partial=tp)
        dx = reduce_scatter(dx, "s", tp).expect("b s:tp d")
        # END TODO
        # TODO: un-fuse dWqkv into the original weights' layouts
        dWq, dWk, dWv = dWqkv.view("d hkv:tp h/hkv+2 hd").split(["h/hkv", 1, 1], dim=2)
        dWq, dWk, dWv = dWq.view("d (h*hd):tp"), dWk.view("d (hkv*hd):tp"), dWv.view("d (hkv*hd):tp")
        # END TODO
        return {"x": dx, "Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo}


HINTS = [
    "Rebuild the fused gradient: dq as 'b s hkv:tp h/hkv hd', dk/dv as 'b s hkv:tp 1 hd', cat on dim 3, view as 'b s (hkv*(h/hkv+2)*hd):tp'.",
    "dWqkv = einsum(x, dqkv). Un-fuse with view 'd hkv:tp h/hkv+2 hd' and split(['h/hkv', 1, 1], dim=2): every slice is already on the right rank.",
]
