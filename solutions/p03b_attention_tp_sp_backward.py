from shardpuzzles import *
from solutions.p03a_attention_tp_sp import Attention


class AttentionBackward(Attention):
    def backward(self, dy):
        dy = dy.expect("b s:tp d")
        # TODO: undo the reduce_scatter; Wo's gradient; dO per head as [b h:tp s hd]
        dy = all_gather(dy, "s", tp).expect("b s d")
        dWo = einsum("b s e, b s d -> e d", self.o, dy).expect("(h*hd):tp d")
        do = (dy @ self.Wo.transpose(0, 1)).expect("b s (h*hd):tp").view("b s h:tp hd").transpose(1, 2)
        # END TODO
        dv = (self.probs.transpose(-1, -2) @ do).expect("b h:tp s hd")
        dprobs = (do @ self.v.transpose(-1, -2)).expect("b h:tp s s")
        ds = softmax_backward(dprobs, self.probs, dim=-1) * size("hd") ** -0.5
        dq = (ds @ self.k).expect("b h:tp s hd")
        dk = (ds.transpose(-1, -2) @ self.q).expect("b h:tp s hd")
        # TODO: undo the kv repeat, then back to [b s (heads*hd):tp]
        dk = repeat_interleave_backward(dk, "h/hkv", dim=1)
        dv = repeat_interleave_backward(dv, "h/hkv", dim=1)
        dq = dq.transpose(1, 2).view("b s (h*hd):tp")
        dk = dk.transpose(1, 2).view("b s (hkv*hd):tp")
        dv = dv.transpose(1, 2).view("b s (hkv*hd):tp")
        # END TODO
        # TODO: weight gradients and dx
        dWq = einsum("b s d, b s e -> d e", self.x, dq).expect("d (h*hd):tp")
        dWk = einsum("b s d, b s e -> d e", self.x, dk).expect("d (hkv*hd):tp")
        dWv = einsum("b s d, b s e -> d e", self.x, dv).expect("d (hkv*hd):tp")
        dx = dq @ self.Wq.transpose(0, 1) + dk @ self.Wk.transpose(0, 1) + dv @ self.Wv.transpose(0, 1)
        dx = reduce_scatter(dx.expect("b s d", partial=tp), "s", tp).expect("b s:tp d")
        # END TODO
        return {"x": dx, "Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo}


HINTS = [
    "After all_gather(dy): dWo = einsum(o, dy); dO = dy @ Wo^T, viewed back into heads.",
    "repeat_interleave_backward(dk, 'h/hkv', dim=1) sums the query heads that shared a kv head.",
    "dx = dq @ Wq^T + dk @ Wk^T + dv @ Wv^T contracts this rank's heads: partial over tp -> reduce_scatter along s.",
]
