from shardpuzzles import *
from solutions.p13a_ulysses import UlyssesAttention


class UlyssesAttentionBackward(UlyssesAttention):
    def backward(self, dy):
        dy = dy.expect("b s:cp d")
        dWo = all_reduce(einsum("b s e, b s d -> e d", self.o, dy).expect("h*hd d", partial=cp), cp).expect("h*hd d")
        do = (dy @ self.Wo.transpose(0, 1)).expect("b s:cp (h*hd)").view("b s:cp h hd")
        # TODO: dO to head-split
        do = all_to_all(do, "s", "h", cp).expect("b s h:cp hd", bytes="2*b*s/cp*h*hd * (cp-1)/cp").transpose(1, 2)
        # END TODO
        dprobs = (do @ self.v.transpose(-1, -2)).expect("b h:cp s s")
        dv = (self.probs.transpose(-1, -2) @ do).expect("b h:cp s hd")
        ds = softmax_backward(dprobs, self.probs, dim=-1) * size("hd") ** -0.5
        dq = (ds @ self.k).expect("b h:cp s hd")
        dk = (ds.transpose(-1, -2) @ self.q).expect("b h:cp s hd")
        # TODO: dQ, dK, dV back to sequence-split ([b s:cp heads hd])
        dk = repeat_interleave_backward(dk, "h/hkv", dim=1)  # hkv heads are cheaper to send than h
        dv = repeat_interleave_backward(dv, "h/hkv", dim=1)
        dq = all_to_all(dq.transpose(1, 2), "h", "s", cp).expect("b s:cp h hd", bytes="2*b*s*h/cp*hd * (cp-1)/cp")
        dk = all_to_all(dk.transpose(1, 2), "hkv", "s", cp).expect("b s:cp hkv hd", bytes="2*b*s*hkv/cp*hd * (cp-1)/cp")
        dv = all_to_all(dv.transpose(1, 2), "hkv", "s", cp).expect("b s:cp hkv hd", bytes="2*b*s*hkv/cp*hd * (cp-1)/cp")
        # END TODO
        dq, dk, dv = dq.view("b s:cp (h*hd)"), dk.view("b s:cp (hkv*hd)"), dv.view("b s:cp (hkv*hd)")
        dWq = all_reduce(einsum("b s d, b s e -> d e", self.x, dq).expect("d h*hd", partial=cp), cp).expect("d h*hd")
        dWk = all_reduce(einsum("b s d, b s e -> d e", self.x, dk).expect("d hkv*hd", partial=cp), cp).expect(
            "d hkv*hd"
        )
        dWv = all_reduce(einsum("b s d, b s e -> d e", self.x, dv).expect("d hkv*hd", partial=cp), cp).expect(
            "d hkv*hd"
        )
        dx = (dq @ self.Wq.transpose(0, 1) + dk @ self.Wk.transpose(0, 1) + dv @ self.Wv.transpose(0, 1)).expect(
            "b s:cp d"
        )
        return {"x": dx, "Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo}


HINTS = [
    "Mirror each all_to_all with gather and scatter swapped: dO goes sequence-split -> head-split, dQ/dK/dV go back.",
    "Apply repeat_interleave_backward before the dK/dV all_to_all: hkv heads are cheaper to send than h.",
]
