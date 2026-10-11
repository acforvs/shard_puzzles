"""Puzzle 13b - Backward: Ulysses sequence parallelism

The forward is 13a's reference solution, imported below. It saves self.x, self.q / self.k / self.v
[b h:cp s hd] (k, v repeated), self.probs [b h:cp s s] and self.o [b s:cp (h*hd)]. dy: [b s:cp d]. Return
    {"x": [b s:cp d], "Wq": [d h*hd], "Wk": [d hkv*hd], "Wv": [d hkv*hd], "Wo": [h*hd d]}.

The backward of an all_to_all is an all_to_all with gather_dim and scatter_dim swapped. Run the
attention backward on your heads, between two rounds of all_to_alls. Where should the repeat's
backward go relative to the dK/dV all_to_all, and why?

Helpers: softmax_backward, repeat_interleave_backward.
"""

from puzzles.p13a_ulysses import SPEC as FORWARD
from shardpuzzles import *
from solutions.p13a_ulysses import UlyssesAttention  # 13a's forward; saves the tensors listed above


# ---- your code ---------------------------------------------------------------
class UlyssesAttentionBackward(UlyssesAttention):
    def backward(self, dy):
        dy = dy.expect("b s:cp d")
        dWo = all_reduce(einsum("b s e, b s d -> e d", self.o, dy).expect("h*hd d", partial=cp), cp).expect("h*hd d")
        do = (dy @ self.Wo.transpose(0, 1)).expect("b s:cp (h*hd)").view("b s:cp h hd")
        # TODO: dO to head-split
        do = TODO
        dprobs = (do @ self.v.transpose(-1, -2)).expect("b h:cp s s")
        dv = (self.probs.transpose(-1, -2) @ do).expect("b h:cp s hd")
        ds = softmax_backward(dprobs, self.probs, dim=-1) * size("hd") ** -0.5
        dq = (ds @ self.k).expect("b h:cp s hd")
        dk = (ds.transpose(-1, -2) @ self.q).expect("b h:cp s hd")
        # TODO: dQ, dK, dV back to sequence-split ([b s:cp heads hd])
        dk = TODO
        dv = TODO
        dq = TODO
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


# ---- puzzle definition -------------------------------------------------------
SPEC = FORWARD.replace(
    grads={"x": "b s:cp d", "Wq": "d h*hd", "Wk": "d hkv*hd", "Wv": "d hkv*hd", "Wo": "h*hd d"},
    require_bytes=False,
    max_comm_bytes="(8*b*s*hd*(h+hkv)/cp+8*d*hd*(h+hkv))*(cp-1)/cp",
)
