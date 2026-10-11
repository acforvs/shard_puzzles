from shardpuzzles import *


class UlyssesAttention(Module):
    def on_init(self, p):
        self.Wq, self.Wk, self.Wv, self.Wo = p.Wq, p.Wk, p.Wv, p.Wo

    def forward(self, x):
        x = x.expect("b s:cp d")
        # TODO: q, k, v for every token of this rank's heads, as [b heads s hd]
        q = (x @ self.Wq).expect("b s:cp (h*hd)").view("b s:cp h hd")
        k = (x @ self.Wk).expect("b s:cp (hkv*hd)").view("b s:cp hkv hd")
        v = (x @ self.Wv).expect("b s:cp (hkv*hd)").view("b s:cp hkv hd")
        q = all_to_all(q, "s", "h", cp).expect("b s h:cp hd", bytes="2*b*s/cp*h*hd * (cp-1)/cp").transpose(1, 2)
        k = all_to_all(k, "s", "hkv", cp).expect("b s hkv:cp hd", bytes="2*b*s/cp*hkv*hd * (cp-1)/cp").transpose(1, 2)
        v = all_to_all(v, "s", "hkv", cp).expect("b s hkv:cp hd", bytes="2*b*s/cp*hkv*hd * (cp-1)/cp").transpose(1, 2)
        # END TODO
        k, v = repeat_interleave(k, "h/hkv", dim=1), repeat_interleave(v, "h/hkv", dim=1)
        # TODO: full-sequence causal attention on this rank's heads
        scores = (q @ k.transpose(-1, -2)).expect("b h:cp s s")
        probs = softmax(scores * size("hd") ** -0.5 + causal_mask("s s"), dim=-1)
        o = (probs @ v).expect("b h:cp s hd")
        # END TODO
        # TODO: back to one row per token of this rank's sequence chunk, ready for Wo
        o = all_to_all(o.transpose(1, 2), "h", "s", cp).expect("b s:cp h hd", bytes="2*b*s*h/cp*hd * (cp-1)/cp")
        o = o.view("b s:cp (h*hd)")
        # END TODO
        self.x, self.q, self.k, self.v, self.probs, self.o = x, q, k, v, probs, o  # saved for 13b
        return (o @ self.Wo).expect("b s:cp d")


HINTS = [
    "all_to_all(q, gather_dim='s', scatter_dim='h', group=cp) turns [b s:cp h hd] into [b s h:cp hd]. Do the same for k and v with 'hkv'.",
    "Plain causal attention on [b h:cp s hd]; then all_to_all(o, gather_dim='h', scatter_dim='s') back before Wo.",
]
