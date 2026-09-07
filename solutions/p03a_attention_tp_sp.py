from shardpuzzles import *


class Attention(Module):
    def on_init(self, p):
        # TODO: split the projections by heads
        self.Wq = shard(p.Wq, "d (h*hd):tp")
        self.Wk = shard(p.Wk, "d (hkv*hd):tp")
        self.Wv = shard(p.Wv, "d (hkv*hd):tp")
        self.Wo = shard(p.Wo, "(h*hd):tp d")
        # END TODO

    def forward(self, x):
        x = x.expect("b s:tp d")
        # TODO: gather the sequence, then project q, k, v into this rank's heads, as [b heads s hd]
        x = all_gather(x, "s", tp).expect("b s d")
        q = (x @ self.Wq).expect("b s (h*hd):tp").view("b s h:tp hd").transpose(1, 2)
        k = (x @ self.Wk).expect("b s (hkv*hd):tp").view("b s hkv:tp hd").transpose(1, 2)
        v = (x @ self.Wv).expect("b s (hkv*hd):tp").view("b s hkv:tp hd").transpose(1, 2)
        # END TODO
        k = repeat_interleave(k, "h/hkv", dim=1)  # each kv head serves h/hkv query heads
        v = repeat_interleave(v, "h/hkv", dim=1)
        # TODO: causal attention on this rank's heads; then o as one row per token, ready for Wo
        scores = (q @ k.transpose(-1, -2)).expect("b h:tp s s")
        probs = softmax(scores * size("hd") ** -0.5 + causal_mask("s s"), dim=-1)
        o = (probs @ v).expect("b h:tp s hd").transpose(1, 2).view("b s (h*hd):tp")
        # END TODO
        # TODO: output projection, back to sequence-parallel
        y = (o @ self.Wo).expect("b s d", partial=tp)
        y = reduce_scatter(y, "s", tp).expect("b s:tp d")
        # END TODO
        self.x, self.q, self.k, self.v, self.probs, self.o = x, q, k, v, probs, o  # saved for 03b
        return y


HINTS = [
    "Shard Wq/Wk/Wv by columns, i.e. by heads: 'd (h*hd):tp'. Then (x @ Wq) viewed as 'b s h:tp hd' gives each rank h/tp heads.",
    "Rank r's query heads use kv heads r*hkv/tp ...; the same column split of Wk/Wv holds exactly those.",
    "Wo is row-parallel, '(h*hd):tp d': o @ Wo is partial over tp, then reduce_scatter along s.",
]
