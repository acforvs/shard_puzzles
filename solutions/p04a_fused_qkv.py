from shardpuzzles import *


class FusedAttention(Module):
    def on_init(self, p):
        # TODO: build Wqkv so that each rank's contiguous column slice is [its q heads | their k | their v]
        wq = p.Wq.view("d hkv (h/hkv)*hd")  # group the query columns by the kv head they use
        wk = p.Wk.view("d hkv hd")
        wv = p.Wv.view("d hkv hd")
        wqkv = shard(cat([wq, wk, wv], dim=2), "d hkv:tp (h/hkv+2)*hd")
        self.Wqkv = wqkv.view("d (hkv*(h/hkv+2)*hd):tp")
        # END TODO
        self.Wo = shard(p.Wo, "(h*hd):tp d")

    def forward(self, x):
        x = x.expect("b s:tp d")
        x = all_gather(x, "s", tp).expect("b s d")
        # TODO: one projection, then split it into q [b h:tp s hd] and k, v [b hkv:tp s hd]
        qkv = (x @ self.Wqkv).expect("b s (hkv*(h/hkv+2)*hd):tp").view("b s hkv:tp h/hkv+2 hd")
        q, k, v = qkv.split(["h/hkv", 1, 1], dim=3)
        q = q.view("b s h:tp hd").transpose(1, 2)
        k = k.view("b s hkv:tp hd").transpose(1, 2)
        v = v.view("b s hkv:tp hd").transpose(1, 2)
        # END TODO
        k, v = repeat_interleave(k, "h/hkv", dim=1), repeat_interleave(v, "h/hkv", dim=1)
        scores = (q @ k.transpose(-1, -2)).expect("b h:tp s s")
        probs = softmax(scores * size("hd") ** -0.5 + causal_mask("s s"), dim=-1)
        o = (probs @ v).expect("b h:tp s hd").transpose(1, 2).view("b s (h*hd):tp")
        self.x, self.q, self.k, self.v, self.probs, self.o = x, q, k, v, probs, o  # saved for 04b
        y = (o @ self.Wo).expect("b s d", partial=tp)
        return reduce_scatter(y, "s", tp).expect("b s:tp d")


HINTS = [
    "Rank 0's slice of cat([Wq, Wk, Wv]) is all query columns. Group the columns by kv head instead: [q heads of group j | k_j | v_j] for each j.",
    "In on_init: view Wq as 'd hkv (h/hkv)*hd' and Wk/Wv as 'd hkv hd', cat on dim 2, shard 'd hkv:tp (h/hkv+2)*hd', then view back to 2-D.",
    "In forward: view the projection as 'b s hkv:tp h/hkv+2 hd' and split ['h/hkv', 1, 1] on dim 3.",
]
