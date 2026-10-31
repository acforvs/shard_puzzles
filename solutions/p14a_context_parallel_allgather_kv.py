from shardpuzzles import *


class AllGatherKVAttention(Module):
    def on_init(self, p):
        self.Wq, self.Wk, self.Wv, self.Wo = p.Wq, p.Wk, p.Wv, p.Wo  # replicated

    def forward(self, x):
        x = x.expect("b s:cp d")
        # TODO: project this rank's tokens into q [b h s:cp hd], k and v [b hkv s:cp hd]
        q = (x @ self.Wq).expect("b s:cp (h*hd)").view("b s:cp h hd").transpose(1, 2)
        k = (x @ self.Wk).expect("b s:cp (hkv*hd)").view("b s:cp hkv hd").transpose(1, 2)
        v = (x @ self.Wv).expect("b s:cp (hkv*hd)").view("b s:cp hkv hd").transpose(1, 2)
        # END TODO
        # TODO: give every rank the whole sequence of keys and values (as few bytes as possible)
        k = all_gather(k, "s", cp).expect("b hkv s hd", bytes="2*b*hkv*s*hd * (cp-1)/cp")
        v = all_gather(v, "s", cp).expect("b hkv s hd", bytes="2*b*hkv*s*hd * (cp-1)/cp")
        # END TODO
        k, v = repeat_interleave(k, "h/hkv", dim=1), repeat_interleave(v, "h/hkv", dim=1)
        # TODO: attention for this rank's queries; then o as one row per token, ready for Wo
        scores = (q @ k.transpose(-1, -2)).expect("b h s:cp s")
        probs = softmax(scores * size("hd") ** -0.5 + causal_mask("s:cp s"), dim=-1)
        o = (probs @ v).expect("b h s:cp hd").transpose(1, 2).view("b s:cp (h*hd)")
        # END TODO
        return (o @ self.Wo).expect("b s:cp d")


HINTS = [
    "Project q, k, v locally (the weights are replicated). all_gather only k and v along s, BEFORE repeat_interleave (hkv < h).",
    "scores are [b h s:cp s]; causal_mask('s:cp s') offsets each rank's query positions.",
]
