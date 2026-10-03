from shardpuzzles import *


class LatentMoE(Module):
    def on_init(self, p):
        self.Wr, self.Wdown, self.Wup, self.Ws1, self.Ws2 = p.Wr, p.Wdown, p.Wup, p.Ws1, p.Ws2  # replicated
        self.W1 = shard(p.W1, "E:ep l f")
        self.W2 = shard(p.W2, "E:ep f l")

    def forward(self, x):
        x = x.expect("T:ep d")
        top, expert_ids = topk((x @ self.Wr).expect("T:ep E"), "k", dim=-1)  # route on the full-width token
        weights = softmax(top, dim=-1)
        # TODO: compress, then dispatch l-wide vectors to the experts' ranks
        z = (x @ self.Wdown).expect("T:ep l")
        buffers, plan = dispatch(z, expert_ids, "E", "C")
        buffers = all_to_all(buffers.expect("E C:ep l"), "C", "E", ep).expect(
            "E:ep C l", bytes="2*E*C/ep*l * (ep-1)/ep"
        )
        # END TODO
        h = (buffers @ self.W1).expect("E:ep C f")
        y = (gelu(h) @ self.W2).expect("E:ep C l")
        # TODO: back home, combine in the latent space, then expand once
        y = all_to_all(y, "E", "C", ep).expect("E C:ep l", bytes="2*E*C/ep*l * (ep-1)/ep")
        y = combine(y, plan).expect("T:ep k l")
        routed_l = sum(y * weights.view("T:ep k 1"), "k").expect("T:ep l")
        routed = (routed_l @ self.Wup).expect("T:ep d")
        # END TODO
        hs = (x @ self.Ws1).expect("T:ep fs")
        shared = (gelu(hs) @ self.Ws2).expect("T:ep d")
        self.x, self.expert_ids, self.weights, self.plan = x, expert_ids, weights, plan  # saved for 12b
        self.buffers, self.h, self.y, self.routed_l, self.hs = buffers, h, y, routed_l, hs
        return routed + shared


HINTS = [
    "Order: route on x; z = x @ Wdown; dispatch z (l wide); experts in l; combine and the weighted sum in l; @ Wup; add the full-width shared expert.",
    "Bytes per all_to_all: 2*E*C/ep*l*(ep-1)/ep. With E -> alpha*E, l -> d/alpha and C unchanged, that's 09a's number.",
]
