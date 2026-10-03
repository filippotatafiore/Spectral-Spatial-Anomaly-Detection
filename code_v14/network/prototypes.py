import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalityPrototypes(nn.Module):
    """
        Memory of K prototypes of normal token embeddings.
        EMA-updated (k-means style) from the tokens of normal images only.
        Deviation of a token = cosine distance to its nearest normal prototype, in [0, 2].
    """

    def __init__(self, dim=128, num_prototypes=64, momentum=0.99):
        super(NormalityPrototypes, self).__init__()
        self.momentum = momentum
        self.register_buffer("protos", F.normalize(torch.randn(num_prototypes, dim), dim=1))  # (K, dim)
        self.register_buffer("usage", torch.ones(num_prototypes))                               # EMA of the assignment frequency
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, z):
        # z: (B, N, dim) L2-normalized token embeddings
        sim = z @ self.protos.t()                   # (B, N, K) cosine similarities
        return 1.0 - sim.max(dim=-1).values         # (B, N) deviation

    @torch.no_grad()
    def update(self, z, rate=1.0, reseed=True):
        # z: (..., dim) L2-normalized embeddings of normal tokens
        # rate: scales the EMA step (1 - momentum) (lr / LR -> the memory slows down and freezes with the LR schedule)
        # reseed: allow re-seeding of dead prototypes
        # returns the number of re-seeded prototypes

        z = z.reshape(-1, z.shape[-1]).float()
        K = self.protos.shape[0]
        if z.shape[0] < K or rate <= 0:
            return 0

        # First call: initialize the prototypes on random normal tokens
        if not self.initialized:
            self.protos.copy_(z[torch.randperm(z.shape[0], device=z.device)[:K]])
            self.initialized.fill_(True)
            return 0

        # Assign each token to its nearest prototype
        onehot = F.one_hot((z @ self.protos.t()).argmax(dim=1), K).float()  # (M, K)
        counts = onehot.sum(dim=0)                                         # (K,)
        means = (onehot.t() @ z) / counts.clamp(min=1).unsqueeze(1)        # (K, dim) mean of the assigned tokens

        # EMA update of the used prototypes (step scaled by rate)
        step = (1 - self.momentum) * rate
        used = counts > 0
        self.protos[used] = F.normalize((1 - step) * self.protos[used] + step * means[used], dim=1)

        # Usage EMA with the same step (a frozen memory cannot "die")
        self.usage.mul_(1 - step).add_(step * counts / counts.sum() * K)
        if not reseed:
            return 0

        # Re-seed dead prototypes (never assigned) on random normal tokens
        dead = self.usage < 0.01
        n_dead = int(dead.sum())
        if n_dead:
            self.protos[dead] = z[torch.randint(0, z.shape[0], (n_dead,), device=z.device)]
            self.usage[dead] = 1.0
        return n_dead