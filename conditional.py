"""R2: observation-conditioned top-down prediction and fine-to-coarse revision.

Separate from R1. Inference never sees the missing ground-truth pixels.
Local patch codes and their learned pairwise statistics come from R1.
"""
from dataclasses import dataclass
import numpy as np
from relational import RelationalMedium, SIZE, STRIDE, GRID
from sihti3 import patchify, gaussian_fill


def block_observations(observation, mask, mean):
    """Mask-aware 4x4 cell readings, with uncertainty for partial coverage."""
    obs = np.asarray(observation, float)
    m = np.asarray(mask, bool)
    if obs.shape != (32, 32) or m.shape != (32, 32) or not m.any():
        raise ValueError("Expected 32x32 observation/mask with some visible pixels")
    values = np.where(m, obs, 0.0)  # Never read hidden input placeholders.
    sums = values.reshape(GRID, 4, GRID, 4).sum(axis=(1, 3)).ravel()
    counts = m.reshape(GRID, 4, GRID, 4).sum(axis=(1, 3)).ravel()
    b = np.asarray(mean, float).copy()
    np.divide(sums, counts, out=b, where=counts > 0)
    fractions = counts / 16.0
    return b, fractions


@dataclass
class ConditionalBridge:
    base: RelationalMedium
    observation_strength: float = 18.0
    prior_strength: float = 0.38
    coupling_strength: float = 2.5

    @classmethod
    def fit(cls, train, codes=32, seed=2026):
        return cls(RelationalMedium.fit(train, codes=codes, seed=seed))

    @property
    def coarse_mean(self):
        return self.base.coarse_mean

    def context(self, observation, mask):
        b, coverage = block_observations(observation, mask, self.coarse_mean)
        w = self.observation_strength * coverage**2
        # Patch support reflects only the visible sample locations, never data values.
        patch_fraction = patchify(np.asarray(mask, bool).astype(float), SIZE, STRIDE).mean(1)
        reliability = .12 + .88*patch_fraction
        P = self.prior_strength * self.base.coarse_precision
        A = P + np.diag(w)
        rhs = P @ self.coarse_mean + w*b
        c0 = np.linalg.solve(A, rhs)
        return {"b": b, "coverage": coverage, "w": w,
                "reliability": reliability, "P": P, "c0": c0}

    def conditional_predict(self, observation, mask):
        """The first top-down message is independent of any fine patch guess."""
        return self.context(observation, mask)["c0"].reshape(GRID, GRID)

    def _factors(self, observation, mask, relation=True):
        return self.base._factors(observation, mask, relation=relation)

    def _q_codes(self):
        return self.base.codes.reshape(-1, SIZE, SIZE)[:, 3:, 3:].mean(axis=(1, 2))

    def energy(self, z, c, factors, ctx):
        unary, horizontal, vertical = factors
        grid = z.reshape(GRID, GRID)
        data = float(unary[np.arange(64), z].sum()
                     + horizontal[grid[:, :-1], grid[:, 1:]].sum()
                     + vertical[grid[:-1, :], grid[1:, :]].sum())
        q = self._q_codes()[z]
        d = c-self.coarse_mean
        return data + .5*float(d @ ctx["P"] @ d) + .5*np.sum(ctx["w"]*(c-ctx["b"])**2) + .5*self.coupling_strength*np.sum(ctx["reliability"]*(c-q)**2)

    def _site_cost(self, ix, z, c, factors, ctx):
        unary, h, v = factors
        row, col = divmod(ix, GRID)
        cost = unary[ix].copy()
        if col < GRID-1:
            cost += h[:, z[ix+1]]
        if col > 0:
            cost += h[z[ix-1], :]
        if row < GRID-1:
            cost += v[:, z[ix+GRID]]
        if row > 0:
            cost += v[z[ix-GRID], :]
        q = self._q_codes()
        cost += .5*self.coupling_strength*ctx["reliability"][ix]*(q-c[ix])**2
        return cost

    def _sweep(self, z, c, factors, ctx):
        z = z.copy()
        changed = 0
        for i in range(GRID*GRID):
            choice = int(np.argmin(self._site_cost(i, z, c, factors, ctx)))
            changed += choice != z[i]
            z[i] = choice
        return z, changed

    def _solve_coarse(self, z, ctx):
        q = self._q_codes()[z]
        upward = self.coupling_strength * ctx["reliability"]
        A = ctx["P"] + np.diag(ctx["w"] + upward)
        rhs = ctx["P"] @ self.coarse_mean + ctx["w"]*ctx["b"] + upward*q
        return np.linalg.solve(A, rhs)

    def _init_codes(self, observation, mask):
        # Shared initialization for all arms, independent of coarse predictions.
        fill = gaussian_fill(observation, mask)
        p = patchify(fill, SIZE, STRIDE)
        seen = patchify(np.asarray(mask, bool).astype(float), SIZE, STRIDE)
        observed = patchify(np.asarray(observation, float), SIZE, STRIDE)
        cost = np.sum((self.base.codes[None]-p[:, None])**2*.30 +
                      (self.base.codes[None]-observed[:, None])**2*seen[:, None], axis=2)
        return np.argmin(cost, axis=1)

    def run(self, observation, mask, steps=12, mode="live", relation=True,
            seed=0):
        """Modes: mean, shuffled, frozen, live, no_coarse. No ping in R2."""
        if mode not in ("mean", "shuffled", "frozen", "live", "no_coarse"):
            raise ValueError(f"Unknown mode: {mode}")
        observation = np.asarray(observation, float)
        mask = np.asarray(mask, bool)
        ctx = self.context(observation, mask)
        conditioned = ctx["c0"].copy()
        rng = np.random.default_rng(seed)
        if mode in ("mean", "no_coarse"):
            c = self.coarse_mean.copy()
        elif mode == "shuffled":
            c = conditioned.copy()
            missing = np.flatnonzero(ctx["coverage"] < 1.0)
            c[missing] = rng.permutation(c[missing])
        else:
            c = conditioned.copy()
        z = self._init_codes(observation, mask)
        factors = self._factors(observation, mask, relation)
        if mode == "no_coarse":
            # True ablation: remove coupling from both site and energy terms.
            # Prior/observation energy remains constant, independent of z.
            coupling = self.coupling_strength
            self.coupling_strength = 0.0
        else:
            coupling = None
        try:
            frames = [self.base._render(z, observation, mask)]
            coarse_frames = [c.reshape(GRID, GRID).copy()]
            energies = [self.energy(z, c, factors, ctx)]
            changed = []
            for _ in range(steps):
                z, n_changed = self._sweep(z, c, factors, ctx)
                if mode == "live":
                    c = self._solve_coarse(z, ctx)
                changed.append(int(n_changed))
                frames.append(self.base._render(z, observation, mask))
                coarse_frames.append(c.reshape(GRID, GRID).copy())
                energies.append(self.energy(z, c, factors, ctx))
        finally:
            if coupling is not None:
                self.coupling_strength = coupling
        return {
            "image": frames[-1],
            "frames": np.asarray(frames),
            "coarse_frames": np.asarray(coarse_frames),
            "coarse_prediction": conditioned.reshape(GRID, GRID),
            "z": z.copy(),
            "changes": changed,
            "energy": energies,
            "coarse_shift_norm": float(np.linalg.norm(c-conditioned)) if mode == "live" else 0.0,
            "context": {"visible_fraction": ctx["coverage"].reshape(GRID, GRID)},
        }
