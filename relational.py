"""Learned directional patch compatibility and a coupled coarse/fine image energy.

No transformer, labels, target pixels at inference, or pretrained network.
Inference uses discrete local energy minimization and an exact coarse solve.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sihti3 import patchify, unpatchify, gaussian_fill

GRID = 8
STRIDE = 4
SIZE = 7
K_DEFAULT = 32


def _pair_overlap(codes):
    d = codes.reshape(-1, SIZE, SIZE)
    horizontal = ((d[:, None, :, 4:] - d[None, :, :, :3])**2).mean(axis=(2, 3))
    vertical = ((d[:, None, 4:, :] - d[None, :, :3, :])**2).mean(axis=(2, 3))
    return horizontal, vertical


def _cooccurrence_cost(assignments, k, smoothing=0.5):
    counts = np.zeros((2, k, k), dtype=float)
    for z in assignments:
        grid = z.reshape(GRID, GRID)
        for a, b in zip(grid[:, :-1].ravel(), grid[:, 1:].ravel()):
            counts[0, a, b] += 1
        for a, b in zip(grid[:-1, :].ravel(), grid[1:, :].ravel()):
            counts[1, a, b] += 1
    costs = []
    for matrix in counts:
        cond = (matrix + smoothing) / (matrix.sum(axis=1, keepdims=True) + smoothing*k)
        cost = -np.log(cond)
        cost -= cost.min()
        costs.append(cost)
    return tuple(costs)


@dataclass
class RelationalMedium:
    codes: np.ndarray
    cooc_h: np.ndarray
    cooc_v: np.ndarray
    coarse_mean: np.ndarray
    coarse_precision: np.ndarray
    exemplar_images: np.ndarray
    data_weight: float = 12.0
    overlap_weight: float = 2.0
    relation_weight: float = 0.14
    coarse_weight: float = 1.1
    prior_weight: float = 0.38

    @classmethod
    def fit(cls, train, codes=K_DEFAULT, seed=2026):
        images = np.asarray(train, dtype=np.float32)
        if images.ndim != 3 or images.shape[1:] != (32, 32) or len(images) < 20:
            raise ValueError("Need at least 20 grayscale 32x32 training images")
        rng = np.random.default_rng(seed)
        patch_sets = [patchify(img, SIZE, STRIDE) for img in images]
        train_patches = []
        for patches in patch_sets:
            contrast = patches.std(1)
            edges = np.flatnonzero(contrast > .07)
            plain = np.flatnonzero(contrast <= .07)
            if len(edges):
                train_patches.append(patches[rng.choice(edges, min(24, len(edges)), replace=False)])
            if len(plain):
                train_patches.append(patches[rng.choice(plain, min(8, len(plain)), replace=False)])
        samples = np.concatenate(train_patches).astype(np.float32)
        km = MiniBatchKMeans(n_clusters=codes, batch_size=512, n_init=2,
                             max_iter=75, random_state=seed)
        km.fit(samples)
        templates = km.cluster_centers_.astype(np.float64)
        labels = [km.predict(p.astype(np.float32)) for p in patch_sets]
        cooc_h, cooc_v = _cooccurrence_cost(labels, codes)
        coarse = images.reshape(-1, GRID, STRIDE, GRID, STRIDE).mean((2, 4)).reshape(len(images), -1)
        mean = coarse.mean(0)
        cov = np.cov(coarse, rowvar=False)
        scale = max(float(np.trace(cov)/len(mean)), 0.001)
        cov += (.16*scale + 0.015)*np.eye(len(mean))
        precision = np.linalg.inv(cov)
        precision *= 1. / np.mean(np.diag(precision))
        return cls(templates, cooc_h, cooc_v, mean, precision, images)

    @property
    def k(self):
        return len(self.codes)

    def _factors(self, observation, mask, relation=True):
        observed_patches = patchify(observation, SIZE, STRIDE)
        mask_patches = patchify(mask.astype(float), SIZE, STRIDE)
        delta = self.codes[None, :, :] - observed_patches[:, None, :]
        unary = self.data_weight * np.sum(delta**2*mask_patches[:, None, :], axis=2) / SIZE**2
        h, v = _pair_overlap(self.codes)
        rel = self.relation_weight if relation else 0.
        return unary, self.overlap_weight*h + rel*self.cooc_h, self.overlap_weight*v + rel*self.cooc_v

    def _coarse_shape(self, z):
        patches = self.codes[z].reshape(-1, SIZE, SIZE)
        return patches[:, 3:, 3:].mean((1, 2))

    def _optimal_coarse(self, q):
        prior = self.prior_weight*self.coarse_precision
        mat = self.coarse_weight*np.eye(len(q)) + prior
        rhs = self.coarse_weight*q + prior@self.coarse_mean
        return np.linalg.solve(mat, rhs)

    def energy(self, z, c, factors, global_on=True):
        unary, h, v = factors
        grid = z.reshape(GRID, GRID)
        total = unary[np.arange(len(z)), z].sum()
        total += h[grid[:, :-1], grid[:, 1:]].sum()
        total += v[grid[:-1, :], grid[1:, :]].sum()
        if global_on:
            q = self._coarse_shape(z)
            error = c-self.coarse_mean
            total += .5*self.coarse_weight*np.sum((q-c)**2)
            total += .5*self.prior_weight*float(error@self.coarse_precision@error)
        return float(total)

    def _site_cost(self, ix, z, c, factors, global_on=True, field=None):
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
        q = self.codes.reshape(-1, SIZE, SIZE)[:, 3:, 3:].mean((1, 2))
        if global_on:
            cost += .5*self.coarse_weight*(q-c[ix])**2
        if field is not None:
            cost -= field[ix]*q
        return cost

    def _sweep(self, z, c, factors, global_on=True, field=None):
        z = z.copy()
        for ix in range(GRID*GRID):
            z[ix] = np.argmin(self._site_cost(ix, z, c, factors, global_on, field))
        return z

    def _render(self, z, observation, mask):
        x = unpatchify(self.codes[z], (32, 32), SIZE, STRIDE)
        x = np.clip(x, 0, 1)
        x[mask] = observation[mask]
        return x

    def _probe_field(self, c, seed=0, sham=False):
        """Unlabelled neutral coarse pulse through the coarse covariance response."""
        rng = np.random.default_rng(seed)
        dipole = np.zeros(GRID*GRID)
        dipole[27] = 1.
        dipole[36] = -1.
        mat = self.coarse_weight*np.eye(GRID*GRID) + self.prior_weight*self.coarse_precision
        response = np.linalg.solve(mat, dipole)
        response /= max(np.linalg.norm(response), 1e-12)
        if sham:
            response = rng.permutation(response)
        return .12*response

    def infer(self, observation, mask, sweeps=12, relation=True, feedback=True,
              ping=False, sham=False, extra_search=False, seed=0):
        observation = np.asarray(observation, float)
        mask = np.asarray(mask, bool)
        if observation.shape != (32, 32) or mask.shape != (32, 32) or not mask.any():
            raise ValueError("32x32 observation with at least one visible pixel required")
        base = gaussian_fill(observation, mask)
        init = patchify(base, SIZE, STRIDE)
        visible = patchify(mask.astype(float), SIZE, STRIDE)
        observed = patchify(observation, SIZE, STRIDE)
        fit = np.sum((self.codes[None]-init[:, None])**2*.30 +
                     (self.codes[None]-observed[:, None])**2*visible[:, None], axis=2)
        z = np.argmin(fit, axis=1)
        c = self._optimal_coarse(self._coarse_shape(z))
        factors = self._factors(observation, mask, relation=relation)
        initial_coarse = c.copy()
        frames = [self._render(z, observation, mask)]
        energies = [self.energy(z, c, factors)]
        accepts = 0
        for step in range(sweeps + (sweeps//2 if extra_search else 0)):
            z = self._sweep(z, c, factors)
            c = self._optimal_coarse(self._coarse_shape(z)) if feedback else initial_coarse
            current = self.energy(z, c, factors)
            if (ping or sham) and step in (2, 6, 10):
                field = self._probe_field(c, seed+step, sham=sham)
                proposal = self._sweep(z, c, factors, field=field)
                test = self.energy(proposal, c, factors)
                if test < current - 1e-10:
                    z, current = proposal, test
                    accepts += 1
            frames.append(self._render(z, observation, mask))
            energies.append(current)
        return {"image": frames[-1], "frames": np.asarray(frames),
                "energy": np.asarray(energies), "accepted_pings": accepts,
                "codes": z.copy(), "coarse": c.reshape(GRID, GRID).copy()}
