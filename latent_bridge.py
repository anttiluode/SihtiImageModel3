"""R5: explicit Jacobian pullback for Sihti3's *actual* linear PCA decoder.

This is a diagnostic of latent expressivity, not a replacement for R4's
free-pixel continuous inverse or an implementation of a cortical gradient.
All ordinary methods accept only observation, mask and a predicted R4 image.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA

SHAPE = (32, 32)


def _check(observation, mask):
    x = np.asarray(observation, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if x.shape != SHAPE or m.shape != SHAPE or not m.any() or m.all():
        raise ValueError("Need 32x32 image and a nonempty partially visible mask")
    return x, m


@dataclass(frozen=True)
class LatentPCADecoder:
    mean: np.ndarray  # flat 1024, learned from training images only
    basis: np.ndarray  # (rank, 1024): same PCA decoder as sihti3.LearnedMedium

    @classmethod
    def fit(cls, train, rank=32, seed=2026):
        images = np.asarray(train, dtype=float)
        if images.ndim != 3 or images.shape[1:] != SHAPE or len(images) <= rank:
            raise ValueError("Need more complete 32x32 training images than rank")
        pca = PCA(n_components=rank, svd_solver="randomized", random_state=seed)
        pca.fit(images.reshape(len(images), -1))
        return cls(pca.mean_.copy(), pca.components_.copy())

    @property
    def rank(self):
        return self.basis.shape[0]

    def decode(self, z):
        z = np.asarray(z, float)
        if z.shape != (self.rank,):
            raise ValueError("latent coefficient shape mismatch")
        return (self.mean + z @ self.basis).reshape(SHAPE)

    def latent_from_visible(self, observation, mask, ridge=.05):
        """Condition PCA coefficients using only measured, visible pixels."""
        obs, m = _check(observation, mask)
        a = self.basis[:, m.ravel()].T
        b = obs[m] - self.mean[m.ravel()]
        return np.linalg.solve(a.T @ a + ridge*np.eye(self.rank), a.T @ b)

    def compose(self, z, observation, mask):
        obs, m = _check(observation, mask)
        out = np.clip(self.decode(z), 0., 1.)
        out[m] = obs[m]
        return out

    def hidden_jacobian(self, mask):
        m = np.asarray(mask, bool)
        if m.shape != SHAPE:
            raise ValueError("Expected 32x32 mask")
        return self.basis[:, (~m).ravel()].T  # d D(hidden) / d z

    def correction(self, observation, mask, pixel_target, steps=1,
                   method="pullback", ridge=.02):
        """Same ordinary image correction signal, transported into latent z.

        pullback: z <- z + J_hidden.T (pixel_target - D(z))_hidden,
        repeated steps times (one step is the literal Jacobian pullback).
        preconditioned: optimal ridge least-squares fit to this same pixel
        target in the missing pixels, starting at visible-fitted z0.
        """
        obs, m = _check(observation, mask)
        p = np.asarray(pixel_target, float)
        if p.shape != SHAPE:
            raise ValueError("pixel target must be 32x32")
        if method not in ("pullback", "preconditioned") or steps < 1:
            raise ValueError("unknown method or invalid steps")
        z0 = self.latent_from_visible(obs, m)
        a = self.hidden_jacobian(m)
        delta = p[~m] - self.decode(z0)[~m]
        if method == "preconditioned":
            dz = np.linalg.solve(a.T @ a + ridge*np.eye(self.rank),
                                 a.T @ delta)
            z = z0 + dz
        else:
            z = z0.copy()
            for _ in range(steps):
                # J^T is a derivative pullback, not an inverse Jacobian.
                error = p[~m] - self.decode(z)[~m]
                z += a.T @ error
        return self.compose(z, obs, m)

    def shuffled_correction(self, observation, mask, pixel_target, seed):
        obs, m = _check(observation, mask)
        z0 = self.latent_from_visible(obs, m)
        baseline = self.decode(z0)
        delta = np.asarray(pixel_target, float)[~m] - baseline[~m]
        shuffled = baseline.copy()
        shuffled[~m] += np.random.default_rng(seed).permutation(delta)
        return self.correction(obs, m, shuffled, steps=16)

    def oracle_best(self, target, observation, mask):
        """Privileged CEILING: solves using true hidden pixels, never inference."""
        obs, m = _check(observation, mask)
        true = np.asarray(target, float)
        if true.shape != SHAPE:
            raise ValueError("target shape mismatch")
        a = self.hidden_jacobian(m)
        z = np.linalg.lstsq(a, true[~m] - self.mean[(~m).ravel()],
                            rcond=None)[0]
        return self.compose(z, obs, m)

    def unreachable_fraction(self, target, observation, mask):
        """Fraction of true required hidden correction outside col(J_hidden).

        Optimistic linear tangent test: ignores constraints imposed by fitting
        visible pixels and measures raw (unclipped) Euclidean directions.
        """
        obs, m = _check(observation, mask)
        z0 = self.latent_from_visible(obs, m)
        delta = np.asarray(target, float)[~m] - self.decode(z0)[~m]
        a = self.hidden_jacobian(m)
        best = a @ np.linalg.lstsq(a, delta, rcond=None)[0]
        return float(np.dot(delta-best, delta-best) /
                     max(float(delta @ delta), 1e-15))
