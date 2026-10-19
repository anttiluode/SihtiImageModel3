"""Sihti Image Model 3: coupled coarse / fine patch energy model (CPU).

No transformer, Qwen, neural training, or ground-truth pixels at inference.
The local prior and global low-rank basis are fit only on training images.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from numpy.lib.stride_tricks import sliding_window_view


def sieve(image, depths=(0, 1, 4, 16, 64), dt=0.2):
    """Exact telescoping Sihti decomposition with a stable heat step."""
    x = np.asarray(image, float)
    states = []
    cur = x.copy()
    t = 0
    for depth in depths:
        while t < depth:
            pad = np.pad(cur, 1, mode="edge")
            lap = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:] - 4*cur
            cur = cur + dt*lap
            t += 1
        states.append(cur.copy())
    return [states[i]-states[i+1] for i in range(len(states)-1)] + [states[-1]]


def patchify(x, size=7, stride=2):
    p = size//2
    windows = sliding_window_view(np.pad(x, p, mode="reflect"), (size, size))[::stride, ::stride]
    return windows.reshape(-1, size*size)


def unpatchify(patches, shape, size=7, stride=2):
    h, w = shape
    pad = size//2
    acc = np.zeros((h+2*pad, w+2*pad), float)
    weight = np.zeros_like(acc)
    n = 0
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            q = patches[n].reshape(size, size)
            acc[i:i+size, j:j+size] += q
            weight[i:i+size, j:j+size] += 1
            n += 1
    return (acc/np.maximum(weight, 1))[pad:pad+h, pad:pad+w]


def gaussian_fill(observation, mask, sigma=1.8, cycles=60):
    """Ordinary linear diffusion with visible pixels clamped."""
    x = np.asarray(observation, float).copy()
    visible = np.asarray(mask, bool)
    x[~visible] = float(np.mean(x[visible])) if np.any(visible) else .5
    for _ in range(cycles):
        x[~visible] = gaussian_filter(x, sigma=sigma)[~visible]
        x[visible] = observation[visible]
    return x


@dataclass
class LearnedMedium:
    mean: np.ndarray
    basis: np.ndarray
    codes: np.ndarray
    examples: np.ndarray
    patch_size: int = 7
    stride: int = 2

    @classmethod
    def fit(cls, train, rank=32, codes=96, patch_size=7, stride=2, seed=2026):
        imgs = np.asarray(train, np.float32)
        assert imgs.ndim == 3 and imgs.shape[1:] == (32, 32)
        pca = PCA(n_components=rank, svd_solver="randomized", random_state=seed)
        pca.fit(imgs.reshape(len(imgs), -1))
        rng = np.random.default_rng(seed)
        pieces = []
        for img in imgs:
            p = patchify(img, patch_size, stride)
            contrast = p.std(axis=1)
            # Prevent background-only patches from overwhelming edge examples.
            edges = np.flatnonzero(contrast > .08)
            quiet = np.flatnonzero(contrast <= .08)
            if len(edges):
                pieces.append(p[rng.choice(edges, min(12, len(edges)), replace=False)])
            if len(quiet):
                pieces.append(p[rng.choice(quiet, min(3, len(quiet)), replace=False)])
        samples = np.concatenate(pieces).astype(np.float32)
        km = MiniBatchKMeans(
            n_clusters=codes, random_state=seed, batch_size=512,
            n_init=2, max_iter=80,
        )
        km.fit(samples)
        return cls(
            pca.mean_.reshape(32, 32).copy(),
            pca.components_.reshape(rank, 32, 32).copy(),
            km.cluster_centers_.copy(),
            imgs.copy(), patch_size, stride,
        )

    def nearest_exemplar(self, obs, mask, k=1):
        """Masked nearest-training-image retrieval. Strong ordinary control."""
        visible = np.asarray(mask, bool)
        if not np.any(visible):
            raise ValueError("at least one observed pixel is required")
        distances = ((self.examples[:, visible]-np.asarray(obs)[visible])**2).mean(axis=1)
        indices = np.argsort(distances)[:k]
        out = np.mean(self.examples[indices], axis=0)
        out[visible] = np.asarray(obs)[visible]
        return out

    def independent_patch_sampling(self, obs, mask, seed=0, top_k=8):
        """Sample patch templates independently, without iterative feedback."""
        rng = np.random.default_rng(seed)
        base = gaussian_fill(obs, mask)
        size, stride = self.patch_size, self.stride
        P = patchify(base, size, stride)
        seen = patchify(np.asarray(mask, bool).astype(float), size, stride)
        Y = patchify(obs, size, stride)
        D = self.codes
        weight = .1+3*seen
        target = (.1*P+3*seen*Y)/weight
        dist = ((target[:, None, :]-D[None, :, :])**2*weight[:, None, :]).sum(axis=2)
        kk = min(top_k, len(D))
        choices = np.argpartition(dist, kk-1, axis=1)[:, :kk]
        indices = choices[np.arange(len(P)), rng.integers(kk, size=len(P))]
        out = unpatchify(D[indices], base.shape, size, stride)
        out[mask] = obs[mask]
        return np.clip(out, 0, 1)

    def global_projection(self, image, reg=.25):
        """Re-estimate global PCA state from the *current fine image*."""
        B = self.basis.reshape(len(self.basis), -1)
        target = (image-self.mean).ravel()
        coefficients = (B@target)/(1+reg)
        return np.clip(
            (self.mean.ravel()+coefficients@B).reshape(image.shape), 0, 1,
        )

    def patch_projection(self, image, observation, mask, temperature=0.0, rng=None):
        """Choose whole learned patch states with observed-pixel data fidelity.

        Overlap consensus acts as the lateral constraint: each actual pixel
        must agree with all selected overlapping local patch states.
        """
        size, stride = self.patch_size, self.stride
        current = patchify(image, size, stride)
        seen = patchify(mask.astype(float), size, stride)
        obs = patchify(observation, size, stride)
        current_w = .20+3.0*seen
        target = (.20*current+3.0*seen*obs)/current_w
        D = self.codes
        dist = ((target[:, None, :]-D[None, :, :])**2*current_w[:, None, :]).sum(axis=2)
        if temperature > 0:
            if rng is None:
                raise ValueError("rng required when temperature > 0")
            # One structured state chosen per patch, not pixelwise noise.
            dist = dist-temperature*rng.gumbel(size=dist.shape)
        indices = np.argmin(dist, axis=1)
        projection = unpatchify(D[indices], image.shape, size, stride)
        sorted_dist = np.partition(dist, 1, axis=1)
        ambiguity = sorted_dist[:, 1]-sorted_dist[:, 0]
        return projection, float(np.median(ambiguity))

    def energy(self, x, obs, mask, patch_weight=.28, global_weight=.08):
        patch, _ = self.patch_projection(x, obs, mask)
        macro = self.global_projection(x)
        data = np.sum((x[mask]-obs[mask])**2)
        fine = np.sum((x-patch)**2)
        glob = np.sum((gaussian_filter(x, 1.4)-gaussian_filter(macro, 1.4))**2)
        return float(data+patch_weight*fine+global_weight*glob)

    def run(self, obs, mask, steps=16, feedback=True, patch=True, ping=False,
            seed=0, start=None, ping_scale=.025):
        """Alternating coarse/fine projections with clean causal ablations.

        feedback=False freezes the global state after the initial fill.
        ping=True adds local template-selection perturbations at cycles 4/8/12;
        they are only accepted when the stated energy falls. Their effect on
        ground-truth reconstruction is evaluated *afterward*, independently.
        """
        obs = np.asarray(obs, float)
        mask = np.asarray(mask, bool)
        if obs.shape != (32, 32) or mask.shape != (32, 32):
            raise ValueError("expected 32x32 observation/mask")
        x = gaussian_fill(obs, mask) if start is None else np.asarray(start, float).copy()
        x[mask] = obs[mask]
        anchor = self.global_projection(x)
        rng = np.random.default_rng(seed)
        trace = [x.copy()]
        energies = [self.energy(x, obs, mask)]
        accepts = 0
        for t in range(steps):
            global_state = self.global_projection(x) if feedback else anchor
            if patch:
                local, _ = self.patch_projection(x, obs, mask)
            else:
                local = gaussian_filter(x, .9)
            # Fine-to-coarse revision happens every cycle via global_projection(x);
            # coarse-to-fine revision enters local target on the next cycle.
            target = .65*local+.22*x+.13*global_state
            x = np.clip(.60*x+.40*target, 0, 1)
            x[mask] = obs[mask]
            if ping and t in (3, 7, 11):
                trial = x.copy()
                candidate, _ = self.patch_projection(
                    x, obs, mask, temperature=ping_scale, rng=rng,
                )
                trial[~mask] = np.clip((.88*x+.12*candidate)[~mask], 0, 1)
                if self.energy(trial, obs, mask) < self.energy(x, obs, mask):
                    x = trial
                    accepts += 1
            trace.append(x.copy())
            energies.append(self.energy(x, obs, mask))
        return {
            "image": x, "frames": np.asarray(trace),
            "energy": np.asarray(energies), "accepted_pings": accepts,
        }
