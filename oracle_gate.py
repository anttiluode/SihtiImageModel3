"""R3: explicitly privileged oracle diagnostics on R2's unchanged patch reader.

This file is not a new image model. Target pixels are used only inside
functions with an explicit 'oracle' name and in benchmark scoring.
"""
from dataclasses import dataclass, replace
import numpy as np
from conditional import ConditionalBridge
from relational import GRID, SIZE, STRIDE
from sihti3 import patchify


def true_coarse(target):
    """Full target block averages: evaluation-only privileged information."""
    arr = np.asarray(target, float)
    if arr.shape != (32, 32):
        raise ValueError("Oracle target must be 32x32")
    return arr.reshape(GRID, STRIDE, GRID, STRIDE).mean((1, 3)).ravel()


def affected_cells(mask):
    arr = np.asarray(mask, bool)
    if arr.shape != (32, 32):
        raise ValueError("Expected 32x32 observation mask")
    fractions = arr.reshape(GRID, STRIDE, GRID, STRIDE).mean((1, 3)).ravel()
    return fractions < 1.0


def oracle_hidden_coarse(conditional_coarse, target, mask):
    """Transfer GT coarse values ONLY on affected cells; never use in deployable arms."""
    c = np.asarray(conditional_coarse, float).reshape(-1).copy()
    c[affected_cells(mask)] = true_coarse(target)[affected_cells(mask)]
    return c


def oracle_target_codes(model, target, observation, mask):
    """An optimistic target-selected codebook reference, NOT an inference method.

    Chooses each patch using complete GT pixels and then renders with the normal
    overlapping-patch renderer. This is not the optimal global masked MSE.
    """
    target_patches = patchify(np.asarray(target, float), SIZE, STRIDE)
    codes = model.base.codes
    z = np.argmin(np.sum((target_patches[:, None]-codes[None])**2, axis=2), axis=1)
    return model.base._render(z, np.asarray(observation, float),
                              np.asarray(mask, bool)), z


@dataclass
class FixedCoarseDiagnostic:
    """Reuse R2 code dictionary and local energy exactly; vary only coarse message."""
    bridge: ConditionalBridge

    def infer(self, observation, mask, steps=12, strength=1.0,
              oracle_target=None):
        obs = np.asarray(observation, float)
        m = np.asarray(mask, bool)
        ctx = self.bridge.context(obs, m)
        c_conditioned = ctx["c0"].copy()
        if oracle_target is not None:
            c = oracle_hidden_coarse(c_conditioned, oracle_target, m)
            kind = "ORACLE_GT_COARSE"
        else:
            c = c_conditioned.copy()
            kind = "visible_only"
        model = replace(self.bridge, coupling_strength=
                        self.bridge.coupling_strength*float(strength))
        if strength <= 0 or not np.isfinite(strength):
            raise ValueError("strength must be finite and positive")
        # EXACT SAME initialization for oracle and nonoracle arms.
        z = model._init_codes(obs, m)
        factors = model._factors(obs, m)
        frames = [model.base._render(z, obs, m)]
        changes = []
        initial_codes = z.copy()
        for _ in range(steps):
            z, n_changed = model._sweep(z, c, factors, ctx)
            changes.append(int(n_changed))
            frames.append(model.base._render(z, obs, m))
        return {
            "image": frames[-1], "frames": np.asarray(frames),
            "z": z.copy(), "initial_codes": initial_codes,
            "changes": changes, "coarse": c.reshape(GRID, GRID).copy(),
            "kind": kind, "strength": float(strength),
            "coverage": ctx["coverage"].reshape(GRID, GRID).copy(),
        }
