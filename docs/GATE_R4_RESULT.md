# R4 result — directional signal helps MSE, not contour edges (MIXED / overall gate FAIL)

Pre-registration: [GATE_R4_PREREG.md](GATE_R4_PREREG.md), committed **before** implementation.
Clean CI receipt: [R4 workflow run 39275112619](https://github.com/anttiluode/SihtiImageModel3/actions/runs/39275112619). All **34 tests passed**. The workflow uploads `comparison.png`, `flux_vectors.png`, `loop.gif`, and full `metrics.json`.

## Scientific change

Previous R2/R3 conveyed the mean brightness in each 4×4 coarse block and used a discrete 32-code patch MRF. **R4 replaces the lower-level codebook with an actual continuous free-pixel field**. Its coarse signal has three channels for each 4×4 cell:

- block mean brightness / DC;
- **signed average horizontal difference** (net horizontal flux);
- **signed average vertical difference** (net vertical flux).

The 192-vector Gaussian covariance is learned only from complete training images. For partially observed images, *each feature is measured from actual visible pixels only*: an X/Y finite difference enters the conditioning system only when **both adjacent endpoints are visible**. The regularized conditional prior predicts missing signed flux. The decoder then minimizes a quadratic involving 32×32 pixel gradients, coarse means, predicted signed flux and a weak interpolation anchor, with visible pixels clamped. There is no k-means patch choice and no transformer.

The original continuous Sihti decomposition is still available and unit-tested for lossless telescoping; **R4 does not yet propagate all the original multiscale Sihti residual bands through its solver**. The decoder computes the optimum of an unconstrained positive-definite pixel energy and clips any values outside [0,1] for valid output. This is **not** a learned nonlinear dendrite or a simulated action potential.

The `FluxMedium.response` function additionally measures an explicitly **subtracted differential response** to a neutral gradient perturbation (perturbed result minus unperturbed result), rather than mixing raw pulse artifacts into a prediction channel. This is a math/susceptibility diagnostic, not a claimed image-improving ping.

## Frozen outcome

Run: 320 synthetic training contours, **32 fresh test contours seed 383**, alternating 5-column gaps and 8×8 holes. All ordinary arms use exactly the same continuous solver, observations and regularization. Oracle arms intentionally use withheld truth and must not be reported as ordinary inference.

| Arm | Missing MSE ↓ | Masked gradient error ↓ | Strong-edge cosine ↑ |
|---|---:|---:|---:|
| Gaussian interpolation / linear | 0.125129 | 0.070272 | 0.5123 |
| Plain continuous smoothing | 0.099771 | 0.055107 | 0.6486 |
| Original SihtiImageModel3 | 0.095148 | **0.051635** | 0.6295 |
| Predicted scalar brightness alone | 0.084797 | 0.053369 | **0.6552** |
| Predicted X/Y flux alone | 0.084259 | 0.055541 | 0.6139 |
| **Predicted brightness + X/Y flux** | **0.080203** | 0.056317 | 0.6168 |
| Same model with spatially shuffled signed gradient vectors | 0.154817 | 0.109547 | 0.1582 |
| Oracle true gradients on affected coarse cells | 0.044075 | 0.037379 | 0.7841 |
| Oracle true coarse scalar values | 0.052896 | 0.048840 | 0.6912 |
| Oracle true means + gradients | **0.019060** | **0.026703** | **0.9008** |

Fixed R4 gate: **FAIL**. Combining learned mean+gradient improves MSE by 5.42% vs scalar-only and far exceeds the same-model shuffled-gradient control by 48.2% MSE, but its gradient error is **5.53% worse**, and its edge-orientation score also drops. The precommitted gate required >=5% improvement in BOTH MSE and edge error against scalar-only plus >=5% MSE improvement against shuffled gradients. Only 2 of the 3 necessary conditions passed.

The large oracle gap demonstrates **potential representational and inversion capacity under privileged orientation information**, NOT success at learning the right missing contours. A 4×4 net signed gradient can cancel opposing contours; it also omits within-block phase/location. A bright/dark edge aligned exactly with a 4×4 boundary can be completely missed by the *within-block* gradient feature! This limits the message's sufficiency and should not be quietly hidden.

## Instructions

```cmd
python3.13 -m pytest -q tests
python3.13 flux_demo.py --output runs/flux_r4
python3.13 flux_demo.py --train-folder train --eval-folder unseen --train-limit 500 --eval-limit 64 --show-index 8 --output runs/flux_photos
```

Use `--eval-limit 0` for all unseen photos (slower), `--no-old` to omit original Sihti3 in exploratory mode, `--show-index` to show one particular image while evaluating the full requested set.

Outputs: `comparison.png` for all inference and explicitly privileged oracle arms, `flux_vectors.png` comparing predicted and **ground truth (oracle)** arrows, `metrics.json` with each case's MSE, masked-edge error, and orientation, and `loop.gif`. **The R4 GIF is illustrative interpolation between two solved images, NOT a simulated gradient-descent trajectory.** Do not mislabel the animation as emergent self-organization.

## Interpretation

Unlike R1–R3's discrete pipeline, this system does use a continuous gradient-sensitive pixel inverse. The signed flux carries spatially relevant information as indicated by the controlled MSE improvement and destructive shuffle control. However, it does **not** yet predict local contour orientation sharply enough for the edge metric. The results do not show hallucination, semantic parts, nonlinear branch gating, multiscale up-and-down relaxation, or image-generation quality.

The next conceptual problem is whether we need a richer observable than net gradient: e.g., **oriented energy / structure tensors, multi-resolution signed flux, or explicit edge-phase constraints**. These are hypotheses requiring their own untouched test populations, not permission to tune R4 against seed 383.
