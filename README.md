# SihtiImageModel3 — can fine detail talk back to the core?

One **CPU-only visual experiment**. No transformer, Qwen, pretrained weights or labels.

The question is whether an image is more than a core decorated with residue bands: can a learned *local* image prior, coupled through overlapping patches, repeatedly revise a *global* state and produce a better completion than ordinary methods?

> **First result: the mechanism runs, but the strong baseline wins.** Do not interpret a visible change, an energy decrease, or an accepted ping as proof of better reconstruction.

## Run and watch it

Python 3.10+:

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q tests
python demo.py --output runs/first
```

Open `runs/first/comparison.png`, `timeline.png` and **`loop.gif`**. The GIF shows all 16 successive predictions of the same held-out object, using the same unchanging visible pixels. `metrics.json` contains every held-out case and every ablation. The GitHub Actions workflow also makes the image, GIF and metrics available as a downloadable workflow artifact.

This first run is a deterministic **32×32 synthetic contour-world smoke test**, not proof on MNIST, Fashion-MNIST, BSDS500, CIFAR-10 or real photos. Training uses 320 procedurally generated images; the 12 held-out images have disjoint generation seeds. Half of the targets have a missing five-pixel-wide vertical strip; half an 8×8 square. The original unmasked target enters *only* the post-hoc evaluation function.

## What is learned, exactly?

- A 32-dimensional global PCA basis, trained on full images. Its coefficients are **re-estimated on the current reconstructed image every cycle**.
- A 96-code dictionary of 7×7 whole-patch templates, trained by minibatch k-means. Edge-bearing patches are deliberately sampled more often to avoid filling the dictionary with empty background.
- A current-state-dependent **nonlinear template selection** per overlapping patch, conditioned on visible pixels. Different patches must share the same reconstructed pixels, so their proposals compete through overlap consensus. This creates a simple learned local structural prior, not a fixed local pixel filter.

The loop is: **local reconstruction → revised global PCA state → coarse-to-fine input to the next local reconstruction → repeat**.

The baseline Sihti diffusion sieve is available as `sihti3.sieve`. The exact telescoping identity remains true. Here we do not simply copy the image's hidden true residues back: *the model does not have them*. It infers plausible patch states using a learned prior.

## What does ping mean here?

In this first model, a small, structured *code-selection perturbation* is attempted at steps 4, 8, and 12. It is constructed from learned patches, and accepted only if the same defined model energy decreases. No missing target pixel or semantic label is available.

**This is an exploratory discrete kick, not a faithful dendritic return current or an experimentally established conductance-based \(H(W)^{-1}\) ping.** The code does not simulate axons or NMDA channels. We deliberately keep the name `ping` restricted to its programmatic meaning until its causal advantage has been demonstrated.

## First run: frozen pilot results

These results were observed while building the experiment, **not preregistered**. The workflow repeats the code, and the exact per-image measurements are preserved.

| Arm | Mean missing-pixel MSE ↓ |
| --- | ---: |
| Ordinary linear diffusion into hole | 0.11143 |
| One nearest full training image (strong control) | 0.08672 |
| **Average of four nearest full training images (strong control)** | **0.05899** |
| Independent local patch sampling | 0.10178 |
| Iterative model, global feedback frozen | 0.08796 |
| Iterative model, patch prior removed | 0.08880 |
| Bidirectional coupled model, no kick | 0.08727 |
| Bidirectional coupled model, structured kick | 0.08728 |

All arms see the same unmasked pixels. No target data is used by inference. A lower number is better.

The coupled process improves on linear filling, **but** freezing global feedback only slightly changes accuracy, removing the patch prior only slightly hurts, and the coupled process loses decisively to averaging four examples. The structured kick lowers the model's internal energy when accepted, but **does not improve held-out MSE**. This is a **negative first mechanism test**, not a visual image-model breakthrough.

The 12 cases are very few, and all come from the same synthetic generator; results should not be generalized beyond that world.

## Mathematical sketch

Let `z` be the current fine image and `c` the global PCA reconstruction. For each overlapping local window `p`, the model chooses a learned patch prototype `D[k_p]` from the current image and observed pixels. The overlapping prototypes vote for one shared image.

A simplified prior is

```text
E_local(z) = sum_p min_k || P_p z - D_k ||²
E_global(z,c) = || lowpass(z) - lowpass(c) ||²
E_data(z) = || mask * (z - observation) ||²
```

The implemented step is a damped heuristic alternating projection, **not an exact gradient solver for the sum above**. Visible pixels are clamped after every update. The learned patch model makes the operation nonlinear; overlap consensus makes local proposals interact.

The global PCA code is recomputed from the new fine image on each iteration. Thus the coarse level does listen to the fine level, instead of being frozen ahead of the detail pass.

## Controls and scientific boundary

`docs/PROTOCOL.md` records the conditions and falsifiers. Code and tests check the exact Sihti sum, patch consensus on constants, no access to hidden targets, preservation of observed pixels, and active fine→coarse changes.

**Not shown:** learned physical conductance matrices, a true soma-downward pulse, unsupervised object categories, reliable hallucination of a wing, higher fidelity than simple exemplar retrieval, or meaningful generalization to natural images.

The next useful modification is not just increasing the number of iterations. It is either learning an explicitly state-dependent coupling law (with equal-information controls) or replacing nearest-template averaging with a coherent probabilistic structured-patch prior, then repeating the same benchmark. If the nearest-example baseline keeps winning, stop.

See [the original SihtiImageModel](https://github.com/anttiluode/SihtiImageModel), [SihtiImageModel2](https://github.com/anttiluode/SihtiImageModel2), and [SihtiBrain](https://github.com/anttiluode/SihtiBrain). This repository is a deliberately narrower test of the up-and-down *inference* idea.
