# SihtiImageModel3 — can fine detail talk back to the core?

> **Gate R5 — latent vs image-space gradient diagnostic:** [latent_bridge_demo.py](latent_bridge_demo.py) tests whether the original Sihti3 **32-dimensional PCA decoder** can express R4's predicted pixel/gradient correction. The same visible-only flux prediction is fed into direct pixels, one/16 exact Jacobian-transpose steps and a regularized latent least-squares update; a shuffled signal is the negative control. Privileged **oracle PCA** and **oracle coarse-flux** reveal whether poor completion comes from the learned message or the decoder's limited directions. Uses fresh held-out seed **569**, records missing-pixel/edge metrics, and explicitly measures the fraction of required hidden correction outside the PCA's masked Jacobian span. **No positive result is claimed until CI is complete.** [Frozen preregistration](docs/GATE_R5_PREREG.md) · [PR #6](https://github.com/anttiluode/SihtiImageModel3/pull/6).

```bash
python latent_bridge_demo.py --output runs/latent_r5
python latent_bridge_demo.py --train-folder train --eval-folder unseen --eval-limit 64 --show-index 8 --output runs/latent_photos
```

---


> **Gate R4 — signed gradient/flux coarse messages:** [flux.py](flux.py) replaces R2/R3's k-means patch code selection with a continuous image reconstruction driven by learned conditional **X/Y block gradients plus brightness**. On 32 untouched synthetic contours, adding signed flux improved missing-pixel MSE from **0.08480 → 0.08020**, while a shuffled-gradient control scored **0.15482**. But edge-gradient error *worsened* from **0.05337 → 0.05632**; **the preregistered combined gate FAILS**. A privileged true-gradient oracle scores 0.04407 MSE; true scalar+gradient oracle scores 0.01906. No biological neuron or true image generator is claimed. See [R4 preregistration](docs/GATE_R4_PREREG.md), [R4 results](docs/GATE_R4_RESULT.md), and the [reproducible CI visuals](https://github.com/anttiluode/SihtiImageModel3/actions/runs/39275112619).

```cmd
python3.13 flux_demo.py --output runs/flux_r4
python3.13 flux_demo.py --train-folder train --eval-folder unseen --train-limit 500 --eval-limit 64 --show-index 8 --output runs/flux_photos
```

The GIF in this experiment illustrates a transition from interpolation to the solved image; it is **not** actual time-stepped cortical relaxation.

---


> **Gate R3 (oracle-coarse diagnostic):** [oracle_gate.py](oracle_gate.py) tests R2's *unchanged* patch reader after giving it the **true masked coarse structure**, for evaluation only. On 32 new synthetic contours, at a predeclared 16× top-down coupling strength, the oracle changed hidden patch choices on **24/32 images** and reduced missing-pixel MSE by **17.2% against an equally strengthened learned-message control** (0.08408 vs 0.10151). But ordinary inference still loses, and giving it exact target-selected fine patch codes can score 0.04575. The lower-level vocabulary is not entirely helpless; the learned prediction/readout combination is inadequate. No ping, new image generator or rewrite of R2 was added. Read [R3 pre-registration](docs/GATE_R3_PREREG.md) and [results](docs/GATE_R3_RESULT.md).

```cmd
python3.13 oracle_demo.py --output runs/oracle
python3.13 oracle_demo.py --train-folder train --eval-folder unseen --train-limit 500 --eval-limit 64 --show-index 8 --output runs/r3_photos
```

Visuals in comparison.png and loop.gif explicitly label all oracle panels as privileged demonstrations. Changing --show-index selects the picture but not the evaluation set.

---


> **Gate R2 (mask-conditioned coarse feedback):** [conditional.py](conditional.py) now predicts the missing 8×8 coarse structure using **visible pixels and a learned covariance, before choosing any local missing patch**. Fine-level codes can then revise that coarse prediction under one explicit energy. In the fresh 32-image test, observation-conditioning improved the error of coarse cells touching the mask by **66%**, but **did not improve image completion**: conditional frozen 0.13398, live 0.13457, vs original Sihti3 0.09140 missing-pixel MSE. The feedback changed hidden patch decisions in only **1/32** images; all image-improvement gates failed. See the [R2 preregistration](docs/GATE_R2_PREREG.md), [result](docs/GATE_R2_RESULT.md), and [passing GitHub CPU run](https://github.com/anttiluode/SihtiImageModel3/actions/runs/39259538598). This is a genuine directional message test, **not** a dendritic ping.

```cmd
python3.13 conditional_demo.py --output runs/conditional
python3.13 conditional_demo.py --train-folder train --eval-folder unseen --eval-limit 64 --show-index 8 --output runs/photos_r2
```

It produces `comparison.png`, `loop.gif`, `energy.png` and `metrics.json`. `--show-index` selects which unseen image to **show**; `--eval-limit` chooses how many to evaluate (0 = all). Earlier R1 remains unchanged for audit.

---



> **New Gate R1 (learned relational energy):** Instead of heuristic PCA/patch blending, [relational.py](relational.py) fits directional neighbor-code compatibility, overlap factors and an explicit coarse prior; both local and global updates minimize **one joint energy**. It has a side-by-side GIF demo, seven exactness/leakage tests, a precommitted [protocol](docs/GATE_R1_PREREG.md), and a [post-run negative result](docs/GATE_R1_RESULT.md). The clean CI run showed **no improvement** over original Sihti3 or simple four-neighbor retrieval: relational model 0.13145 missing MSE, original 0.08176, four-example retrieval 0.08018; coarse feedback and pulse ablations did not help. This is a more faithful **mathematical test**, not a solved image model.

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q tests
python relational_demo.py --output runs/relational
# For your image folders (evaluate 64 or use 0 for all):
python relational_demo.py --train-folder train --eval-folder unseen --eval-limit 64 --output runs/photos_relational
```

Outputs: `comparison.png`, `energy.png`, `loop.gif`, `metrics.json`. See [successful reproducible CI run](https://github.com/anttiluode/SihtiImageModel3/actions/runs/39251420922).

---


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

This first run is a deterministic **32×32 synthetic contour-world smoke test**, not proof on MNIST, Fashion-MNIST, BSDS500, CIFAR-10 or real photos.

**To try real photos without changing the program**, provide two strictly separated directories of PNG/JPG images:

```bash
python demo.py --train-folder ./my_train_photos --eval-folder ./my_unseen_photos --train 320 --output runs/photos
```

The loader converts pictures to luminance and center-crops/resizes to 32×32, then masks pixels in each held-out evaluation picture. It rejects duplicate processed images across training/evaluation. These are 32×32 greyscale completion experiments, not color generation or high-resolution restoration. A photo folder is optional; the offline synthetic demo is reproducible with no external downloads. Training uses 320 procedurally generated images; the 12 held-out images have disjoint generation seeds. Half of the targets have a missing five-pixel-wide vertical strip; half an 8×8 square. The original unmasked target enters *only* the post-hoc evaluation function.

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
