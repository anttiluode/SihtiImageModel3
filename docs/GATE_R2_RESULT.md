# R2 — Results: conditional top-down signal, but no useful local correction

Status: **completed, all scientific image-improvement gates failed.** This document is a post-run result, not a replacement for the previously committed [R2 preregistration](GATE_R2_PREREG.md). Neither SihtiBrain nor Claude's repository was modified.

## What is genuinely different from R1

Before choosing a single missing-patch code, `conditional.py` computes an 8×8 coarse image estimate from *visible pixels alone*, plus the covariance of complete training images. Mask fractions are treated as uncertain measurements. It does **not** generate the coarse field from R1's already guessed `q(z)`.

```
c0 = argmin_c [
    prior_strength/2 * (c-mu)^T P(c-mu)
  + 1/2 * sum_i visible_weight_i*(c_i-visible_mean_i)^2
].
```

The conditional state `c0` influences local code selection in a joint discrete/continuous energy. The frozen arm holds it fixed; the live arm recomputes the same coarse field with an additional reliability-weighted bottom-up message from the current fine states. Both arms start from the **same** initial patch choices and `c0`, have the same number of local code sweeps and preserve all observed pixels. This is a true directional information test. It is *not* a physical Sihti dendritic pulse, and R2 deliberately has no ping.

A controlled 64-cell Gaussian-correlation test verifies that an observed cell sends information to a correlated unobserved cell while remaining invariant to arbitrary values placed under the mask. Local site decisions have been checked against the same joint energy; the conditioned coarse solver satisfies its stationarity condition. The clean workflow ran **20 passing tests**.

## Main receipt — clean GitHub CPU runner

Workflow: [R2 run and visual artifacts](https://github.com/anttiluode/SihtiImageModel3/actions/runs/39259538598).

Fixed evaluation: 320 synthetic training contours; 32 separate held-out contours (seed 179), alternating center vertical stripe and square occlusion, 12 identical local sweeps per arm; unchanged R1 codebook and relation-learning method. The held-out targets enter scoring only.

| Arm | Masked-pixel MSE ↓ |
| --- | ---: |
| Four nearest training examples | **0.084147** |
| Original SihtiImageModel3 | 0.091398 |
| Coarse mean, frozen | 0.130858 |
| Pairwise with coarse coupling disabled | 0.132610 |
| Coarse conditional but hidden estimates spatially shuffled | 0.133589 |
| **Conditioned top-down state, frozen** | 0.133976 |
| **Conditioned top-down state, updated from local evidence** | 0.134568 |
| Linear diffusion fill | 0.136840 |

**The upper level's conditional prediction is not trivial:**

- Coarse-block MSE on **coarse cells intersecting the hidden region**: unconditional prior 0.068875 → observation-conditioned 0.023445, a **66% reduction**.
- Conditional prediction changes when visible evidence changes; test placeholders beneath the mask don't change it.
- This is a valid lower-frequency Gaussian prediction result. It is not evidence of object understanding.

**But the downward message doesn't produce a useful finer image:**

- Conditional-frozen whole-image MSE is **2.4% worse** than unconditional-frozen (0.133976 vs 0.130858).
- Updating the coarse field from inferred patches increases MSE by another **0.4%** (0.134568 vs 0.133976).
- Live and frozen finish with different hidden-region patch codes on **1 out of 32 images**, approximately 3%, below the precommitted 10% criterion.
- The coarse state itself *does* move under fine evidence (mean coarse shift L2 0.348686), but this almost never flips local patch choices.
- All three R2 image-improvement criteria **FAIL**. The separate information-integrity plumbing gate **PASSES**.

These results are better diagnosed than R1: the top level can predict a coarse missing region from actual observations, and it mathematically accepts bottom-up evidence, but the learned coarse message does not overcome the discrete patch selector's competing energy terms. This could reflect coarse/fine coupling strength, model capacity, insufficiently multimodal patch states, scale mismatch, weak training correlations, or the coarse Gaussian prior's limited expressiveness; this run does not isolate the cause.

**Do not use the successful coarse-MSE outcome as proof that the recursive vision hypothesis succeeds.** The strong full-image baseline is still much better, and there is no ping test here.

## Reproduce and explore visually

```cmd
python3.13 -m pip install -r requirements.txt pytest
python3.13 -m pytest -q tests
python3.13 conditional_demo.py --output runs/conditional
```

The output includes `comparison.png` (original / mask / unconditional mean / **conditional prediction before patch inference** / revised coarse / final frozen and live images), `loop.gif` (observation, frozen and live side by side, across sweeps), `energy.png` and `metrics.json` (all cases, both missing-pixel and masked-gradient errors, and causal per-case diagnostics).

For your actual image folders:

```cmd
python3.13 conditional_demo.py --train-folder train --eval-folder unseen --eval-limit 64 --show-index 8 --output runs/my_conditional
```

`--show-index` selects which held-out example the picture and GIF show. `--eval-limit 0` evaluates the complete unseen folder (slow). `--no-old` skips the costly original-Sihti3 baseline, but removes that comparison. Source images are 32×32 center-cropped grayscale and duplicate resized train/evaluation examples are rejected.

## Interpreting MSE and the original idea

MSE prefers conditional means, so the experiment also records masked gradient discrepancy. Sharp coherent but alternate valid images may lose MSE; neither metric alone certifies perceptual naturalness. Do not count best-of-N using the hidden ground truth as ordinary test-time inference.

This is a limited inference model: Gaussian coarse statistics plus categorical patches and train-set pairwise code frequencies. It is not a learnt scene grammar, nonlinear soma-axon dynamics, a connected Sihti residue hierarchy, or a demonstration of image generation. Unlike R1, though, the independent top-down message is now **actually present and verified**.

The next question has become narrower: *how can a bottom-level state represent and respond to a multimodal coarse prediction instead of making an almost irreversible single template choice?* That is a different project from increasing sweeps or ping amplitude. Freeze R2 as the negative gate rather than adjust it against seed 179.
