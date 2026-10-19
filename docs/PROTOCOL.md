# Initial visual-loop protocol and result ledger

**Status:** exploratory pilot; written *after* first code run. Do not call it preregistered.

## Question

Can learned overlapping local patch constraints, repeatedly interacting with a re-estimated global image state, reconstruct withheld 32×32 contour pixels more accurately than a one-pass fill or strong non-dynamical memory controls? Does a structured 'kick' contribute?

## Fixed demonstration

- Synthetic 32×32 procedural contour world (five line/shape families), 320 train images, 12 disjoint held-out images.
- Masks alternate between an erased vertical 5-column band and an erased 8×8 square.
- Images are grayscale; 32 PCA modes, 96 minibatch kmeans local templates (7×7, stride 2), 16 iterations, three scheduled proposals.
- Fixed global and local mixing weights in `sihti3.py`; no hyperparameter tuning on held-out cases.
- The missing image pixels are only used by the scoring function and nowhere in reconstruction.
- Outcome: average missing-region MSE (lower better), plus qualitative contour coherence shown for a representative held-out case.

## Controls

1. Plain Gaussian/diffusion interpolation into missing pixels.
2. One and four nearest **training images**, compared using only observed pixels.
3. Independent learned patch-code sampling, without feedback.
4. Identical iterative model but initial global estimate frozen.
5. Identical iterative model with patch prior replaced by simple Gaussian local smoothing.
6. Full coupled model without ping, and with structured selected-template ping.

The 'ping' is a probabilistic template-choice perturbation, accepted if internal patch/lowpass energy falls. It is not a physical returning current. Measure separately its causal effect on held-out MSE.

## First measured outcome

The initial 12 cases gave MSE: linear 0.11143; one nearest 0.08672; four nearest 0.05899; independent patches 0.10178; global-frozen 0.08796; no patch 0.08880; coupled no ping 0.08727; coupled ping 0.08728.

**Inference:** no evidence so far that repeated global/local feedback provides an important additional benefit on this generator. An ordinary four-exemplar mean wins clearly. A smaller value of the model's own objective does not imply better missing-pixel predictions. The ping arm ties the no-ping arm.

## Rules for follow-ups

- Keep the held-out seed fixed only as a diagnostic; tune on distinct training/validation seeds. Evaluate on fresh untouched seeds at every claimed improvement.
- Show every ablation, the training-example baseline, and both in-paint masks.
- Do not use the target or hidden mask as a steering signal.
- A plausible-looking GIF can accompany, not replace, quantitative tests.
- Distinguish learned local patch priors from learned *conductance* or nonlinear dendritic physiology.
- Before claiming better inference, establish meaningful held-out improvement over four-example retrieval and over the strongest no-feedback ablation, with uncertainty intervals across multiple generators.
