# R2 preregistration — condition on what is visible, then allow revision

Recorded before implementing or executing the R2 evaluation. Previous R1 test cases are diagnostic/historical only; R2 uses fresh synthetic held-out seed 179, plus a separate synthetic positive control.

## Hypothesis
The original R1 coarse field was obtained almost entirely from the current patch guesses. In R2 the upper-level prediction **must be computed from observed pixels alone** before any fine-state inference, using training-set covariance and an uncertainty-aware observation model. This prediction is passed down to missing patches. Later fine-state evidence can revise the coarse state. Compare against frozen top-down prediction and identical optimization budgets.

## Mathematical mechanism
Learn a regularized Gaussian prior `c ~ N(mu, P^-1)` over an 8×8 field of block-mean luminance from *training images only*. For block `i`, its visible-pixel mean `b_i` and visible fraction `f_i` are computed from the original observation and mask without reading hidden pixels. Treat partially observed blocks as noisy measurements, weighted by f (more missing -> less certainty), not as fully known block means.

The **initial** state solves:

```
c0 = argmin_c  tau/2 (c-mu)^T P (c-mu)
              + 1/2 sum_i w_i(c_i-b_i)^2
where w_i = observation_strength * f_i².
```

When observations are exact on a subset V (and absent on H), Gaussian conditioning yields `c_H = mu_H - P_HH^-1 P_HV (c_V - mu_V)`; this is a special case of the observation-weighted solve, NOT a formula using true hidden pixels. A test verifies that altering arbitrary hidden placeholders never changes `c0`.

The learned patch codebook and directional neighbor compatibilities come from R1. R2 uses the energy:

```
E(z,c|o,M)= Σ U_i(z_i; o,M)
          + Σ V_direction(z_i,z_j)
          + tau/2 (c-mu)^T P(c-mu)
          + 1/2 Σ w_i(c_i-b_i)^2
          + gamma/2 Σ s_i (q(z_i)-c_i)^2.
```

`s_i` is a *fixed mask-derived reliability factor* chosen on validation, bounded below by a small constant, and independent of hidden pixel values. It gives strong upward messages when a patch has observed support and weak messages when a patch is entirely unseen; the same `s_i` appears in the downward site cost and coarse solve. Local single-site code updates and exact coarse minimization must both be valid energy descent operations, with fixed observations throughout.

## Fair arms, same initialization
- unconditional mean coarse field, frozen;
- observed-conditioned coarse field, frozen (**top-down information**);
- observed-conditioned coarse field with iterative fine-to-coarse revision (**feedback**);
- observed-conditioned but spatially scrambled missing coarse prediction, frozen (**negative control**);
- optional zero coarse/patch coupling (ablated).
- linear diffusion, original SihtiImageModel3, four nearest train exemplars as strong quality baselines.

All frozen/live arms start from the same `c0` and the same initial patch codes. Same number of complete local site-update sweeps and same codebook. No pings or temperature in R2.

## Three separately scored gates
1. **Message integrity / conditional prediction:** exact invariance to modifications of masked input placeholders; a positive-control correlated image process must yield a lower coarse-hidden MSE from a conditioned prior than an unconditional prior. Without this, stop before interpreting image results.
2. **Downward causal utility:** frozen conditioned state must alter patch choices compared with unconditional/shuffled coarse and improve missing-pixel MSE by at least 5% against unconditional on 32 fresh synthetic images. Log number of changed patch sites, not only error.
3. **Upward causal utility:** live revision must change at least 10% of images' *hidden-region patch configurations* compared with frozen, and improve average missing-pixel MSE by at least 5% compared with frozen. Check also whether the global `c` changes nontrivially; don't confuse `c\ne c0` with improved images.

Compare the live arm against the original SihtiImageModel3 and simple four-example retrieval. Report both missing-region MSE and a separate masked-region gradient/edge discrepancy to expose blur. Neither metric on its own certifies perceptual coherence. Per-image JSON, GIF and coarse prediction panels required. Retain negative outcomes.

## Evaluation and anti-leakage
320 synthetic training images using existing generator; fresh 32 held-out synthetic examples with seed 179. Alternate vertical 5-pixel gaps and 8×8 holes; fixed 12 sweeps. Separate optional user-supplied train/test folders (32×32 center-cropped grayscale), duplicate processed images rejected. Don't infer output quality from a positive-control correlated Gaussian process; that's a plumbing check, not natural image perception. The target image is used only for scoring.

No hyperparameter changes on seed 179 after viewing it, no comparing arbitrary numbers of sweeps to call a pass, and no change to the previously reported R1 numbers. If an experiment doesn't pass, archive the result with exact values and return to the core question.

## Scope boundary
Do not implement another "ping" or transformer. R2 is specifically the **conditional top-down information** test Claude suggested. The model remains a simple Gaussian / discrete-patch energy system; even success would not establish a new brain mechanism or a generative natural-image model.
