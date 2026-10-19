# Gate R1 — learned relational image energy (PRE-REGISTRATION)

**Commit this before implementing or running Gate R1.** A source/architecture experiment, not a claim that SihtiBrain's conductance dynamics are physically reproduced.

## Scientific claim under test
Learned *joint* compatibility between nearby patch states should improve held-out missing-region reconstruction beyond independent patch selection, a comparable overlap-only model, and ordinary interpolation. Re-estimating a learned coarse global state after fine changes should add measurable value beyond keeping that same global state frozen. A neutral returning perturbation should improve upon equal-compute sham/no-perturbation search if it is useful.

A pass cannot be inferred from prettier GIFs, lower internal training energy, or perfect observed-pixel reproduction.

## Defined energy, not a heuristic update
The inference variables are discrete local patch codes `z_i` at an 8×8 spatial grid and a continuous coarse field `c_i` at the same grid resolution.

```
E(z,c|o,M) =
  sum_i U_i(z_i; observed pixels)
  + lambda * sum_(i,j horizontal/vertical) V_dir(z_i,z_j)
  + gamma/2 * sum_i (q(z_i)-c_i)^2
  + tau/2 * (c-mu)^T P (c-mu)
```

- `U_i` compares only **visible** pixels of the proposed patch code with the input; each pixel's contributions are normalized by coverage.
- `V_dir` combines measured train-set conditional **neighbor-code co-occurrence** (`-log P_train(z_j|z_i,dir)`) with patch overlap compatibility. This is a learned joint relationship, not the original independent dictionary.
- `q(z_i)` summarizes the code patch's central region; `mu` and regularized SPD `P` are derived solely from training images.
- Optimize by deterministic coordinate descent in `z` followed by an exact positive-definite linear solve for `c`; both are required to be non-increasing in the unperturbed energy.

A ping is a deliberately temporary, **unlabelled global field** propagated by the learned coarse linear response; it proposes a changed `z` configuration and is accepted only if the unperturbed energy decreases. This is an algorithmic *returning perturbation*, not a simulated action potential.

## Frozen dataset and cases
- 320 synthetic procedural contour training images from `data.train_set(320)` (seed=0).
- 32 untouched synthetic test images from `data.held_out(32, seed=73)`; 16 vertical gaps, 16 square occlusions, alternating and seed-specific.
- Transform full images to grayscale, 32×32; codebook 7×7, stride 4; K=32 default; global prior 8×8.
- Separate optional user-supplied 32×32 grayscale photo folders (train/eval). Disallow content-duplicate test images; evaluate the first **up to 64** by default, or all when requested. This is a *secondary* domain-shift check, not a substitute for synthetic testing.
- No GT pixels, missing target, mask-shape answer or class labels enter inference/learning. Masks themselves are available only for identifying which pixels were observed.
- No pretrained weights, transformers or external data downloads.

## Arms and evaluation
0. Old `gaussian_fill` baseline (same inputs).
1. Original SihtiImageModel3 coupled `LearnedMedium.run` baseline (same train data).
2. Nearest one and average of four full training images, matched on visible pixels (strong retrieval controls).
3. New energy model: unary + overlap-only pairwise (learned cooccurrence disabled), coarse fixed, no ping.
4. New energy model: learned directional co-occurrences on, coarse fixed, no ping.
5. New energy model: learned pairwise on, coarse **updated**, no ping.
6. Same as 5 + true coarse-mediated neutral ping.
7. Same as 5 + equal-budget **shuffled global field** sham ping.
8. Equal-sweep extra-search baseline (no ping).

Primary outcome: mean missing-pixel MSE on 32 fresh test images. Secondary: paired per-case MSE, edge continuation along occlusion boundary, whether both horizontal and vertical gap classes survive, computational cost, energy monotonicity, and visual 0→N reconstruction trajectory.

**Meaningful win** (no post-hoc threshold changing): (a) co-occurrence arm 4 improves missing-pixel MSE by ≥5% relative to overlap-only arm 3 on held-out 32, (b) coarse-updated arm 5 improves by ≥5% relative to coarse-frozen arm 4, and (c) arm 5 also improves by ≥5% over the better of original Sihti3 and four-image retrieval. Report all three as independent gates. Do not call any single pass "image model solved".

**Ping bonus gate**: true ping beats both sham ping and extra-search by ≥2% MSE. If not, record **no added value**, even if energy decreases. It may be better or worse than no-ping.

Do not increase iterations, try many seeds, change loss weights or cherry-pick mask cases after inspecting these test results. Tune on separate validation, or declare a new follow-up protocol.

## Limits
A directional co-occurrence table is a small learned Markov/random-field prior, not unique neuroscience or new theory. Global `c` inference is learned covariance geometry, not an actual soma-axon circuit. A faster energy decrease alone is not a better perceptual completion. Misleading high scores are possible when the erased region contains only black background. Always show the original, mask, results and per-case statistics.
