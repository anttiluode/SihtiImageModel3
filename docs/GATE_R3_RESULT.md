# Gate R3 — Results: oracle coarse signal can influence the patch reader

**Status:** completed on the clean GitHub CPU workflow. 26 tests pass. This is a privileged **evaluation-only oracle**, not a deployable inference result. Protocol: [R3 preregistration](GATE_R3_PREREG.md), committed before source. [GitHub run with images, GIF, and casewise JSON](https://github.com/anttiluode/SihtiImageModel3/actions/runs/39266818826).

## Procedure

R2's ConditionalBridge, dictionary, pairwise statistics and code initialization were not changed. R3 adds an independent benchmark with the **exact same greedy local update**, receiving either (a) the learned coarse estimate derived from visible pixels only or (b) an **oracle** where ONLY coarse blocks intersecting the mask are replaced with their true target block averages. Coarse blocks that are entirely visible are identical.

Strengths 1x, 4x and 16x were fixed before testing. Each oracle has a nonoracle comparison using the same strength, same initial code states, same 12 sweeps and same observations. There is no ping or annealing.

A second, stronger oracle independently picks each patch code using complete true target pixels, then renders those codes with the original overlapping dictionary renderer, clamping all observed pixels. That codebook reference uses the answer and is not available during normal inference. Nor is it the globally optimal code combination.

## Clean benchmark, 320 train / 32 fresh synthetic heldouts (seed 287)

Missing-region MSE; smaller is better.

| Arm | MSE |
|---|---:|
| Plain diffusion | 0.118194 |
| Learned coarse, strength 1x | 0.109336 |
| Oracle coarse, strength 1x | 0.109329 |
| Learned coarse, strength 4x | 0.106003 |
| Oracle coarse, strength 4x | 0.100820 |
| Learned coarse, strength 16x | 0.101513 |
| **Oracle coarse, strength 16x** | **0.084082** |
| Original SihtiImageModel3 | 0.082339 |
| **Target-selected fine patch dictionary, oracle** | **0.045753** |

Oracle changed hidden-region patch choices relative to the equal-strength learned arm in 1/32 images at 1x, 12/32 at 4x and **24/32 at 16x**. Relative MSE reductions are 0.006%, 4.89% and **17.17%** respectively. At 16x, this passes the preregistered diagnostic threshold of 10% MSE gain and 25% changed images.

**Interpretation:** the reader can change local decisions and improve reconstruction when supplied with privileged, accurate coarse information with sufficient strength. This refutes a blanket claim that the discrete reader is incapable of using top-down information. Nevertheless the R3 oracle remains **slightly worse** than the old Sihti3 result on these cases, despite knowing true coarse values. No normal completion advantage has been shown.

## Coarse predictions: separate genuinely invisible cells

These are the mean **per-case** block MSEs among cases with at least one block in the relevant class. Not pooled over all blocks.

| Coarse blocks | Unconditional prior MSE | Observed-conditional MSE | Cases / blocks |
|---|---:|---:|---|
| **Fully hidden** | 0.080459 | **0.042353** | 29 cases / 131 blocks |
| Partially observed | 0.065209 | **0.010557** | 30 cases / 244 blocks |
| All mask-affected | 0.066947 | **0.019016** | 32 cases / 375 blocks |

The conditional prediction improves even fully hidden blocks by about **47.4%**, but that is less dramatic than the mixed partly-visible summary. The top-level signal carries real low-frequency correlational information, without being an object-level generative representation.

## What we can and cannot conclude

The target-selected dictionary's MSE **0.045753** suggests the vocabulary can reproduce considerably more of the true fine structure than R2 inference retrieves. It is an optimistic oracle, not a strict lower bound on every allowable dictionary reconstruction. Both the conditional prediction and the code-selection/readout mechanism limit usable completion.

R3 does not simulate continuous Sihti gradient/residue dynamics, prove a working natural-image model, or demonstrate a biological ping. The true coarse signal and true patch codes are *answers leaked intentionally under named diagnostic conditions*. They must never be presented as ordinary output of the trained model.

R1 and R2 remain unmodified negative gates. We did not tune on seed 287 after seeing results.

## Run

Install requirements and test with:

    python3.13 -m pytest -q tests
    python3.13 oracle_demo.py --output runs/oracle

Your optional THINGS photo folders:

    python3.13 oracle_demo.py --train-folder train --eval-folder unseen --train-limit 500 --eval-limit 64 --show-index 8 --output runs/r3_photos

Use --eval-limit 0 for the complete unseen folder (slow), --no-old to skip the original baseline, and --show-index to choose the displayed example. Outputs: comparison.png, loop.gif and metrics.json; each oracle panel explicitly identifies its privileged input.

The next conceptual question is how to learn **accurate and influential** downward messages without access to truth. This diagnostic locates the deficiency but is not permission to tune against the heldout targets.
