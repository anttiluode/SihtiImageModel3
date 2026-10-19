# Gate R1 — Results (post-run, **not** preregistration)

R1 preregistration: [GATE_R1_PREREG.md](GATE_R1_PREREG.md), committed **before** `relational.py` was implemented.

## What changed mathematically

The previous SihtiImageModel3 blended a local patch proposal, existing pixels and a global PCA projection in a heuristic sequence. This revision adds a **true joint energy** over 64 categorical local states `z_i` and a 64-dimensional coarse state `c`:

```text
E(z,c|observed) =
    Σ_i U_i(z_i)
  + Σ_(i,j) V_orientation(z_i,z_j)
  + (gamma/2) ||q(z)-c||²
  + (tau/2) (c-mu)^T P (c-mu).
```

The `U` terms see observed pixels only, `V` is derived from train-set horizontal/vertical code co-occurrences **plus** overlap consistency, and `P` is the SPD regularized inverse covariance of training images' 8×8 coarse layouts. One local ICM site update minimizes the **exact** energy for that site with other states fixed. Each unlocked coarse update minimizes the **same** energy via a linear solve. Tests verify agreement of local deltas with global energy, coarse gradient near zero, monotonic non-increase, hidden-target isolation and identical initializations across frozen/live arms.

The neutral coarse pulse is propagated through the learned coarse response matrix and temporarily perturbs the state-selecting local energy. Its proposal is accepted only if the original unperturbed energy is lower. The sham has the same norm with shuffled spatial arrangement; extra-search has additional unperturbed sweeps. **Neither is a physical somatic action potential.**

## Clean GitHub Actions result: all scientific gates failed

The reproducible clean runner executed `python relational_demo.py --train-limit 320 --eval-limit 32 --seed 73`. It installed NumPy 2.4.6, scikit-learn 1.9.1, SciPy 1.17.1 on Python 3.11.17. The measurements below are **32 held-out synthetic images**, not natural photographs. Lower is better.

| Method | Missing-region MSE |
| --- | ---: |
| Plain linear/diffusion fill | 0.130929 |
| One nearest training image | 0.116300 |
| Four nearest training images | **0.080178** |
| Original SihtiImageModel3 coupled heuristic | 0.081758 |
| Relational model: overlap only, coarse frozen | 0.125084 |
| Relational model: learned neighbor compatibility, coarse frozen | 0.130387 |
| Relational model: learned compatibility, updated coarse | 0.131447 |
| Updated coarse + structured pulse | 0.131447 |
| Updated coarse + shuffled pulse | 0.131447 |
| Updated coarse + extra unperturbed search | 0.131447 |

**Gate decisions** against the original precommitted 5%/2% thresholds:

- Learned co-occurrence vs overlap-only: **FAIL** (error rose approximately 4.2%).
- Feedback vs frozen coarse: **FAIL** (error rose approximately 0.8%).
- Full relational model vs strongest conventional comparator: **FAIL** (0.131447 vs 0.080178).
- Neutral pulse vs sham and equal-search: **FAIL** (all exactly tied in this run).

Code does what its stated energy says. That energy is simply **not a good enough learned image distribution** on this test. This is an inference failure, not a software-test failure.

## Cross-environment sensitivity: reason for further caution

The same branch ran locally with NumPy 2.3.5 and scikit-learn 1.8.0, producing an *opposite* small comparison for the pairwise prior: 0.143273 overlap-only vs 0.132447 learned compatibility, while coarse updating produced 0.133036 and both pulse arms tied. The local advantage therefore did **not** replicate on CI. Different minibatch k-means implementations and/or floating-point behavior can change the learned dictionary and pair statistics. Exact weights, numerical libraries, and codebook formation matter at this scale.

In a local post-hoc 32-case paired analysis, learned neighbors beat overlap-only in only 14/32 images; an illustrative paired bootstrap interval for the mean MSE reduction spans zero. These are not precommitted hypothesis tests.

Do **not** claim a 7.6% success based on the favorable local result. The clean CI run is the reference receipt, and no gate passes.

## How to see it and run on your own photos

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q tests
python relational_demo.py --output runs/relational
python relational_demo.py --train-folder train --eval-folder unseen --eval-limit 64 --output runs/my_images
# --eval-limit 0: all unseen images, slower
```

The default test uses only **32** synthetic targets. With folders, the default selects up to **32 unseen images**; `--eval-limit 64` evaluates 64 and `--eval-limit 0` evaluates all. All inputs are center-cropped and resized to 32×32 grayscale; identical processed train/test images are rejected. The model uses no class labels or target pixels during inference.

Visual outputs: `comparison.png`, `loop.gif`, `energy.png`; full per-image receipts: `metrics.json`. The GitHub Actions workflow uploads those artifacts at [the successful R1 run](https://github.com/anttiluode/SihtiImageModel3/actions/runs/39251420922).

## What this does **not** establish

This is a toy conditional random-field / multiscale inverse solver with measured directional pairwise tables. It is not evidence for a new biologically faithful mechanism, a working natural-image generator, or a unique learning principle. The current local states are too limited, the conditional relation may simply encode prevalence of blank space, and a learned Gaussian coarse covariance does not create a multimodal object prior. The neutral pulse never found a useful state change on the reference run.

The experiment answers one narrower question: **does replacing independent patch choices with a learned joint energy and exact coarse-state feedback fix the first model?** Under this protocol, **no**.

A valid future study would have to learn the *conditional structured energy itself* from held-out masked reconstruction (using separate validation), represent multiple global hypotheses rather than one Gaussian coarse state, and retain all the baselines. Avoid tuning R1 on its test images.
