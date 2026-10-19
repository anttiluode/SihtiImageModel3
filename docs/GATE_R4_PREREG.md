# Gate R4 — directional gradient-flux bridge (PRE-REGISTERED)

Committed before implementing R4. R1–R3 and Claude's repo stay untouched.

**Hypothesis.** The scalar 4×4 block means used as R2/R3 coarse messages discard directional edge information. Test whether a training-learned conditional vector message (signed X/Y flux per 4×4 block), decoded into a continuous image through an explicit quadratic gradient energy, improves held-out completion.

**Warning:** even signed mean gradient flux can cancel opposing boundaries. It is not a complete contour representation, an actual dendritic soma current, or proof of the original Sihti hypothesis.

Train 320 synthetic contours from data.train_set. **New held-out seed 383**, 32 images with alternating stripe/hole masks; use only training full images to fit regularized Gaussian covariance of 8×8 block means plus signed within-block X/Y finite differences (192 features). Infer features from observed pixel means and differences where **both endpoints are visible**, with observation confidence proportional to coverage. Never use hidden placeholders to compute gradients; test explicitly.

A continuous 32×32 reconstruction must clamp observed pixels and solve an explicit gradient+intensity quadratic objective; verify objective descent/normal equation. Compare **equal solver, same observations**: smooth-only, predicted scalar means only, predicted gradients only, both, predicted gradients spatially shuffled (vectors shuffled as pairs), oracle true gradients on affected coarse cells, oracle true scalar brightness, oracle both. Oracle panels use hidden truth and can never be called regular image inference. Include ordinary Sihti3 as historical baseline.

Metrics: missing MSE, masked gradient error, strong-edge orientation cosine, per-case differences, fully-hidden vs partly-visible feature prediction and comparison/GIF. **Pass** only if predicted both beats scalar by at least 5% in missing MSE AND masked-gradient error, and beats shuffled gradients by at least 5% MSE. Compare both against original Sihti3 but do not redefine the pass gate afterward. No ping/annealing in R4, since the top-down content must earn its value first.

Also test `sihti3.sieve` exact telescoping reconstruction separately. The R4 model is a continuous *flux reconstruction test*, not yet a learned nonlinear multiscale image generator. If gradients provide no advantage, report that failure without retuning against seed 383.
