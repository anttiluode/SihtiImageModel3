# Gate R5 — does image-space flux fail because the decoder cannot represent its direction?

Protocol frozen before R5 code. The earlier R1–R4 results remain untouched.

## Architecture being tested
R4 (`flux.py`) is a **continuous pixel-space convex solver**, not a latent neural decoder; it is a control. The actual learned continuous global latent decoder in `sihti3.py` is **32-dimensional PCA**, with `D(z)=mean+B.T@z` and a constant Jacobian `J=B.T`. R2/R3 discrete patch-code selection is not differentiable. R5 tests the PCA decoder (fitted using the same randomized PCA configuration as Sihti3), rather than pretending the R4 solver has latent coordinates.

## Frozen evaluation
- Fit on **320** synthetic 32×32 contour images, evaluate **32 new unseen contours using seed 569**, alternating R4's `gap` and `hole` mask definitions. Existing R4 seed 383 is not reused for the primary score.
- Optional disjoint user-photo train/evaluation folders. Exact duplicate processed images rejected.
- Only visible pixels are available to the ordinary predictor. The missing truth enters oracle and post-hoc scoring **only**. Visible pixel outputs are clamped identically.
- Construct a fixed R4 ordinary message from visible pixels, and a fixed R4 pixel-space prediction from it. No parameter search on held-out examples.
- Fit initial PCA coefficients by ridge least squares against **visible pixels**. Use the *same* initial PCA state and the *same* R4 pixel correction signal for latent arms.
- Compare: Gaussian, initial visible-fit PCA, R4 pixel solver, one Jacobian transpose step, 16 Jacobian transpose steps (unit step), exact ridge/pseudoinverse preconditioned latent correction, and an identically powered shuffled correction control.
- Privileged ceilings: optimal hidden-pixel PCA reconstruction chosen with *true hidden pixels* (best decoder expressivity for this mask, not deployable inference), R4 oracle gradient+scalar coarse message, and the fraction of missing target correction orthogonal to the PCA decoder's *hidden-pixel Jacobian column space*.
- Primary: hidden-pixel MSE. Secondary: masked spatial-gradient MSE; strong-edge cosine. All values are reported even when gates fail.

## Hypotheses and falsifiers
1. **Latent expressivity**: if the oracle PCA hidden-pixel floor remains high even though oracle-flux reconstruction is good, the constrained 32-dimensional global latent representation lacks relevant spatial directions. If the PCA oracle floor is low, a representation-only explanation is not supported.
2. **Message quality**: if the unconstrained R4 pixel result beats every latent correction (with identical R4 message), projection loses useful directions; but if both are poor and oracle R4 works, the learned message is inaccurate/insufficient.
3. **No miracle pullback**: with a linear orthonormal PCA decoder, a Jacobian transpose step is just a projection of the supplied image-space correction. It cannot generate out-of-subspace edge detail. This is a mathematical sanity gate, not a hoped-for positive result.
4. **Controls**: shuffled gradient directions and masked-pixel placeholder changes must not produce a false success. Any oracle is prominently labeled, never blended into normal inference.

Do not claim that R5 validates a biological dendritic learning law or a neural generative manifold. These experiments isolate coordinate compatibility, observable sufficiency, and expressivity for this decoder and synthetic contour data.
