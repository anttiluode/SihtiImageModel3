"""R5 diagnostic: image-space R4 flux vs Sihti3 PCA latent correction.

Ordinary arms never use withheld pixels. Only `oracle_*`, unreachable
fraction and metrics see a complete held-out target.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from data import train_set, held_out, damage, load_images
from flux import FluxMedium, image_features
from flux_demo import evaluate
from latent_bridge import LatentPCADecoder
from sihti3 import gaussian_fill


def _dataset(train_folder, eval_folder, train_limit, eval_limit, seed):
    if (train_folder is None) != (eval_folder is None):
        raise ValueError("Supply BOTH --train-folder and --eval-folder")
    if train_folder:
        train, _ = load_images(train_folder, max_images=train_limit)
        test, _ = load_images(eval_folder, max_images=eval_limit or None)
        a = {hashlib.sha256(x.tobytes()).digest() for x in train}
        if any(hashlib.sha256(x.tobytes()).digest() in a for x in test):
            raise ValueError("Duplicate processed training/evaluation image")
        domain = "disjoint user grayscale photos, 32x32"
    else:
        train = train_set(train_limit)
        test = held_out(eval_limit or 32, seed=seed)
        domain = "fresh synthetic contours, 32x32"
    return train, test, domain


def evaluate_case(decoder, flux, target, mask_kind, seed):
    """Ordinary predictions are built before any ground-truth oracle."""
    obs, mask = damage(target, seed=seed, kind=mask_kind)
    message, _ = flux.predict(obs, mask)
    pixel = flux.solve(obs, mask, message, mode="both")
    z0 = decoder.latent_from_visible(obs, mask)
    ordinary = {
        "gaussian": gaussian_fill(obs, mask),
        "latent_visible": decoder.compose(z0, obs, mask),
        "flux_pixel": pixel,
        "latent_pullback_1": decoder.correction(obs, mask, pixel, steps=1),
        "latent_pullback_16": decoder.correction(obs, mask, pixel, steps=16),
        "latent_preconditioned": decoder.correction(
            obs, mask, pixel, method="preconditioned"),
        "latent_shuffled": decoder.shuffled_correction(
            obs, mask, pixel, seed=seed + 91),
    }
    # PRIVILEGED: all operations below explicitly use target (only diagnostics).
    affected = np.flatnonzero(mask.reshape(8,4,8,4).mean((1,3)).ravel() < 1)
    privileged = message.copy()
    actual_features = image_features(target)
    privileged[affected] = actual_features[affected]
    privileged[64 + affected] = actual_features[64 + affected]
    privileged[128 + affected] = actual_features[128 + affected]
    oracles = {
        "oracle_latent": decoder.oracle_best(target, obs, mask),
        "oracle_flux_message": flux.solve(obs, mask, privileged, mode="both"),
    }
    scores = {name: evaluate(out, target, mask)
              for name, out in {**ordinary, **oracles}.items()}
    return {
        "scores": scores,
        "unreachable_hidden_correction_fraction": decoder.unreachable_fraction(
            target, obs, mask),
        "raw_visible_latent_fit_mse": float(
            np.mean((decoder.decode(z0)[mask] - obs[mask])**2)),
    }, (obs, mask, {**ordinary, **oracles})


def run(output="runs/latent_r5", train_folder=None, eval_folder=None,
        train_limit=320, eval_limit=32, seed=569, show_index=0):
    train, test, domain = _dataset(
        train_folder, eval_folder, train_limit, eval_limit, seed)
    if not 0 <= show_index < len(test):
        raise ValueError("--show-index outside evaluation images")
    decoder = LatentPCADecoder.fit(train, rank=32, seed=2026)
    flux = FluxMedium.fit(train)
    rows = []
    shown = None
    for i, target in enumerate(test):
        mask_kind = "gap" if i%2 == 0 else "hole"
        report, images = evaluate_case(
            decoder, flux, target, mask_kind, seed*10 + i)
        report.update(index=i, mask=mask_kind)
        rows.append(report)
        if i == show_index:
            shown = (target, *images)

    names = list(rows[0]["scores"])
    fields = ("mse", "masked_gradient_mse", "strong_edge_cosine")
    results = {
        arm: {metric: float(np.mean([r["scores"][arm][metric]
                                    for r in rows])) for metric in fields}
        for arm in names
    }
    report = {
        "gate": "R5 -- R4 predicted pixel flux into Sihti3 PCA latent Jacobian",
        "domain": domain, "train_images": len(train), "test_images": len(test),
        "seed": seed, "pca_rank": decoder.rank,
        "oracle_warning": "oracle_latent and oracle_flux_message use hidden truth; not deployable",
        "ordinary_message": "R4 visible-conditioned means + signed 8x8 flux",
        "average_unreachable_hidden_correction_fraction": float(
            np.mean([r["unreachable_hidden_correction_fraction"]
                     for r in rows])),
        "average_raw_visible_latent_fit_mse": float(
            np.mean([r["raw_visible_latent_fit_mse"] for r in rows])),
        "scores": results,
        "cases": rows,
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(report, indent=2)+"\n",
                                     encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target, obs, mask, images = shown
    panels = [
        ("True (evaluation only)", target),
        ("Observed", obs),
        ("Gaussian", images["gaussian"]),
        ("Visible-fit PCA", images["latent_visible"]),
        ("R4 pixel flux", images["flux_pixel"]),
        ("Jacobian transpose 1x", images["latent_pullback_1"]),
        ("Jacobian transpose 16x", images["latent_pullback_16"]),
        ("Latent least squares", images["latent_preconditioned"]),
        ("ORACLE true-hidden PCA", images["oracle_latent"]),
        ("ORACLE true coarse flux", images["oracle_flux_message"]),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for ax, (label, img) in zip(axes.flat, panels):
        ax.imshow(img, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.suptitle("R5: actual PCA decoder / J^T / flux / explicitly privileged ceilings")
    fig.tight_layout()
    fig.savefig(out/"comparison.png", dpi=140)
    plt.close(fig)
    print(json.dumps({
        "test_images": len(test),
        "scores": results,
        "average_unreachable_fraction":
            report["average_unreachable_hidden_correction_fraction"],
        "output": str(out)}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/latent_r5")
    parser.add_argument("--train-folder")
    parser.add_argument("--eval-folder")
    parser.add_argument("--train-limit", type=int, default=320)
    parser.add_argument("--eval-limit", type=int, default=32, help="0=all")
    parser.add_argument("--seed", type=int, default=569)
    parser.add_argument("--show-index", type=int, default=0)
    args = parser.parse_args()
    run(args.output, args.train_folder, args.eval_folder,
        args.train_limit, args.eval_limit, args.seed, args.show_index)
