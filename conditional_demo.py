"""R2 mask-conditioned top-down prediction and reversible coarse/fine audit.

Visualizes what the upper level predicts BEFORE patch inference and whether
subsequent fine evidence actually revises that prediction. No oracle steering.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from data import train_set, held_out, damage, load_images
from conditional import ConditionalBridge
from sihti3 import LearnedMedium, gaussian_fill


def score(image, target, mask):
    missing = ~mask
    diff = image-target
    mse = float(np.mean(diff[missing]**2))
    # Edge differences on pixel pairs intersecting masked region.
    vx = np.logical_or(missing[:, 1:], missing[:, :-1])
    vy = np.logical_or(missing[1:, :], missing[:-1, :])
    dx = np.diff(diff, axis=1)
    dy = np.diff(diff, axis=0)
    edge = float((np.sum(dx[vx]**2) + np.sum(dy[vy]**2)) /
                 max(1, int(vx.sum() + vy.sum())))
    return {"missing_mse": mse, "masked_gradient_mse": edge}


def coarse_truth(target):
    return target.reshape(8, 4, 8, 4).mean(axis=(1, 3))


def load_datasets(train_folder, eval_folder, train_limit, eval_limit, test_seed):
    if (train_folder is None) != (eval_folder is None):
        raise ValueError("Supply both --train-folder and --eval-folder")
    if train_folder:
        train, _ = load_images(train_folder, max_images=train_limit)
        unseen, _ = load_images(eval_folder, max_images=eval_limit or None)
        hashes = {hashlib.sha256(x.tobytes()).digest() for x in train}
        if any(hashlib.sha256(x.tobytes()).digest() in hashes for x in unseen):
            raise ValueError("Identical resized images found in both sets")
        domain = "unseen 32x32 grayscale user photos"
    else:
        train = train_set(train_limit)
        unseen = held_out(eval_limit or 32, seed=test_seed)
        domain = "synthetic held-out contour world"
    return train, unseen, domain


def run(output="runs/conditional", train_folder=None, eval_folder=None,
        train_limit=320, eval_limit=32, test_seed=179, show_index=0, steps=12,
        compare_original=True):
    train, unseen, domain = load_datasets(
        train_folder, eval_folder, train_limit, eval_limit, test_seed)
    if not 0 <= show_index < len(unseen):
        raise ValueError(f"--show-index must be from 0 to {len(unseen)-1}")
    model = ConditionalBridge.fit(train)
    original = LearnedMedium.fit(train, seed=2026) if compare_original else None
    arms = ("mean", "shuffled", "frozen", "live", "no_coarse")
    all_cases, display = [], None
    for i, target in enumerate(unseen):
        obs, mask = damage(target, seed=test_seed*10+i,
                           kind="gap" if i % 2 == 0 else "hole")
        states = {name: model.run(obs, mask, steps=steps, seed=i, mode=name)
                  for name in arms}
        other = {
            "linear": gaussian_fill(obs, mask),
        }
        distances = np.mean((train[:, mask]-obs[mask])**2, axis=1)
        nearest = np.argsort(distances)
        other["nearest4"] = np.mean(train[nearest[:4]], axis=0)
        if original is not None:
            other["original_sihti3"] = original.run(obs, mask, steps=16)["image"]
        outputs = {**other, **{name: state["image"] for name, state in states.items()}}
        coverage = states["live"]["context"]["visible_fraction"]
        hidden_cells = (coverage.ravel() < 1)
        coarse_target = coarse_truth(target)
        cond = states["live"]["coarse_prediction"]
        mu = model.coarse_mean.reshape(8, 8)
        row = {
            "index": i, "mask_kind": "gap" if i % 2 == 0 else "hole",
            "scores": {name: score(pred, target, mask) for name, pred in outputs.items()},
            "conditional_coarse_hidden_mse": float(np.mean((cond.ravel()[hidden_cells]-coarse_target.ravel()[hidden_cells])**2)),
            "mean_coarse_hidden_mse": float(np.mean((mu.ravel()[hidden_cells]-coarse_target.ravel()[hidden_cells])**2)),
            "conditioned_vs_mean_coarse_l2": float(np.linalg.norm(cond-mu)),
            "fine_revised_coarse_l2": float(np.linalg.norm(states["live"]["coarse_frames"][-1]-cond)),
            "live_vs_frozen_hidden_patch_changes": int(np.sum(states["live"]["z"][hidden_cells] != states["frozen"]["z"][hidden_cells])),
            "live_patch_changes_by_sweep": states["live"]["changes"],
            "frozen_patch_changes_by_sweep": states["frozen"]["changes"],
            "coarse_revision_energy_nonincreasing": bool(np.max(np.diff(states["live"]["energy"])) <= 1e-8),
        }
        all_cases.append(row)
        if i == show_index:
            display = (target, obs, mask, states, outputs, model.coarse_mean.reshape(8, 8))
    means = {name: float(np.mean([row["scores"][name]["missing_mse"] for row in all_cases]))
             for name in all_cases[0]["scores"]}
    edge_means = {name: float(np.mean([row["scores"][name]["masked_gradient_mse"] for row in all_cases]))
                  for name in all_cases[0]["scores"]}
    gain = lambda x, y: 1.-means[x]/max(means[y], 1e-12)
    difference_count = sum(r["live_vs_frozen_hidden_patch_changes"] > 0 for r in all_cases)
    down_count = sum(np.abs(r["scores"]["frozen"]["missing_mse"] -
                            r["scores"]["mean"]["missing_mse"]) > 1e-12 for r in all_cases)
    report = {
        "protocol": "R2; no ping; compare frozen vs live under same learned conditioned coarse prior",
        "dataset": domain, "training_images": len(train),
        "evaluation_images": len(unseen), "steps": steps,
        "mean_missing_mse": means, "mean_masked_gradient_mse": edge_means,
        "mean_conditional_coarse_hidden_mse": float(np.mean([r["conditional_coarse_hidden_mse"] for r in all_cases])),
        "mean_unconditional_coarse_hidden_mse": float(np.mean([r["mean_coarse_hidden_mse"] for r in all_cases])),
        "downward_changed_output_count": down_count,
        "live_vs_frozen_changed_hidden_patches_images": difference_count,
        "mean_live_coarse_revision_l2": float(np.mean([r["fine_revised_coarse_l2"] for r in all_cases])),
        "gates": {
            "conditioned_frozen_vs_unconditional_5pct": gain("frozen", "mean") >= .05 and down_count > 0,
            "live_vs_frozen_patch_choice_10pct": difference_count >= .1*len(all_cases),
            "live_vs_frozen_mse_5pct": gain("live", "frozen") >= .05,
        },
        "cases": all_cases,
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target, obs, mask, states, outputs, mu = display
    figure, axs = plt.subplots(2, 5, figsize=(13, 6))
    views = [
        ("Held-out truth", target), ("Visible / masked", obs),
        ("Unconditional coarse", np.repeat(np.repeat(mu, 4, 0), 4, 1)),
        ("Conditional coarse BEFORE patches", np.repeat(np.repeat(states["live"]["coarse_prediction"], 4, 0), 4, 1)),
        ("Coarse AFTER local revision", np.repeat(np.repeat(states["live"]["coarse_frames"][-1], 4, 0), 4, 1)),
        ("Linear interpolation", outputs["linear"]),
        ("Original Sihti3" if "original_sihti3" in outputs else "Nearest4",
         outputs.get("original_sihti3", outputs["nearest4"])),
        ("Coarse mean (unconditional)", outputs["mean"]),
        ("Conditional coarse frozen", outputs["frozen"]),
        ("Conditional coarse LIVE", outputs["live"]),
    ]
    for ax, (title, img) in zip(axs.flat, views):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    figure.suptitle(f"R2: image {show_index} / {len(unseen)} — information from observed → coarse → local → revised coarse")
    figure.tight_layout()
    figure.savefig(out/"comparison.png", dpi=145)
    plt.close(figure)
    gif = []
    a = states["frozen"]["frames"]
    b = states["live"]["frames"]
    for t in range(len(a)):
        pair = np.concatenate((obs, a[t], b[t]), axis=1)
        pic = Image.fromarray(np.uint8(np.clip(pair, 0, 1)*255+.5))
        pic = pic.resize((768, 256), Image.Resampling.NEAREST).convert("RGB")
        ImageDraw.Draw(pic).text((8, 8), f"obs / frozen / live ; step {t}", fill=(255, 0, 0))
        gif.append(pic)
    gif[0].save(out/"loop.gif", save_all=True, append_images=gif[1:],
                duration=240, loop=0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(states["frozen"]["energy"], label="frozen coarse")
    ax.plot(states["live"]["energy"], label="live coarse")
    ax.set(xlabel="sweep", ylabel="joint energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"energy.png", dpi=140)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in (
        "training_images", "evaluation_images", "mean_missing_mse",
        "mean_conditional_coarse_hidden_mse", "mean_unconditional_coarse_hidden_mse",
        "live_vs_frozen_changed_hidden_patches_images",
        "mean_live_coarse_revision_l2", "gates"
    )}, indent=2))
    print(f"Visuals and full casewise results: {out}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="runs/conditional")
    p.add_argument("--train-folder")
    p.add_argument("--eval-folder")
    p.add_argument("--train-limit", type=int, default=320)
    p.add_argument("--eval-limit", type=int, default=32, help="0 = all unseen images")
    p.add_argument("--show-index", type=int, default=0, help="Which unseen image to visualize (0-based)")
    p.add_argument("--seed", type=int, default=179)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--no-old", action="store_true", help="Skip old Sihti3 baseline to save time")
    a = p.parse_args()
    run(a.output, a.train_folder, a.eval_folder, a.train_limit, a.eval_limit,
        a.seed, a.show_index, a.steps, not a.no_old)
