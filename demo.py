"""One visual, reproducible held-out comparison. Produces PNG, GIF and JSON."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sihti3 import LearnedMedium, gaussian_fill
from data import train_set, held_out, damage


def score(x, y, mask):
    missing = ~mask
    return {
        "missing_mse": float(np.mean((x[missing]-y[missing])**2)),
        "missing_mae": float(np.mean(np.abs(x[missing]-y[missing]))),
    }


def export_image(path, array):
    img = Image.fromarray(np.uint8(np.clip(array, 0, 1)*255+.5), "L")
    img.resize((256*array.shape[1]//32, 256*array.shape[0]//32),
               Image.Resampling.NEAREST).save(path)


def demo(out="runs/first", seed=17, train_n=320, steps=16):
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    model = LearnedMedium.fit(train_set(train_n), seed=2026)
    targets = held_out(12, seed)
    cases = []
    shown = None
    for idx, target in enumerate(targets):
        kind = "gap" if idx % 2 == 0 else "hole"
        observation, mask = damage(target, seed=seed*10+idx, kind=kind)
        initial = gaussian_fill(observation, mask)
        arms = {
            "open_loop": {"image": initial},
            "nearest_train_image": {"image": model.nearest_exemplar(observation, mask)},
            "four_train_images": {"image": model.nearest_exemplar(observation, mask, k=4)},
            "independent_patches": {"image": model.independent_patch_sampling(observation, mask, seed=idx)},
            "no_global_feedback": model.run(observation, mask, steps, feedback=False, seed=idx),
            "no_patch_prior": model.run(observation, mask, steps, patch=False, seed=idx),
            "coupled_no_ping": model.run(observation, mask, steps, seed=idx),
            "coupled_ping": model.run(observation, mask, steps, ping=True, seed=idx),
        }
        cases.append({
            "sample": idx, "occlusion": kind,
            "arms": {name: score(v["image"], target, mask) for name, v in arms.items()},
            "ping_accepted": arms["coupled_ping"]["accepted_pings"],
        })
        if idx == 0:
            shown = (target, observation, mask, arms)
    names = list(cases[0]["arms"])
    summary = {
        "training_images": train_n, "held_out_images": len(targets),
        "steps": steps, "dataset": "synthetic contour world; disjoint training/test seeds",
        "measure": "MSE on never-seen missing pixels (lower is better)",
        "mean_missing_mse": {
            name: float(np.mean([r["arms"][name]["missing_mse"] for r in cases]))
            for name in names
        }, "cases": cases,
    }
    (folder/"metrics.json").write_text(json.dumps(summary, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Target (evaluation only)", "Observed with gap", "Linear fill",
              "Nearest exemplar", "Independent patches", "Global frozen",
              "No patch prior", "Coupled, no ping", "Coupled, ping"]
    imgs = [
        shown[0], shown[1], shown[3]["open_loop"]["image"],
        shown[3]["nearest_train_image"]["image"],
        shown[3]["independent_patches"]["image"],
        shown[3]["no_global_feedback"]["image"],
        shown[3]["no_patch_prior"]["image"],
        shown[3]["coupled_no_ping"]["image"],
        shown[3]["coupled_ping"]["image"],
    ]
    fig, axs = plt.subplots(1, len(imgs), figsize=(19, 2.9))
    for ax, img, label in zip(axs, imgs, labels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    fig.suptitle("SihtiImageModel3: held-out contours, identical visible pixels; no transformer", fontsize=12)
    fig.tight_layout()
    fig.savefig(folder/"comparison.png", dpi=140)
    plt.close(fig)

    frames = shown[3]["coupled_ping"]["frames"]
    indexes = [0, 1, 2, 4, 8, 12, len(frames)-1]
    export_image(folder/"timeline.png", np.concatenate([frames[i] for i in indexes], axis=1))
    from PIL import ImageDraw
    gif_frames = []
    for t, frame in enumerate(frames):
        im = Image.fromarray(np.uint8(np.clip(frame, 0, 1)*255+.5), "L")
        im = im.resize((320, 320), Image.Resampling.NEAREST).convert("RGB")
        ImageDraw.Draw(im).text((8, 8), f"iteration {t}", fill=(255, 0, 0))
        gif_frames.append(im)
    gif_frames[0].save(
        folder/"loop.gif", save_all=True, append_images=gif_frames[1:],
        duration=230, loop=0,
    )
    print("Missing-pixel MSE (lower is better):", summary["mean_missing_mse"])
    print("Outputs:", folder)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/first")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train", type=int, default=320)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()
    demo(args.output, args.seed, args.train, args.steps)
