"""Visual and numerical R1 benchmark, without using hidden pixels at inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from data import train_set, held_out, damage, load_images
from sihti3 import LearnedMedium, gaussian_fill
from relational import RelationalMedium


def missing_mse(image, target, mask):
    return float(np.mean((image[~mask]-target[~mask])**2))


def run(output="runs/relational", train_folder=None, eval_folder=None,
        train_limit=320, eval_limit=32, test_seed=73, sweeps=12, original=True):
    if (train_folder is None) != (eval_folder is None):
        raise ValueError("Supply both train and evaluation folders")
    if train_folder:
        train, _ = load_images(train_folder, max_images=train_limit)
        test, _ = load_images(eval_folder, max_images=eval_limit or None)
        import hashlib
        fingerprints = {hashlib.sha256(x.tobytes()).digest() for x in train}
        if any(hashlib.sha256(x.tobytes()).digest() in fingerprints for x in test):
            raise ValueError("Identical processed images in train and evaluation")
        dataset = "user photos, 32x32 grayscale center crop"
    else:
        train = train_set(train_limit)
        test = held_out(eval_limit or 32, test_seed)
        dataset = "synthetic held-out contours"
    model = RelationalMedium.fit(train)
    old = LearnedMedium.fit(train, rank=32, codes=96, seed=2026) if original else None
    all_rows = []
    display = None
    arm_options = {
        "overlap_frozen": {"relation": False, "feedback": False},
        "learned_frozen": {"feedback": False},
        "learned_live": {},
        "live_ping": {"ping": True},
        "live_sham": {"sham": True},
        "live_extra_search": {"extra_search": True},
    }
    for i, target in enumerate(test):
        kind = "gap" if i % 2 == 0 else "hole"
        obs, mask = damage(target, seed=test_seed*10+i, kind=kind)
        distances = ((train[:, mask]-obs[mask])**2).mean(1)
        ordering = np.argsort(distances)
        images = {
            "linear": gaussian_fill(obs, mask),
            "nearest1": train[ordering[0]].copy(),
            "nearest4": train[ordering[:4]].mean(axis=0),
        }
        if old is not None:
            images["original_sihti3"] = old.run(obs, mask, steps=16)["image"]
        states = {name: model.infer(obs, mask, sweeps=sweeps, seed=i, **options)
                  for name, options in arm_options.items()}
        images.update({name: result["image"] for name, result in states.items()})
        all_rows.append({
            "index": i, "mask_kind": kind,
            "errors": {name: missing_mse(x, target, mask) for name, x in images.items()},
            "accepted_ping_proposals": states["live_ping"]["accepted_pings"],
            "accepted_sham_proposals": states["live_sham"]["accepted_pings"],
            "energy_nonincreasing": bool(np.max(np.diff(states["learned_live"]["energy"])) <= 1e-8),
        })
        if i == 0:
            display = (target, obs, images, states)
    names = all_rows[0]["errors"].keys()
    means = {name: float(np.mean([row["errors"][name] for row in all_rows])) for name in names}
    gain = lambda a,b: float(1 - means[a]/means[b])
    strong = min((n for n in ("nearest4", "original_sihti3") if n in means),
                 key=lambda n: means[n])
    gates = {
        "learned_neighbors_vs_overlap_5pct": gain("learned_frozen", "overlap_frozen") >= .05,
        "coarse_revision_vs_frozen_5pct": gain("learned_live", "learned_frozen") >= .05,
        "better_than_strongest_prior_5pct": gain("learned_live", strong) >= .05,
        "ping_vs_sham_extra_search_2pct": means["live_ping"] <= .98*min(
            means["live_sham"], means["live_extra_search"]),
    }
    report = {
        "dataset": dataset, "training_images": len(train), "evaluation_images": len(test),
        "steps": sweeps, "codes": model.k, "primary": "missing-pixel MSE; lower is better",
        "averages": means, "relative_gain_neighbors": gain("learned_frozen", "overlap_frozen"),
        "relative_gain_coarse": gain("learned_live", "learned_frozen"),
        "strong_baseline": strong, "pass_gates": gates, "per_case": all_rows,
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target, obs, images, states = display
    order = ["linear", "nearest4", "original_sihti3", "overlap_frozen",
             "learned_frozen", "learned_live", "live_ping", "live_sham"]
    imgs = [target, obs] + [images[n] for n in order if n in images]
    labels = ["Target (evaluation only)", "Visible data / mask"] + [
        name.replace("_", " ") for name in order if name in images]
    fig, axes = plt.subplots(2, 5, figsize=(12.5, 5.8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, label in zip(axes.flat, imgs, labels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(label, fontsize=9)
    fig.suptitle("R1: learned directional interactions, identical observations", fontsize=12)
    fig.tight_layout()
    fig.savefig(out/"comparison.png", dpi=140)
    plt.close(fig)

    frames = states["learned_live"]["frames"]
    gif = []
    for tick, img in enumerate(frames):
        im = Image.fromarray(np.uint8(np.clip(img, 0, 1)*255+.5))
        im = im.resize((256, 256), Image.Resampling.NEAREST).convert("RGB")
        ImageDraw.Draw(im).text((8, 8), f"sweep {tick}", fill=(255, 20, 20))
        gif.append(im)
    gif[0].save(out/"loop.gif", save_all=True, append_images=gif[1:], duration=250, loop=0)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(states["learned_live"]["energy"], label="learned live")
    ax.plot(states["learned_frozen"]["energy"], label="coarse frozen")
    ax.set_xlabel("sweep")
    ax.set_ylabel("explicit energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"energy.png", dpi=140)
    plt.close(fig)
    print(json.dumps({"means": means, "pass_gates": gates, "output": str(out)}, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="runs/relational")
    p.add_argument("--train-folder")
    p.add_argument("--eval-folder")
    p.add_argument("--train-limit", type=int, default=320)
    p.add_argument("--eval-limit", type=int, default=32, help="0 = all evaluation photos")
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--sweeps", type=int, default=12)
    p.add_argument("--no-old", action="store_true", help="Skip old Sihti3 comparison to save time")
    a = p.parse_args()
    run(a.output, a.train_folder, a.eval_folder, a.train_limit,
        a.eval_limit, a.seed, a.sweeps, not a.no_old)
