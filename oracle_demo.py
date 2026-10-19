"""R3 visual oracle-coarse diagnostic; no truth enters deployable inference.

Run: python oracle_demo.py --output runs/oracle
Uses existing R2 reader verbatim through FixedCoarseDiagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from conditional import ConditionalBridge
from data import train_set, held_out, damage, load_images
from oracle_gate import (FixedCoarseDiagnostic, affected_cells,
                         oracle_target_codes, oracle_hidden_coarse, true_coarse)
from sihti3 import LearnedMedium, gaussian_fill


STRENGTHS = (1, 4, 16)


def score(image, target, mask):
    diff = np.asarray(image) - np.asarray(target)
    missing = ~mask
    dx = np.diff(diff, axis=1)
    dy = np.diff(diff, axis=0)
    mx = missing[:, 1:] | missing[:, :-1]
    my = missing[1:, :] | missing[:-1, :]
    return {
        "missing_mse": float(np.mean(diff[missing]**2)),
        "masked_gradient_mse": float((np.sum(dx[mx]**2)+np.sum(dy[my]**2))/
                                     max(1, mx.sum()+my.sum())),
    }


def coarse_strata(observation_mask, prediction, target):
    coverage = observation_mask.reshape(8, 4, 8, 4).mean((1, 3)).ravel()
    qtrue = true_coarse(target)
    estimate = np.asarray(prediction).ravel()
    out = {}
    groups = {
        "fully_hidden": coverage == 0,
        "partially_observed": (coverage > 0) & (coverage < 1),
        "all_affected": coverage < 1,
    }
    for name, match in groups.items():
        count = int(np.sum(match))
        out[name] = {"count": count,
                     "mse": float(np.mean((estimate[match]-qtrue[match])**2))
                     if count else None}
    return out


def run(output="runs/oracle", train_folder=None, eval_folder=None,
        train_limit=320, eval_limit=32, seed=287, show_index=0,
        steps=12, compare_original=True):
    if (train_folder is None) != (eval_folder is None):
        raise ValueError("Supply both --train-folder and --eval-folder")
    if train_folder:
        train, trainpaths = load_images(train_folder, max_images=train_limit)
        test, testpaths = load_images(eval_folder, max_images=eval_limit or None)
        hashes = {hashlib.sha256(x.tobytes()).digest() for x in train}
        if any(hashlib.sha256(x.tobytes()).digest() in hashes for x in test):
            raise ValueError("Duplicated processed image across train/evaluation")
        domain = "user photos, center-cropped 32x32 grayscale"
    else:
        train, test = train_set(train_limit), held_out(eval_limit or 32, seed=seed)
        domain = "synthetic contours with unseen test seeds"
    if not 0 <= show_index < len(test):
        raise ValueError("show_index outside evaluation set")

    bridge = ConditionalBridge.fit(train)
    diag = FixedCoarseDiagnostic(bridge)
    old = LearnedMedium.fit(train, seed=2026) if compare_original else None
    cases = []
    shown = None
    arm_names = ["linear", "normal_1x", "oracle_1x", "normal_4x", "oracle_4x",
                 "normal_16x", "oracle_16x", "oracle_codebook"]
    if old is not None:
        arm_names.append("original_sihti3")

    for i, target in enumerate(test):
        kind = "gap" if i % 2 == 0 else "hole"
        obs, mask = damage(target, seed=seed*10+i, kind=kind)
        results = {}
        for strength in STRENGTHS:
            results[f"normal_{strength}x"] = diag.infer(
                obs, mask, steps=steps, strength=strength)
            results[f"oracle_{strength}x"] = diag.infer(
                obs, mask, steps=steps, strength=strength, oracle_target=target)
        codebook_img, codebook_ids = oracle_target_codes(bridge, target, obs, mask)
        predicted = results["normal_1x"]["coarse"]
        actual = true_coarse(target).reshape(8, 8)
        baseline_images = {
            "linear": gaussian_fill(obs, mask),
            "oracle_codebook": codebook_img,
        }
        if old is not None:
            baseline_images["original_sihti3"] = old.run(obs, mask, steps=16)["image"]
        all_images = {**baseline_images,
                      **{name: value["image"] for name, value in results.items()}}
        hidden_cells = affected_cells(mask)
        changed = {}
        for strength in STRENGTHS:
            a = results[f"normal_{strength}x"]
            b = results[f"oracle_{strength}x"]
            changed[f"{strength}x"] = int(np.sum(a["z"][hidden_cells] !=
                                                   b["z"][hidden_cells]))
        row = {
            "index": i, "mask_kind": kind,
            "scores": {name: score(img, target, mask)
                       for name, img in all_images.items()},
            "conditional_coarse": coarse_strata(mask, predicted, target),
            "prior_coarse": coarse_strata(mask, bridge.coarse_mean, target),
            "oracle_changed_affected_cell_codes": changed,
            "oracle_codebook_unique_codes": int(len(np.unique(codebook_ids))),
        }
        cases.append(row)
        if i == show_index:
            shown = (target, obs, mask, all_images, results, predicted, actual)

    averages = {name: float(np.mean([r["scores"][name]["missing_mse"] for r in cases]))
                for name in arm_names}
    edge_averages = {name: float(np.mean([r["scores"][name]["masked_gradient_mse"]
                                          for r in cases])) for name in arm_names}
    changes = {f"{s}x": sum(r["oracle_changed_affected_cell_codes"][f"{s}x"] > 0
                            for r in cases) for s in STRENGTHS}
    condition_gains = {f"{s}x": float(1.0-averages[f"oracle_{s}x"] /
                                      max(averages[f"normal_{s}x"], 1e-12))
                       for s in STRENGTHS}
    coarse_summary = {}
    for source in ("conditional_coarse", "prior_coarse"):
        coarse_summary[source] = {}
        for group in ("fully_hidden", "partially_observed", "all_affected"):
            vals = [r[source][group]["mse"] for r in cases
                    if r[source][group]["mse"] is not None]
            coarse_summary[source][group] = {
                "cases": len(vals),
                "total_cells": sum(r[source][group]["count"] for r in cases),
                "mean_per_case_mse": float(np.mean(vals)) if vals else None,
            }
    passes = {
        "oracle_alters_25pct_images_and_10pct_mse_at_matched_strength": {
            f"{s}x": (changes[f"{s}x"] >= .25*len(cases) and
                       condition_gains[f"{s}x"] >= .10)
            for s in STRENGTHS
        },
        "oracle_nearly_inert_at_all_strengths_under_10pct_changed_images": all(
            changes[f"{s}x"] < .10*len(cases) for s in STRENGTHS),
    }
    report = {
        "gate": "R3 — privileged coarse oracle, evaluation-only",
        "dataset": domain,
        "training_images": len(train), "evaluation_images": len(test),
        "steps": steps, "strengths": list(STRENGTHS),
        "primary": "masked-pixel MSE; lower is better",
        "averages": averages, "edge_averages": edge_averages,
        "conditional_prediction_mse_by_coarse_coverage": coarse_summary,
        "oracle_changed_patch_decisions_images": changes,
        "oracle_relative_mse_gain_vs_same_strength_normal": condition_gains,
        "diagnostics": passes, "cases": cases,
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target, obs, mask, img, results, predicted, coarse_actual = shown
    coarse_orig = np.repeat(np.repeat(predicted, 4, 0), 4, 1)
    coarse_oracle = np.repeat(np.repeat(
        oracle_hidden_coarse(predicted, target, mask).reshape(8, 8), 4, 0), 4, 1)
    panels = [
        ("Ground truth (oracle only)", target),
        ("Visible / gap", obs),
        ("Coarse from visible only", coarse_orig),
        ("GT coarse ON masked cells", coarse_oracle),
        ("Normal reader 1x", img["normal_1x"]),
        ("Oracle 1x", img["oracle_1x"]),
        ("Normal reader 4x", img["normal_4x"]),
        ("Oracle 4x", img["oracle_4x"]),
        ("Oracle 16x", img["oracle_16x"]),
        ("Oracle chosen patch codes", img["oracle_codebook"]),
    ]
    fig, axs = plt.subplots(2, 5, figsize=(14, 6))
    for ax, (label, arr) in zip(axs.flat, panels):
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"R3: true coarse oracle vs observation-conditioned reader | image {show_index}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out/"comparison.png", dpi=140)
    plt.close(fig)

    a = results["normal_1x"]["frames"]
    b = results["oracle_1x"]["frames"]
    c = results["oracle_16x"]["frames"]
    frames = []
    for t in range(len(a)):
        strip = np.concatenate((obs, a[t], b[t], c[t]), axis=1)
        pic = Image.fromarray(np.uint8(np.clip(strip, 0, 1)*255+.5))
        pic = pic.resize((896, 224), Image.Resampling.NEAREST).convert("RGB")
        ImageDraw.Draw(pic).text((5, 7), f"observed | learned | GT coarse 1x | GT coarse 16x ; t={t}",
                                 fill=(255, 0, 0))
        frames.append(pic)
    frames[0].save(out/"loop.gif", save_all=True,
                   append_images=frames[1:], duration=270, loop=0)
    print(json.dumps({
        "training_images": report["training_images"],
        "evaluation_images": report["evaluation_images"],
        "averages": averages,
        "coarse_strata": coarse_summary,
        "changed_images": changes,
        "oracle_relative_gain": condition_gains,
        "diagnostics": passes,
    }, indent=2))
    print(f"Saved comparison.png, loop.gif, metrics.json in {out}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/oracle")
    parser.add_argument("--train-folder")
    parser.add_argument("--eval-folder")
    parser.add_argument("--train-limit", type=int, default=320)
    parser.add_argument("--eval-limit", type=int, default=32, help="0: all images")
    parser.add_argument("--seed", type=int, default=287)
    parser.add_argument("--show-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--no-old", action="store_true", help="Skip expensive Sihti3 reference")
    args = parser.parse_args()
    run(args.output, args.train_folder, args.eval_folder, args.train_limit,
        args.eval_limit, args.seed, args.show_index, args.steps, not args.no_old)
