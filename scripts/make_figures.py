#!/usr/bin/env python3
"""Builds proof-of-execution figures from a pipeline run.

Creates in --output_dir:
  figures/montage.png   input vs. GPU edge map for several images
  figures/timing.png    GPU kernel vs. CPU time per image (log scale)
  figures/speedup.png   speedup as a function of megapixels

Requires: pip install pillow matplotlib
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def make_montage(input_dir, output_dir, fig_dir, max_rows=6):
    edges = sorted(glob.glob(os.path.join(output_dir, "*_edges.pgm")))
    if not edges:
        return
    step = max(1, len(edges) // max_rows)
    picks = edges[::step][:max_rows]
    fig, axes = plt.subplots(len(picks), 2, figsize=(10, 3.2 * len(picks)))
    if len(picks) == 1:
        axes = [axes]
    for row, edge_path in zip(axes, picks):
        stem = os.path.basename(edge_path)[: -len("_edges.pgm")]
        src = glob.glob(os.path.join(input_dir, stem + ".p?m"))
        if src:
            row[0].imshow(Image.open(src[0]).convert("RGB"))
        row[0].set_title(f"input: {stem}", fontsize=9)
        row[1].imshow(Image.open(edge_path), cmap="gray", vmin=0, vmax=255)
        row[1].set_title("GPU edge map", fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "montage.png"), dpi=110)
    plt.close(fig)


def make_timing_plots(output_dir, fig_dir):
    path = os.path.join(output_dir, "timing.csv")
    if not os.path.exists(path):
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    names = [r["image"].replace(".ppm", "").replace(".pgm", "") for r in rows]
    kernel = [float(r["gpu_kernel_ms"]) for r in rows]
    total = [float(r["gpu_total_ms"]) for r in rows]
    cpu = [float(r["cpu_ms"]) for r in rows]
    mpix = [int(r["width"]) * int(r["height"]) / 1e6 for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(rows))
    ax.plot(x, cpu, "o-", label="CPU reference (1 thread)")
    ax.plot(x, total, "s-", label="GPU incl. transfers")
    ax.plot(x, kernel, "^-", label="GPU kernels only")
    ax.set_yscale("log")
    ax.set_ylabel("time (ms, log scale)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=75, fontsize=7)
    ax.legend()
    ax.set_title(f"Per-image processing time ({len(rows)} images, one run)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "timing.png"), dpi=110)
    plt.close(fig)

    if any(c > 0 for c in cpu):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(mpix, [c / k for c, k in zip(cpu, kernel)],
                   label="kernels only")
        ax.scatter(mpix, [c / t for c, t in zip(cpu, total)],
                   label="incl. transfers")
        ax.set_xlabel("image size (megapixels)")
        ax.set_ylabel("speedup over CPU (x)")
        ax.legend()
        ax.set_title("GPU speedup vs. image size")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "speedup.png"), dpi=110)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="data/input")
    parser.add_argument("--output_dir", default="output")
    args = parser.parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    make_montage(args.input_dir, args.output_dir, fig_dir)
    make_timing_plots(args.output_dir, fig_dir)
    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()
