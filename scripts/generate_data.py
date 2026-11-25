#!/usr/bin/env python3
"""Generates a batch of synthetic RGB test images (binary PPM, P6).

Each image is a random scene of gradients, filled shapes, stripes and
noise at a range of resolutions (512x512 up to 4096x2160), so the GPU
pipeline is exercised on many images of different sizes in one run.

Usage: python3 scripts/generate_data.py --out data/input --count 30
"""

import argparse
import os

import numpy as np

SIZES = [(512, 512), (1024, 768), (1280, 720), (1920, 1080),
         (2048, 2048), (3840, 2160), (4096, 2160)]


def make_scene(width, height, rng):
    """Returns an (H, W, 3) uint8 array with a random synthetic scene."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    img = np.zeros((height, width, 3), np.float32)
    # Smooth colour gradient background.
    for c in range(3):
        a, b = rng.uniform(-1, 1, 2)
        img[..., c] = 128 + 60 * (a * xx / width + b * yy / height)
    # Random filled circles and rectangles (hard edges).
    for _ in range(rng.integers(8, 20)):
        colour = rng.uniform(0, 255, 3)
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        size = rng.uniform(0.03, 0.2) * min(width, height)
        if rng.random() < 0.5:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < size ** 2
        else:
            mask = (abs(xx - cx) < size) & (abs(yy - cy) < size * 0.6)
        img[mask] = colour
    # Sinusoidal stripe band (many parallel edges).
    period = rng.uniform(8, 40)
    band = (yy > height * 0.8) & (yy < height * 0.9)
    stripes = 127 + 127 * np.sin(2 * np.pi * xx / period)
    img[band] = stripes[band][:, None]
    # Sensor-like Gaussian noise (what the blur stage suppresses).
    img += rng.normal(0, rng.uniform(2, 12), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def write_ppm(path, rgb):
    height, width, _ = rgb.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(rgb.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/input")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for i in range(args.count):
        width, height = SIZES[i % len(SIZES)]
        path = os.path.join(args.out, f"scene_{i:03d}_{width}x{height}.ppm")
        write_ppm(path, make_scene(width, height, rng))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
