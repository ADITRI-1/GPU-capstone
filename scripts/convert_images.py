#!/usr/bin/env python3
"""Converts ordinary images (jpg/png/tiff/...) into PPM for the pipeline.

Usage: python3 scripts/convert_images.py --src my_photos --out data/input
Requires: pip install pillow
"""

import argparse
import os

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", default="data/input")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for root, _, files in os.walk(args.src):
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() not in EXTS:
                continue
            stem = os.path.splitext(name)[0]
            dst = os.path.join(args.out, stem + ".ppm")
            Image.open(os.path.join(root, name)).convert("RGB").save(dst)
            print(f"wrote {dst}")


if __name__ == "__main__":
    main()
