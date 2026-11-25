#!/usr/bin/env bash
# End-to-end run: build, create a batch of test images, process them on the
# GPU with CPU verification, and render figures for the proof artifacts.
#
# Env overrides: NUM_IMAGES (default 30), SIGMA (1.5), THRESHOLD (0).
set -euo pipefail
cd "$(dirname "$0")"

NUM_IMAGES="${NUM_IMAGES:-30}"
SIGMA="${SIGMA:-1.5}"
THRESHOLD="${THRESHOLD:-0}"

echo "== Building =="
make

if [ -z "$(ls data/input/*.p[pg]m 2>/dev/null)" ]; then
  echo "== Generating ${NUM_IMAGES} test images =="
  python3 scripts/generate_data.py --out data/input --count "${NUM_IMAGES}"
fi

mkdir -p output
echo "== Running GPU pipeline =="
./bin/edge_detect --input_dir data/input --output_dir output \
  --sigma "${SIGMA}" --threshold "${THRESHOLD}" --verify \
  --log output/timing.csv 2>&1 | tee output/run_log.txt

echo "== Rendering figures =="
python3 scripts/make_figures.py --input_dir data/input --output_dir output \
  || echo "(figures skipped: pip install pillow matplotlib)"
echo "Done. See output/"
