# Video presentation script (target: 6–8 minutes)

Record your screen (OBS, Loom, or Zoom) with Colab open. The rubric needs
**at least 5 minutes**. "Excellent" requires a live demo **and** next steps.

| Time | Show on screen | Say |
|------|----------------|-----|
| 0:00–0:45 | README top | "My capstone is a batch GPU edge detector. It takes a whole folder of images, runs grayscale, Gaussian blur and Sobel on the GPU, and checks every result against a CPU implementation. Edge detection is the first stage of many vision systems, and it is a stencil workload, which suits the GPU well." |
| 0:45–2:30 | `src/gpu_pipeline.cu` | Go through the 4 kernels. (1) One thread per pixel for grayscale. (2)/(3) Separable Gaussian: 16×16 tiles loaded into shared memory with a halo; weights in constant memory, which broadcasts to the warp; separable means 2(2r+1) operations instead of (2r+1)². (4) Sobel 3×3 on a shared tile; per-block shared histogram with atomicAdd, then one global atomic per bin. |
| 2:30–3:15 | `GpuPipeline` class, `main.cc` | RAII class that reuses device buffers across the batch. CUDA events time kernels separately from transfers. CLI arguments (`--help`). Makefile and run.sh. |
| 3:15–5:00 | Colab: run `./run.sh` live | Point at the GPU name, the per-image lines, PASS checks and the summary throughput/speedup. Then run the binary-threshold cell to show the CLI arguments working. |
| 5:00–6:00 | montage.png, timing.png, speedup.png | Explain the results. Speedup grows with image size. Kernel speedup is much larger than total speedup because PCIe transfers dominate. |
| 6:00–7:00 | README "Lessons learned" | Challenges: halo overhead at large radius; FMA makes bit-exact checks impossible, so a tolerance is needed; transfer cost. |
| 7:00–8:00 | README "Future work" | Next steps: CUDA streams + pinned memory to overlap copies; full Canny (non-max suppression + hysteresis); compare against NPP/OpenCV CUDA; video with NVDEC. |
