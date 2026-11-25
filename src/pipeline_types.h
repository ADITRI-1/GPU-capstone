// Copyright 2026 GPU Capstone Project.
//
// Shared data types and constants used by both the CPU reference
// implementation and the CUDA pipeline.

#ifndef GPU_CAPSTONE_SRC_PIPELINE_TYPES_H_
#define GPU_CAPSTONE_SRC_PIPELINE_TYPES_H_

#include <cstdint>
#include <string>
#include <vector>

namespace edge {

// Side length of the square thread block / output tile used by 2D kernels.
constexpr int kTileDim = 16;
// Largest supported Gaussian radius (limits shared-memory tile size).
constexpr int kMaxRadius = 8;
// Number of histogram bins for 8-bit gradient magnitudes.
constexpr int kHistogramBins = 256;
// Gradient magnitude at or above which a pixel counts as a "strong edge".
constexpr int kStrongEdgeLevel = 128;

// An 8-bit image stored row-major with interleaved channels (1 or 3).
struct Image {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<uint8_t> pixels;

  size_t NumPixels() const {
    return static_cast<size_t>(width) * static_cast<size_t>(height);
  }
};

// User-tunable parameters of the edge-detection pipeline.
struct PipelineOptions {
  float sigma = 1.5f;  // Gaussian standard deviation in pixels.
  int threshold = 0;   // 0 = keep magnitudes, else binarize at this level.
};

// Timing and statistics produced by one GPU pipeline run.
struct GpuRunStats {
  float kernel_ms = 0.0f;  // Time spent in the four kernels only.
  float total_ms = 0.0f;   // Kernels plus host<->device transfers.
  std::vector<uint32_t> histogram;
};

}  // namespace edge

#endif  // GPU_CAPSTONE_SRC_PIPELINE_TYPES_H_
