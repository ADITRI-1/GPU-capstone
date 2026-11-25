// Copyright 2026 GPU Capstone Project.
//
// CUDA implementation of the edge-detection pipeline. A single
// GpuPipeline object owns reusable device buffers so that a large batch of
// images can be processed without re-allocating GPU memory per image.

#ifndef GPU_CAPSTONE_SRC_GPU_PIPELINE_H_
#define GPU_CAPSTONE_SRC_GPU_PIPELINE_H_

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "pipeline_types.h"

namespace edge {

class GpuPipeline {
 public:
  GpuPipeline();
  ~GpuPipeline();

  GpuPipeline(const GpuPipeline&) = delete;
  GpuPipeline& operator=(const GpuPipeline&) = delete;

  // Returns a one-line description of the active CUDA device.
  static std::string DescribeDevice();

  // Processes `input` on the GPU and writes the 1-channel edge map to
  // `output`. Timing and the magnitude histogram are written to `stats`.
  void Run(const Image& input, const PipelineOptions& options, Image* output,
           GpuRunStats* stats);

 private:
  // Grows device buffers (never shrinks) to hold an image of this size.
  void EnsureCapacity(size_t num_pixels, int channels);
  void FreeBuffers();

  uint8_t* d_input_ = nullptr;   // Interleaved 8-bit input.
  float* d_gray_ = nullptr;      // Grayscale, later reused for blur output.
  float* d_temp_ = nullptr;      // Horizontal-blur intermediate.
  uint8_t* d_output_ = nullptr;  // Final 8-bit edge map.
  uint32_t* d_histogram_ = nullptr;
  size_t capacity_pixels_ = 0;
  int capacity_channels_ = 0;

  cudaEvent_t start_ = nullptr;
  cudaEvent_t kernels_start_ = nullptr;
  cudaEvent_t kernels_stop_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

}  // namespace edge

#endif  // GPU_CAPSTONE_SRC_GPU_PIPELINE_H_
