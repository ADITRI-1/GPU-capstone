// Copyright 2026 GPU Capstone Project.
//
// Single-threaded CPU implementation of the edge-detection pipeline.
// Used as the correctness oracle and as the speed baseline for the GPU.

#ifndef GPU_CAPSTONE_SRC_CPU_REFERENCE_H_
#define GPU_CAPSTONE_SRC_CPU_REFERENCE_H_

#include <cstdint>
#include <vector>

#include "pipeline_types.h"

namespace edge {

// Returns normalized 1D Gaussian weights of length 2 * radius + 1, where
// radius = clamp(ceil(3 * sigma), 1, kMaxRadius).
std::vector<float> MakeGaussianWeights(float sigma);

// Runs grayscale -> separable Gaussian blur -> Sobel magnitude
// (-> optional threshold) and fills `output` (1 channel) and `histogram`.
void CpuEdgeDetect(const Image& input, const PipelineOptions& options,
                   Image* output, std::vector<uint32_t>* histogram);

}  // namespace edge

#endif  // GPU_CAPSTONE_SRC_CPU_REFERENCE_H_
