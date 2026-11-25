// Copyright 2026 GPU Capstone Project.
//
// Kernels:
//   1. ToGrayKernel       - 1D grid, RGB -> float luma.
//   2. GaussianRowKernel  - 2D tiled, shared-memory halo, weights in
//                           __constant__ memory (horizontal pass).
//   3. GaussianColKernel  - same tile scheme, vertical pass.
//   4. SobelKernel        - shared-memory 3x3 stencil + per-block shared
//                           histogram merged into global with atomics.

#include "gpu_pipeline.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <vector>

#include "cpu_reference.h"

namespace edge {
namespace {

// Aborts with file/line information when a CUDA API call fails.
#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    const cudaError_t err_ = (call);                                   \
    if (err_ != cudaSuccess) {                                         \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,      \
                   __FILE__, __LINE__, cudaGetErrorString(err_));      \
      std::exit(EXIT_FAILURE);                                         \
    }                                                                  \
  } while (0)

// Shared tile side: output tile plus a halo of kMaxRadius on each side.
constexpr int kTileSpan = kTileDim + 2 * kMaxRadius;
constexpr int kGrayBlockSize = 256;

__constant__ float c_gaussian[2 * kMaxRadius + 1];

__device__ __forceinline__ int ClampIndex(int v, int lo, int hi) {
  return min(max(v, lo), hi);
}

// Cooperatively loads a (kTileDim + 2 * radius)^2 window of `src` around
// this block's output tile into shared memory, clamping at image borders.
__device__ void LoadTile(const float* __restrict__ src, int width, int height,
                         int radius, float (*tile)[kTileSpan]) {
  const int span = kTileDim + 2 * radius;
  const int origin_x = blockIdx.x * kTileDim - radius;
  const int origin_y = blockIdx.y * kTileDim - radius;
  for (int ty = threadIdx.y; ty < span; ty += blockDim.y) {
    const int gy = ClampIndex(origin_y + ty, 0, height - 1);
    for (int tx = threadIdx.x; tx < span; tx += blockDim.x) {
      const int gx = ClampIndex(origin_x + tx, 0, width - 1);
      tile[ty][tx] = src[gy * width + gx];
    }
  }
  __syncthreads();
}

__global__ void ToGrayKernel(const uint8_t* __restrict__ src,
                             float* __restrict__ dst, int num_pixels,
                             int channels) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_pixels) return;
  if (channels == 1) {
    dst[i] = src[i];
  } else {
    const uint8_t* p = src + static_cast<size_t>(i) * channels;
    dst[i] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
  }
}

__global__ void GaussianRowKernel(const float* __restrict__ src,
                                  float* __restrict__ dst, int width,
                                  int height, int radius) {
  __shared__ float tile[kTileSpan][kTileSpan];
  LoadTile(src, width, height, radius, tile);
  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y = blockIdx.y * kTileDim + threadIdx.y;
  if (x >= width || y >= height) return;
  const int lx = threadIdx.x + radius;
  const int ly = threadIdx.y + radius;
  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    sum += c_gaussian[k + radius] * tile[ly][lx + k];
  }
  dst[y * width + x] = sum;
}

__global__ void GaussianColKernel(const float* __restrict__ src,
                                  float* __restrict__ dst, int width,
                                  int height, int radius) {
  __shared__ float tile[kTileSpan][kTileSpan];
  LoadTile(src, width, height, radius, tile);
  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y = blockIdx.y * kTileDim + threadIdx.y;
  if (x >= width || y >= height) return;
  const int lx = threadIdx.x + radius;
  const int ly = threadIdx.y + radius;
  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    sum += c_gaussian[k + radius] * tile[ly + k][lx];
  }
  dst[y * width + x] = sum;
}

__global__ void SobelKernel(const float* __restrict__ src,
                            uint8_t* __restrict__ dst,
                            uint32_t* __restrict__ histogram, int width,
                            int height, int threshold) {
  __shared__ float tile[kTileSpan][kTileSpan];
  __shared__ uint32_t local_hist[kHistogramBins];
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int num_threads = blockDim.x * blockDim.y;
  for (int i = tid; i < kHistogramBins; i += num_threads) local_hist[i] = 0;
  LoadTile(src, width, height, /*radius=*/1, tile);  // Also syncs threads.

  const int x = blockIdx.x * kTileDim + threadIdx.x;
  const int y = blockIdx.y * kTileDim + threadIdx.y;
  if (x < width && y < height) {
    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;
    const float gx =
        (tile[ly - 1][lx + 1] + 2.0f * tile[ly][lx + 1] +
         tile[ly + 1][lx + 1]) -
        (tile[ly - 1][lx - 1] + 2.0f * tile[ly][lx - 1] +
         tile[ly + 1][lx - 1]);
    const float gy =
        (tile[ly + 1][lx - 1] + 2.0f * tile[ly + 1][lx] +
         tile[ly + 1][lx + 1]) -
        (tile[ly - 1][lx - 1] + 2.0f * tile[ly - 1][lx] +
         tile[ly - 1][lx + 1]);
    int v = min(__float2int_rn(sqrtf(gx * gx + gy * gy)), 255);
    if (threshold > 0) v = (v >= threshold) ? 255 : 0;
    dst[y * width + x] = static_cast<uint8_t>(v);
    atomicAdd(&local_hist[v], 1u);
  }
  __syncthreads();
  for (int i = tid; i < kHistogramBins; i += num_threads) {
    if (local_hist[i] != 0) atomicAdd(&histogram[i], local_hist[i]);
  }
}

}  // namespace

GpuPipeline::GpuPipeline() {
  CUDA_CHECK(cudaEventCreate(&start_));
  CUDA_CHECK(cudaEventCreate(&kernels_start_));
  CUDA_CHECK(cudaEventCreate(&kernels_stop_));
  CUDA_CHECK(cudaEventCreate(&stop_));
  CUDA_CHECK(cudaMalloc(&d_histogram_, kHistogramBins * sizeof(uint32_t)));
}

GpuPipeline::~GpuPipeline() {
  FreeBuffers();
  cudaFree(d_histogram_);
  cudaEventDestroy(start_);
  cudaEventDestroy(kernels_start_);
  cudaEventDestroy(kernels_stop_);
  cudaEventDestroy(stop_);
}

std::string GpuPipeline::DescribeDevice() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::ostringstream out;
  out << prop.name << " (compute " << prop.major << "." << prop.minor << ", "
      << prop.multiProcessorCount << " SMs, "
      << prop.totalGlobalMem / (1024 * 1024) << " MiB)";
  return out.str();
}

void GpuPipeline::FreeBuffers() {
  cudaFree(d_input_);
  cudaFree(d_gray_);
  cudaFree(d_temp_);
  cudaFree(d_output_);
  d_input_ = nullptr;
  d_gray_ = nullptr;
  d_temp_ = nullptr;
  d_output_ = nullptr;
  capacity_pixels_ = 0;
  capacity_channels_ = 0;
}

void GpuPipeline::EnsureCapacity(size_t num_pixels, int channels) {
  if (num_pixels <= capacity_pixels_ && channels <= capacity_channels_) {
    return;
  }
  const size_t pixels = std::max(num_pixels, capacity_pixels_);
  const int max_channels = std::max(channels, capacity_channels_);
  FreeBuffers();
  CUDA_CHECK(cudaMalloc(&d_input_, pixels * max_channels));
  CUDA_CHECK(cudaMalloc(&d_gray_, pixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_temp_, pixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output_, pixels));
  capacity_pixels_ = pixels;
  capacity_channels_ = max_channels;
}

void GpuPipeline::Run(const Image& input, const PipelineOptions& options,
                      Image* output, GpuRunStats* stats) {
  const int width = input.width;
  const int height = input.height;
  const size_t n = input.NumPixels();
  EnsureCapacity(n, input.channels);

  const std::vector<float> weights = MakeGaussianWeights(options.sigma);
  const int radius = static_cast<int>(weights.size() / 2);
  CUDA_CHECK(cudaMemcpyToSymbol(c_gaussian, weights.data(),
                                weights.size() * sizeof(float)));

  CUDA_CHECK(cudaEventRecord(start_));
  CUDA_CHECK(cudaMemcpy(d_input_, input.pixels.data(), input.pixels.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_histogram_, 0, kHistogramBins * sizeof(uint32_t)));

  CUDA_CHECK(cudaEventRecord(kernels_start_));
  const int gray_blocks =
      static_cast<int>((n + kGrayBlockSize - 1) / kGrayBlockSize);
  ToGrayKernel<<<gray_blocks, kGrayBlockSize>>>(
      d_input_, d_gray_, static_cast<int>(n), input.channels);
  const dim3 block(kTileDim, kTileDim);
  const dim3 grid((width + kTileDim - 1) / kTileDim,
                  (height + kTileDim - 1) / kTileDim);
  GaussianRowKernel<<<grid, block>>>(d_gray_, d_temp_, width, height, radius);
  GaussianColKernel<<<grid, block>>>(d_temp_, d_gray_, width, height, radius);
  SobelKernel<<<grid, block>>>(d_gray_, d_output_, d_histogram_, width,
                               height, options.threshold);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(kernels_stop_));

  output->width = width;
  output->height = height;
  output->channels = 1;
  output->pixels.resize(n);
  stats->histogram.resize(kHistogramBins);
  CUDA_CHECK(cudaMemcpy(output->pixels.data(), d_output_, n,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(stats->histogram.data(), d_histogram_,
                        kHistogramBins * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaEventRecord(stop_));
  CUDA_CHECK(cudaEventSynchronize(stop_));
  CUDA_CHECK(cudaEventElapsedTime(&stats->kernel_ms, kernels_start_,
                                  kernels_stop_));
  CUDA_CHECK(cudaEventElapsedTime(&stats->total_ms, start_, stop_));
}

}  // namespace edge
