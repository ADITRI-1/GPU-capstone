// Copyright 2026 GPU Capstone Project.

#include "cpu_reference.h"

#include <algorithm>
#include <cmath>

namespace edge {
namespace {

inline int Clamp(int v, int lo, int hi) {
  return std::min(std::max(v, lo), hi);
}

}  // namespace

std::vector<float> MakeGaussianWeights(float sigma) {
  const int radius =
      Clamp(static_cast<int>(std::ceil(3.0f * sigma)), 1, kMaxRadius);
  std::vector<float> weights(2 * radius + 1);
  float sum = 0.0f;
  for (int k = -radius; k <= radius; ++k) {
    const float w = std::exp(-(k * k) / (2.0f * sigma * sigma));
    weights[k + radius] = w;
    sum += w;
  }
  for (float& w : weights) w /= sum;
  return weights;
}

void CpuEdgeDetect(const Image& input, const PipelineOptions& options,
                   Image* output, std::vector<uint32_t>* histogram) {
  const int width = input.width;
  const int height = input.height;
  const size_t n = input.NumPixels();

  // 1. Grayscale (same luma weights as the GPU kernel).
  std::vector<float> gray(n);
  for (size_t i = 0; i < n; ++i) {
    if (input.channels == 1) {
      gray[i] = input.pixels[i];
    } else {
      const uint8_t* p = &input.pixels[i * input.channels];
      gray[i] = 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
    }
  }

  // 2. Separable Gaussian blur with clamp-to-edge borders.
  const std::vector<float> weights = MakeGaussianWeights(options.sigma);
  const int radius = static_cast<int>(weights.size() / 2);
  std::vector<float> temp(n), blurred(n);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -radius; k <= radius; ++k) {
        sum += weights[k + radius] *
               gray[y * width + Clamp(x + k, 0, width - 1)];
      }
      temp[y * width + x] = sum;
    }
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum = 0.0f;
      for (int k = -radius; k <= radius; ++k) {
        sum += weights[k + radius] *
               temp[Clamp(y + k, 0, height - 1) * width + x];
      }
      blurred[y * width + x] = sum;
    }
  }

  // 3. Sobel gradient magnitude, optional threshold, histogram.
  output->width = width;
  output->height = height;
  output->channels = 1;
  output->pixels.assign(n, 0);
  histogram->assign(kHistogramBins, 0);
  auto at = [&](int x, int y) {
    return blurred[Clamp(y, 0, height - 1) * width + Clamp(x, 0, width - 1)];
  };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float gx = (at(x + 1, y - 1) + 2.0f * at(x + 1, y) +
                        at(x + 1, y + 1)) -
                       (at(x - 1, y - 1) + 2.0f * at(x - 1, y) +
                        at(x - 1, y + 1));
      const float gy = (at(x - 1, y + 1) + 2.0f * at(x, y + 1) +
                        at(x + 1, y + 1)) -
                       (at(x - 1, y - 1) + 2.0f * at(x, y - 1) +
                        at(x + 1, y - 1));
      int v = static_cast<int>(std::lrint(std::sqrt(gx * gx + gy * gy)));
      v = std::min(v, 255);
      if (options.threshold > 0) v = (v >= options.threshold) ? 255 : 0;
      output->pixels[y * width + x] = static_cast<uint8_t>(v);
      ++(*histogram)[v];
    }
  }
}

}  // namespace edge
