// Copyright 2026 GPU Capstone Project.
//
// Batch GPU edge detector. Processes every .ppm/.pgm image in a directory
// on the GPU, optionally verifies each result against a CPU reference,
// writes edge maps as PGM files and a per-image timing CSV.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cpu_reference.h"
#include "gpu_pipeline.h"
#include "image_io.h"
#include "pipeline_types.h"

namespace fs = std::filesystem;

namespace edge {
namespace {

// A result is "verified" if fewer than this fraction of pixels differ from
// the CPU result by more than 1 gray level (float rounding/FMA tolerance).
constexpr double kMaxMismatchFraction = 0.001;

struct CommandLine {
  std::string input_dir;
  std::string output_dir = "output";
  std::string log_path;  // Defaults to <output_dir>/timing.csv.
  PipelineOptions options;
  bool verify = false;
  bool save_outputs = true;
};

void PrintUsage(const char* program) {
  std::printf(
      "Usage: %s --input_dir DIR [options]\n\n"
      "Batch GPU edge detection (grayscale -> Gaussian blur -> Sobel).\n\n"
      "Options:\n"
      "  --input_dir DIR    Directory of .ppm (P6) / .pgm (P5) images.\n"
      "  --output_dir DIR   Where edge maps are written (default: output).\n"
      "  --sigma F          Gaussian sigma in pixels, 0.3-2.6 "
      "(default: 1.5).\n"
      "  --threshold N      0 = grayscale magnitudes, 1-255 = binary edges "
      "(default: 0).\n"
      "  --verify           Also run the CPU reference and compare.\n"
      "  --no_save          Do not write output images (timing only).\n"
      "  --log FILE         CSV timing log (default: "
      "<output_dir>/timing.csv).\n"
      "  -h, --help         Show this message.\n",
      program);
}

// Parses argv into `cmd`. Returns false (after printing why) on bad input.
bool ParseCommandLine(int argc, char** argv, CommandLine* cmd) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next_value = [&](std::string* value) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", arg.c_str());
        return false;
      }
      *value = argv[++i];
      return true;
    };
    std::string value;
    if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--input_dir") {
      if (!next_value(&cmd->input_dir)) return false;
    } else if (arg == "--output_dir") {
      if (!next_value(&cmd->output_dir)) return false;
    } else if (arg == "--log") {
      if (!next_value(&cmd->log_path)) return false;
    } else if (arg == "--sigma") {
      if (!next_value(&value)) return false;
      cmd->options.sigma = std::stof(value);
    } else if (arg == "--threshold") {
      if (!next_value(&value)) return false;
      cmd->options.threshold = std::stoi(value);
    } else if (arg == "--verify") {
      cmd->verify = true;
    } else if (arg == "--no_save") {
      cmd->save_outputs = false;
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
      return false;
    }
  }
  if (cmd->input_dir.empty()) {
    std::fprintf(stderr, "--input_dir is required\n");
    return false;
  }
  if (cmd->options.sigma < 0.3f || cmd->options.sigma > 2.6f) {
    std::fprintf(stderr, "--sigma must be in [0.3, 2.6] (radius <= %d)\n",
                 kMaxRadius);
    return false;
  }
  if (cmd->options.threshold < 0 || cmd->options.threshold > 255) {
    std::fprintf(stderr, "--threshold must be in [0, 255]\n");
    return false;
  }
  if (cmd->log_path.empty()) {
    cmd->log_path = (fs::path(cmd->output_dir) / "timing.csv").string();
  }
  return true;
}

std::vector<fs::path> ListImages(const std::string& dir) {
  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == ".ppm" || ext == ".pgm") files.push_back(entry.path());
  }
  std::sort(files.begin(), files.end());
  return files;
}

double StrongEdgeFraction(const std::vector<uint32_t>& histogram) {
  uint64_t total = 0, strong = 0;
  for (int i = 0; i < kHistogramBins; ++i) {
    total += histogram[i];
    if (i >= kStrongEdgeLevel) strong += histogram[i];
  }
  return total == 0 ? 0.0 : static_cast<double>(strong) / total;
}

// Compares two edge maps; returns max |a-b| and the fraction of pixels
// that differ by more than one gray level.
void CompareImages(const Image& a, const Image& b, int* max_diff,
                   double* mismatch_fraction) {
  *max_diff = 0;
  size_t mismatches = 0;
  for (size_t i = 0; i < a.pixels.size(); ++i) {
    const int d = std::abs(static_cast<int>(a.pixels[i]) - b.pixels[i]);
    *max_diff = std::max(*max_diff, d);
    if (d > 1) ++mismatches;
  }
  *mismatch_fraction =
      a.pixels.empty() ? 0.0
                       : static_cast<double>(mismatches) / a.pixels.size();
}

int Run(const CommandLine& cmd) {
  const std::vector<fs::path> files = ListImages(cmd.input_dir);
  if (files.empty()) {
    std::fprintf(stderr, "No .ppm/.pgm images found in %s\n",
                 cmd.input_dir.c_str());
    return EXIT_FAILURE;
  }
  fs::create_directories(cmd.output_dir);
  std::ofstream log(cmd.log_path);
  if (!log) {
    std::fprintf(stderr, "Cannot write log %s\n", cmd.log_path.c_str());
    return EXIT_FAILURE;
  }
  log << "image,width,height,channels,gpu_kernel_ms,gpu_total_ms,cpu_ms,"
         "kernel_speedup,max_abs_diff,mismatch_pct,strong_edge_pct,"
         "verified\n";

  std::printf("Device : %s\n", GpuPipeline::DescribeDevice().c_str());
  std::printf("Images : %zu from %s\n", files.size(), cmd.input_dir.c_str());
  std::printf("Params : sigma=%.2f threshold=%d verify=%s\n\n",
              cmd.options.sigma, cmd.options.threshold,
              cmd.verify ? "yes" : "no");
  std::printf("%-28s %11s %10s %10s %10s %8s %7s\n", "image", "size",
              "kernel_ms", "total_ms", "cpu_ms", "speedup", "check");

  GpuPipeline pipeline;
  double sum_kernel_ms = 0, sum_total_ms = 0, sum_cpu_ms = 0;
  double total_megapixels = 0;
  int failures = 0, processed = 0;

  for (const fs::path& path : files) {
    Image input, gpu_out;
    std::string error;
    if (!ReadNetpbm(path.string(), &input, &error)) {
      std::fprintf(stderr, "Skipping: %s\n", error.c_str());
      continue;
    }
    GpuRunStats stats;
    pipeline.Run(input, cmd.options, &gpu_out, &stats);
    ++processed;
    sum_kernel_ms += stats.kernel_ms;
    sum_total_ms += stats.total_ms;
    total_megapixels += input.NumPixels() / 1.0e6;

    double cpu_ms = 0, mismatch = 0;
    int max_diff = 0;
    bool ok = true;
    if (cmd.verify) {
      Image cpu_out;
      std::vector<uint32_t> cpu_hist;
      const auto t0 = std::chrono::steady_clock::now();
      CpuEdgeDetect(input, cmd.options, &cpu_out, &cpu_hist);
      const auto t1 = std::chrono::steady_clock::now();
      cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      sum_cpu_ms += cpu_ms;
      CompareImages(gpu_out, cpu_out, &max_diff, &mismatch);
      ok = mismatch < kMaxMismatchFraction;
      if (!ok) ++failures;
    }

    if (cmd.save_outputs) {
      const fs::path out_path =
          fs::path(cmd.output_dir) / (path.stem().string() + "_edges.pgm");
      if (!WritePgm(out_path.string(), gpu_out, &error)) {
        std::fprintf(stderr, "Write failed: %s\n", error.c_str());
      }
    }

    const std::string size =
        std::to_string(input.width) + "x" + std::to_string(input.height);
    const double speedup = cmd.verify ? cpu_ms / stats.kernel_ms : 0.0;
    std::printf("%-28s %11s %10.3f %10.3f %10.2f %7.1fx %7s\n",
                path.filename().string().c_str(), size.c_str(),
                stats.kernel_ms, stats.total_ms, cpu_ms, speedup,
                cmd.verify ? (ok ? "PASS" : "FAIL") : "-");
    log << path.filename().string() << "," << input.width << ","
        << input.height << "," << input.channels << "," << stats.kernel_ms
        << "," << stats.total_ms << "," << cpu_ms << "," << speedup << ","
        << max_diff << "," << mismatch * 100.0 << ","
        << StrongEdgeFraction(stats.histogram) * 100.0 << ","
        << (cmd.verify ? (ok ? "yes" : "no") : "skipped") << "\n";
  }

  std::printf("\n==== Summary ====\n");
  std::printf("Images processed      : %d (%.1f megapixels)\n", processed,
              total_megapixels);
  std::printf("GPU kernel time       : %.2f ms (%.1f MPix/s)\n",
              sum_kernel_ms, total_megapixels / (sum_kernel_ms / 1000.0));
  std::printf("GPU incl. transfers   : %.2f ms (%.1f MPix/s)\n", sum_total_ms,
              total_megapixels / (sum_total_ms / 1000.0));
  if (cmd.verify) {
    std::printf("CPU reference time    : %.2f ms (%.1f MPix/s)\n", sum_cpu_ms,
                total_megapixels / (sum_cpu_ms / 1000.0));
    std::printf("Kernel speedup vs CPU : %.1fx\n", sum_cpu_ms / sum_kernel_ms);
    std::printf("Verification          : %d/%d passed\n",
                processed - failures, processed);
  }
  std::printf("Timing log            : %s\n", cmd.log_path.c_str());
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace
}  // namespace edge

int main(int argc, char** argv) {
  edge::CommandLine cmd;
  if (!edge::ParseCommandLine(argc, argv, &cmd)) {
    edge::PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }
  return edge::Run(cmd);
}
