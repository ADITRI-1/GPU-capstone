// Copyright 2026 GPU Capstone Project.
//
// Minimal dependency-free reader/writer for binary Netpbm images
// (PGM "P5" grayscale and PPM "P6" RGB, 8 bits per channel).

#ifndef GPU_CAPSTONE_SRC_IMAGE_IO_H_
#define GPU_CAPSTONE_SRC_IMAGE_IO_H_

#include <string>

#include "pipeline_types.h"

namespace edge {

// Reads a P5 or P6 file into `image`. Returns false and fills `error`
// on failure.
bool ReadNetpbm(const std::string& path, Image* image, std::string* error);

// Writes a single-channel image as P5 (PGM). Returns false on failure.
bool WritePgm(const std::string& path, const Image& image,
              std::string* error);

}  // namespace edge

#endif  // GPU_CAPSTONE_SRC_IMAGE_IO_H_
