// Copyright 2026 GPU Capstone Project.

#include "image_io.h"

#include <cctype>
#include <cstdio>
#include <fstream>
#include <string>

namespace edge {
namespace {

// Reads the next whitespace-delimited header token, skipping '#' comments.
// Consumes exactly one whitespace character after the token, as required
// by the Netpbm spec before the binary raster.
bool ReadHeaderToken(std::istream& in, std::string* token) {
  token->clear();
  int c;
  while ((c = in.get()) != EOF) {
    if (c == '#' && token->empty()) {
      while ((c = in.get()) != EOF && c != '\n') {
      }
      continue;
    }
    if (std::isspace(c)) {
      if (!token->empty()) return true;
      continue;
    }
    token->push_back(static_cast<char>(c));
  }
  return !token->empty();
}

bool ParsePositiveInt(const std::string& token, int* value) {
  if (token.empty()) return false;
  for (char ch : token) {
    if (!std::isdigit(static_cast<unsigned char>(ch))) return false;
  }
  *value = std::stoi(token);
  return *value > 0;
}

}  // namespace

bool ReadNetpbm(const std::string& path, Image* image, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return false;
  }
  std::string magic, w_tok, h_tok, max_tok;
  if (!ReadHeaderToken(in, &magic) || !ReadHeaderToken(in, &w_tok) ||
      !ReadHeaderToken(in, &h_tok) || !ReadHeaderToken(in, &max_tok)) {
    *error = "truncated header in " + path;
    return false;
  }
  int channels = 0;
  if (magic == "P5") {
    channels = 1;
  } else if (magic == "P6") {
    channels = 3;
  } else {
    *error = "unsupported format '" + magic + "' (need P5 or P6) in " + path;
    return false;
  }
  int width = 0, height = 0, max_value = 0;
  if (!ParsePositiveInt(w_tok, &width) || !ParsePositiveInt(h_tok, &height) ||
      !ParsePositiveInt(max_tok, &max_value) || max_value > 255) {
    *error = "bad dimensions or max value (must be <= 255) in " + path;
    return false;
  }
  image->width = width;
  image->height = height;
  image->channels = channels;
  image->pixels.resize(image->NumPixels() * channels);
  in.read(reinterpret_cast<char*>(image->pixels.data()),
          static_cast<std::streamsize>(image->pixels.size()));
  if (in.gcount() != static_cast<std::streamsize>(image->pixels.size())) {
    *error = "truncated pixel data in " + path;
    return false;
  }
  return true;
}

bool WritePgm(const std::string& path, const Image& image,
              std::string* error) {
  if (image.channels != 1) {
    *error = "WritePgm expects a single-channel image";
    return false;
  }
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    *error = "cannot create " + path;
    return false;
  }
  out << "P5\n" << image.width << " " << image.height << "\n255\n";
  out.write(reinterpret_cast<const char*>(image.pixels.data()),
            static_cast<std::streamsize>(image.pixels.size()));
  if (!out) {
    *error = "write failed for " + path;
    return false;
  }
  return true;
}

}  // namespace edge
