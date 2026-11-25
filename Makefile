# Build the batch GPU edge detector.
#   make            -> bin/edge_detect
#   make run        -> build + generate data + run (same as ./run.sh)
#   make clean
# Override the GPU architecture if needed, e.g. `make ARCH=-arch=sm_75`.

NVCC     ?= nvcc
ARCH     ?=
NVCCFLAGS ?= -O3 -std=c++17 $(ARCH) -Xcompiler -Wall
TARGET   := bin/edge_detect
SRCS     := src/main.cc src/image_io.cc src/cpu_reference.cc src/gpu_pipeline.cu
HDRS     := $(wildcard src/*.h)

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRCS)

run: $(TARGET)
	./run.sh

clean:
	rm -rf bin output

.PHONY: all run clean
