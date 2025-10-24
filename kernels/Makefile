SHELL := /bin/bash

NVCC ?= nvcc
NVCC_CCBIN ?= /usr/bin/g++-14
NVCC_ARCH ?= compute_86
NVCC_CODE ?= sm_86
NVCC_FLAGS ?= -arch=$(NVCC_ARCH) -code=$(NVCC_CODE) -cubin

BUILD_DIR := build
KERNEL_DIRS := $(filter-out $(BUILD_DIR),$(patsubst %/,%,$(wildcard */)))
KERNEL_TARGETS := $(addprefix $(BUILD_DIR)/,$(addsuffix .cubin,$(KERNEL_DIRS)))

.PHONY: all clean

all: $(KERNEL_TARGETS)

$(BUILD_DIR):
	@mkdir -p $@

build/%.cubin: | $(BUILD_DIR)
	@dir="$*"; \
	src=""; \
	for candidate in "$$dir"/*.cu; do \
	  if [ -f "$$candidate" ]; then \
	    src="$$candidate"; \
	    break; \
	  fi; \
	done; \
	if [ -z "$$src" ]; then \
	  echo "No CUDA source found in $$dir"; \
	  exit 1; \
	fi; \
	echo "Building $@ from $$src"; \
	NVCC_CCBIN=$(NVCC_CCBIN) $(NVCC) $(NVCC_FLAGS) "$$src" -o "$@"

clean:
	@rm -rf $(BUILD_DIR)
