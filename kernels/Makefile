SHELL := /bin/bash

NVCC ?= nvcc
NVCC_CCBIN ?= /usr/bin/g++-14
NVCC_ARCH ?= compute_86
NVCC_CODE ?= sm_86
NVCC_CUBIN_FLAGS ?= -arch=$(NVCC_ARCH) -code=$(NVCC_CODE) -cubin
NVCC_PTX_FLAGS ?= -arch=$(NVCC_ARCH) -ptx

# nvdisasm vars (override if you want extra flags)
NVDISASM ?= nvdisasm
# Commonly useful flags:
#  --print-line-info : include source/line mappings (if available)
#  --print-code      : print SASS (on by default for cubin)
#  -hex             : append encodings as hex comments
NVDISASM_FLAGS ?= --print-line-info -hex

BUILD_DIR := build
KERNEL_DIRS := $(filter-out $(BUILD_DIR),$(patsubst %/,%,$(wildcard */)))
KERNEL_TARGETS := $(addprefix $(BUILD_DIR)/,$(addsuffix .cubin,$(KERNEL_DIRS)))
SASS_TARGETS := $(KERNEL_TARGETS:.cubin=.sass)
PTX_TARGETS := $(KERNEL_TARGETS:.cubin=.ptx)

.PHONY: all clean sass ptx cubin

all: $(KERNEL_TARGETS) $(SASS_TARGETS) $(PTX_TARGETS)
sass: $(SASS_TARGETS)
ptx:  $(PTX_TARGETS)
cubin: $(KERNEL_TARGETS)

$(BUILD_DIR):
	@mkdir -p $@

# Build a .cubin and, as separate targets, the .ptx and .sass dumps
build/%.cubin: | $(BUILD_DIR)
	@dir="$*"; \
	src=""; \
	for candidate in "$$dir"/*.cu; do \
	  if [ -f "$$candidate" ]; then src="$$candidate"; break; fi; \
	done; \
	if [ -z "$$src" ]; then \
	  echo "No CUDA source found in $$dir"; exit 1; \
	fi; \
	echo "Building $@ from $$src"; \
	NVCC_CCBIN=$(NVCC_CCBIN) $(NVCC) $(NVCC_CUBIN_FLAGS) "$$src" -o "$@"

# PTX from the same source (depends on the cubin rule to locate the source once)
build/%.ptx: | $(BUILD_DIR)
	@dir="$*"; \
	src=""; \
	for candidate in "$$dir"/*.cu; do \
	  if [ -f "$$candidate" ]; then src="$$candidate"; break; fi; \
	done; \
	if [ -z "$$src" ]; then \
	  echo "No CUDA source found in $$dir"; exit 1; \
	fi; \
	echo "Emitting $@ from $$src"; \
	NVCC_CCBIN=$(NVCC_CCBIN) $(NVCC) $(NVCC_PTX_FLAGS) "$$src" -o "$@"

# SASS disassembly from the .cubin
build/%.sass: build/%.cubin | $(BUILD_DIR)
	@echo "Disassembling $< -> $@"
	@$(NVDISASM) $(NVDISASM_FLAGS) "$<" > "$@"

clean:
	@rm -rf $(BUILD_DIR)
