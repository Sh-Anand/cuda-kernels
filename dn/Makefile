SHELL := /bin/bash

CMAKE ?= cmake
NINJA ?= ninja
GENERATOR ?= Ninja

BUILD_ROOT := denoiser/build
OUTPUT_DIR := build

.PHONY: all configure cmake_build ninja move clean

all: move

configure:
	@$(CMAKE) -S denoiser -B $(BUILD_ROOT) -G $(GENERATOR)

cmake_build: configure
	@$(CMAKE) --build $(BUILD_ROOT)

ninja: cmake_build
	@$(NINJA) -C $(BUILD_ROOT)

$(OUTPUT_DIR):
	@mkdir -p $@

move: ninja | $(OUTPUT_DIR)
	@$(CMAKE) -E copy $(BUILD_ROOT)/denoiser.cubin $(OUTPUT_DIR)/denoiser.cubin
	@$(CMAKE) -E copy $(BUILD_ROOT)/denoiser.ptx $(OUTPUT_DIR)/denoiser.ptx
	@$(CMAKE) -E copy $(BUILD_ROOT)/denoiser.sass $(OUTPUT_DIR)/denoiser.sass

clean:
	@$(CMAKE) -E rm -rf $(BUILD_ROOT) $(OUTPUT_DIR)
