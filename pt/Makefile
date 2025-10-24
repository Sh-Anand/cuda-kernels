SHELL := /bin/bash

CMAKE ?= cmake
NINJA ?= ninja
GENERATOR ?= Ninja

BUILD_ROOT := pathtracer/build
OUTPUT_DIR := build

.PHONY: all configure cmake_build ninja move clean

all: move

configure:
	@$(CMAKE) -S pathtracer -B $(BUILD_ROOT) -G $(GENERATOR)

cmake_build: configure
	@$(CMAKE) --build $(BUILD_ROOT)

ninja: cmake_build
	@$(NINJA) -C $(BUILD_ROOT)

$(OUTPUT_DIR):
	@mkdir -p $@

move: ninja | $(OUTPUT_DIR)
	@$(CMAKE) -E copy $(BUILD_ROOT)/pathtracer.ptx $(OUTPUT_DIR)/pathtracer.ptx
	@$(CMAKE) -E copy $(BUILD_ROOT)/pathtracer.sass $(OUTPUT_DIR)/pathtracer.sass

clean:
	@$(CMAKE) -E rm -rf $(BUILD_ROOT) $(OUTPUT_DIR)
