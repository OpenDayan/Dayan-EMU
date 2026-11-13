#***************************************************************************************
# Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
#
# Dayan-emu is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
#***************************************************************************************

# 根目录 Makefile

# 项目配置
PROJECT_NAME := dayan-emu
SUB_DIRS := emu rvt64 main utils
BUILD_DIR := build
OUTPUT_DIR := bin
LSU_DMA_TEST := $(BUILD_DIR)/lsu_dma_test
MAT_TEST := $(BUILD_DIR)/mat_test
QEMU_DIR ?= deps/QEMU
QEMU_BIN ?= $(QEMU_DIR)/build/dl-riscv64
QEMU_JOBS ?= 16
QEMU_ENV := env -u CC -u CXX -u AR -u CFLAGS -u CXXFLAGS \
	-u CPPFLAGS -u LDFLAGS -u CROSS_COMPILE -u BUILD_TYPE
# Shared ClusterMMU behavioral core, owned by the QEMU tree and also
# linked into this emulator.
MMU_CORE_SRC := $(QEMU_DIR)/hw/riscv/dayan_mmu_core.c
MMU_CORE_HDR := $(QEMU_DIR)/include/hw/riscv/dayan_mmu_core.h
MMU_CORE_OBJ := $(BUILD_DIR)/dayan_mmu_core.o

.PHONY: all clean distclean help test deps qemu force-qemu-build $(SUB_DIRS)

# 构建类型: Release（默认） / Debug
BUILD_TYPE ?= Release

# 工具链配置
CROSS_COMPILE ?=
CC := $(CROSS_COMPILE)gcc
CXX := $(CROSS_COMPILE)g++
AR := $(CROSS_COMPILE)ar

# 公共编译选项
COMMON_CFLAGS := -std=c11 -Wall -Wextra -ffast-math -march=native
COMMON_CXXFLAGS := -std=c++20 -Wall -Wextra -ffast-math -march=native

# 编译选项（按构建类型区分）
ifeq ($(BUILD_TYPE),Debug)
CFLAGS ?= $(COMMON_CFLAGS) -O0 -g -fsanitize=address
CXXFLAGS ?= $(COMMON_CXXFLAGS) -O0 -g -fsanitize=address
LDFLAGS ?= -lm -lpthread -fsanitize=address
else ifeq ($(BUILD_TYPE),Release)
CFLAGS ?= $(COMMON_CFLAGS) -O2 -DNDEBUG
CXXFLAGS ?= $(COMMON_CXXFLAGS) -O2 -DNDEBUG
LDFLAGS ?= -lm -lpthread -flto
else
$(error Unknown BUILD_TYPE: $(BUILD_TYPE). Use Debug or Release)
endif

CPPFLAGS ?= -I./include -I./emu -I./main -I./utils
CPPFLAGS += -I$(abspath $(QEMU_DIR))/include
CPPFLAGS += -DDAYAN_DEFAULT_QEMU=\"$(abspath $(QEMU_BIN))\"

# 导出变量到子 Makefile
export CC CXX AR CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CROSS_COMPILE BUILD_TYPE

# 目标文件列表（将在构建过程中填充）
OBJ_FILES :=

# 默认目标
all: qemu $(SUB_DIRS) $(MMU_CORE_OBJ) $(BUILD_DIR) $(OUTPUT_DIR)
	@echo "Linking final executable..."
	$(CXX) $(BUILD_DIR)/*.o $(LDFLAGS) -o $(OUTPUT_DIR)/$(PROJECT_NAME)
	@echo "Build completed: $(OUTPUT_DIR)/$(PROJECT_NAME)"

$(MMU_CORE_OBJ): $(MMU_CORE_SRC) $(MMU_CORE_HDR)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

test: $(BUILD_DIR) $(MMU_CORE_OBJ)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I./include tests/lsu_dma_test.cpp emu/lsu.cpp emu/mmu.cpp utils/interrupt.cpp $(MMU_CORE_OBJ) $(LDFLAGS) -o $(LSU_DMA_TEST)
	$(LSU_DMA_TEST)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I./include tests/mat_test.cpp emu/mat.cpp utils/interrupt.cpp $(LDFLAGS) -o $(MAT_TEST)
	$(MAT_TEST)
	$(MAKE) -C tests BUILD_DIR=../$(BUILD_DIR)/tests QEMU_DIR="$(abspath $(QEMU_DIR))" test

deps:
	git submodule update --init --recursive

qemu: $(QEMU_BIN)

ifeq ($(abspath $(QEMU_BIN)),$(abspath $(QEMU_DIR)/build/dl-riscv64))
$(QEMU_BIN): force-qemu-build
endif

force-qemu-build:

$(QEMU_BIN):
	@if [ ! -f "$(QEMU_DIR)/configure" ]; then \
		echo "Missing $(QEMU_DIR). Run 'make deps' first."; \
		exit 1; \
	fi
	mkdir -p $(QEMU_DIR)/build
	@if [ ! -f "$(QEMU_DIR)/build/config-host.mak" ]; then \
		cd $(QEMU_DIR)/build && $(QEMU_ENV) \
			../configure --target-list=riscv64-softmmu; \
	fi
	$(QEMU_ENV) $(MAKE) -C $(QEMU_DIR)/build -j$(QEMU_JOBS)

# 创建构建目录
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(OUTPUT_DIR):
	@mkdir -p $(OUTPUT_DIR)

# 递归构建子目录
$(SUB_DIRS): $(BUILD_DIR)
	@echo "Building $@..."
	$(MAKE) -C $@ BUILD_DIR=../$(BUILD_DIR)

# 清理目标
clean:
	@for dir in $(SUB_DIRS); do \
		echo "Cleaning $$dir..."; \
		$(MAKE) -C $$dir clean; \
	done
	@rm -rf $(BUILD_DIR)
	@rm -rf $(OUTPUT_DIR)
	@echo "Clean completed"

distclean: clean
	@rm -rf $(OUTPUT_DIR)
	@echo "Distclean completed"

# 帮助信息
help:
	@echo "Available targets:"
	@echo "  all       - Build the project (default)"
	@echo "  test      - Build and run unit tests"
	@echo "  deps      - Initialize git submodules"
	@echo "  qemu      - Configure and build deps/QEMU"
	@echo "  clean     - Remove object files and temporary files"
	@echo "  distclean - Remove all generated files"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Build type (BUILD_TYPE):"
	@echo "  make                    - Release build (default, optimized)"
	@echo "  make BUILD_TYPE=Debug   - Debug build (with debug output)"

# 包含依赖文件（如果存在）
-include $(OBJ_FILES:.o=.d)
