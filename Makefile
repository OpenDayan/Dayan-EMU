# 根目录 Makefile
.PHONY: all clean distclean $(SUB_DIRS)

# 项目配置
PROJECT_NAME := rvt_emulator
SUB_DIRS := emu rvt64 main utils
BUILD_DIR := build
OUTPUT_DIR := bin

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

# 导出变量到子 Makefile
export CC CXX AR CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CROSS_COMPILE BUILD_TYPE

# 目标文件列表（将在构建过程中填充）
OBJ_FILES :=

# 默认目标
all: $(SUB_DIRS) $(BUILD_DIR) $(OUTPUT_DIR)
	@echo "Linking final executable..."
	$(CXX) $(BUILD_DIR)/*.o $(LDFLAGS) -o $(OUTPUT_DIR)/$(PROJECT_NAME)
	@echo "Build completed: $(OUTPUT_DIR)/$(PROJECT_NAME)"

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
	@echo "  clean     - Remove object files and temporary files"
	@echo "  distclean - Remove all generated files"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Build type (BUILD_TYPE):"
	@echo "  make                    - Release build (default, optimized)"
	@echo "  make BUILD_TYPE=Debug   - Debug build (with debug output)"

# 包含依赖文件（如果存在）
-include $(OBJ_FILES:.o=.d)
