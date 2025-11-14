# utils 目录 Makefile
.PHONY: all clean

# 源文件配置
SRC_DIR := .
SRC_FILES := $(wildcard $(SRC_DIR)/*.c) $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.c))
OBJ_FILES += $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(wildcard $(SRC_DIR)/*.cpp))

# 包含目录
INCLUDES := -I./include -I../include

# 默认目标
all: $(OBJ_FILES)
	@echo "utils module built"

# C 源文件编译规则
$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@
	@$(CC) $(CFLAGS) $(CPPFLAGS) $(INCLUDES) -MM -MT $@ $< > $(BUILD_DIR)/$*.d

# C++ 源文件编译规则
$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -c $< -o $@
	@$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(INCLUDES) -MM -MT $@ $< > $(BUILD_DIR)/$*.d

# 清理
clean:
	rm -f $(OBJ_FILES) $(OBJ_FILES:.o=.d)
	@echo "utils module cleaned"

# 包含依赖文件
-include $(OBJ_FILES:.o=.d)
