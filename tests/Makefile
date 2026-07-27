.PHONY: all test clean

BUILD_DIR ?= ../build/tests
TARGETS := $(BUILD_DIR)/vec_scalar_test $(BUILD_DIR)/vec_unary_test $(BUILD_DIR)/vec_convert_test $(BUILD_DIR)/vec_mul24_test $(BUILD_DIR)/vec_block_retirement_test
CXXFLAGS ?= -std=c++20 -Wall -Wextra

all: test

test: $(TARGETS)
	@for target in $(TARGETS); do $$target || exit $$?; done

$(BUILD_DIR)/vec_scalar_test: vec_scalar_test.cpp ../include/vec_scalar.hpp ../include/vec.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< -o $@

$(BUILD_DIR)/vec_block_retirement_test: vec_block_retirement_test.cpp ../include/vec.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< -o $@

$(BUILD_DIR)/vec_unary_test: vec_unary_test.cpp ../include/vec_unary.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< -o $@

$(BUILD_DIR)/vec_convert_test: vec_convert_test.cpp ../include/vec_convert.hpp ../include/vec_unary.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< -o $@

$(BUILD_DIR)/vec_mul24_test: vec_mul24_test.cpp ../include/vec_mul24.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< -o $@

clean:
	rm -f $(TARGETS)
