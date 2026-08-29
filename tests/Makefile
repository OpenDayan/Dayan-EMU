.PHONY: all test clean

BUILD_DIR ?= ../build/tests
TARGETS := $(BUILD_DIR)/tensor_context_interrupt_test $(BUILD_DIR)/vec_scalar_test $(BUILD_DIR)/vec_unary_test $(BUILD_DIR)/vec_sfu_execution_test $(BUILD_DIR)/vec_convert_test $(BUILD_DIR)/vec_convert_execution_test $(BUILD_DIR)/vec_mul24_test $(BUILD_DIR)/vec_minmax_execution_test $(BUILD_DIR)/vec_channel_stride_test $(BUILD_DIR)/vec_shuffle_test $(BUILD_DIR)/vec_shuffle_execution_test $(BUILD_DIR)/vec_block_retirement_test $(BUILD_DIR)/mmu_test
CXXFLAGS ?= -std=c++20 -Wall -Wextra

all: test

test: $(TARGETS)
	@for target in $(TARGETS); do $$target || exit $$?; done

$(BUILD_DIR)/tensor_context_interrupt_test: tensor_context_interrupt_test.cpp ../utils/interrupt.cpp ../include/tensor_context.hpp ../include/interrupt.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< ../utils/interrupt.cpp $(LDFLAGS) -o $@

$(BUILD_DIR)/mmu_test: mmu_test.cpp ../emu/mmu.cpp ../include/mmu.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< ../emu/mmu.cpp $(LDFLAGS) -o $@

$(BUILD_DIR)/vec_scalar_test: vec_scalar_test.cpp ../include/vec_scalar.hpp ../include/vec.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< -o $@

$(BUILD_DIR)/vec_block_retirement_test: vec_block_retirement_test.cpp ../include/vec.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< -o $@

$(BUILD_DIR)/vec_unary_test: vec_unary_test.cpp ../include/vec_unary.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< -o $@

$(BUILD_DIR)/vec_sfu_execution_test: vec_sfu_execution_test.cpp ../emu/vec.cpp ../utils/interrupt.cpp ../include/vec.hpp ../include/vec_unary.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< ../emu/vec.cpp ../utils/interrupt.cpp $(LDFLAGS) -o $@

$(BUILD_DIR)/vec_convert_test: vec_convert_test.cpp ../include/vec_convert.hpp ../include/vec_shuffle.hpp ../include/vec_unary.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< -o $@

$(BUILD_DIR)/vec_convert_execution_test: vec_convert_execution_test.cpp ../emu/vec.cpp ../utils/interrupt.cpp ../include/vec.hpp ../include/vec_convert.hpp ../include/vec_unary.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< ../emu/vec.cpp ../utils/interrupt.cpp $(LDFLAGS) -o $@

$(BUILD_DIR)/vec_mul24_test: vec_mul24_test.cpp ../include/vec_mul24.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< -o $@

$(BUILD_DIR)/vec_minmax_execution_test: vec_minmax_execution_test.cpp ../emu/vec.cpp ../utils/interrupt.cpp ../include/vec.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< ../emu/vec.cpp ../utils/interrupt.cpp $(LDFLAGS) -o $@

$(BUILD_DIR)/vec_channel_stride_test: vec_channel_stride_test.cpp ../emu/vec.cpp ../utils/interrupt.cpp ../include/vec.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< ../emu/vec.cpp ../utils/interrupt.cpp $(LDFLAGS) -o $@

$(BUILD_DIR)/vec_shuffle_test: vec_shuffle_test.cpp ../include/vec_shuffle.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< -o $@

$(BUILD_DIR)/vec_shuffle_execution_test: vec_shuffle_execution_test.cpp ../emu/vec.cpp ../utils/interrupt.cpp ../include/vec.hpp ../include/vec_shuffle.hpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I../include $< ../emu/vec.cpp ../utils/interrupt.cpp $(LDFLAGS) -o $@

clean:
	rm -f $(TARGETS)
