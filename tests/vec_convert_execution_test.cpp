/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-emu is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include "vec.hpp"
#include "vec_convert.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr uint32_t CUSTOM_32_OPCODE = 0x0b;
constexpr uint64_t INPUT_ADDR = 0x1000;
constexpr uint64_t OUTPUT_ADDR = 0x2000;

constexpr uint32_t regular_load_raw(uint8_t subfunc = 0)
{
    return (0b010000u << 26)
        | (static_cast<uint32_t>(subfunc & 7) << 12)
        | CUSTOM_32_OPCODE;
}

constexpr uint32_t regular_store_raw()
{
    return (0b010001u << 26) | CUSTOM_32_OPCODE;
}

constexpr uint32_t masked_store_raw(uint8_t scalar)
{
    return (0b010001u << 26) | (1u << 22)
        | (static_cast<uint32_t>(scalar & 7) << 12)
        | CUSTOM_32_OPCODE;
}

constexpr uint32_t shuffle_raw(uint8_t subfunc)
{
    return (0b011111u << 26)
        | (static_cast<uint32_t>(subfunc & 7) << 12)
        | CUSTOM_32_OPCODE;
}

constexpr uint32_t scalar_true_raw()
{
    return (0b111110u << 26) | (1u << 17) | (3u << 15)
        | (3u << 10) | CUSTOM_32_OPCODE;
}

constexpr uint32_t scalar_add_raw()
{
    return (0b111110u << 26) | (1u << 15) | (1u << 12)
        | CUSTOM_32_OPCODE;
}

constexpr uint32_t convert_raw(uint8_t op, uint8_t rm)
{
    return (0b100000u << 26) | (0b00010u << 20) | (0b01u << 18)
        | (static_cast<uint32_t>(op) << 15)
        | (static_cast<uint32_t>(rm) << 12) | CUSTOM_32_OPCODE;
}

bool check(bool condition, const char* message)
{
    if(!condition) {
        std::cerr << "vec_convert_execution_test failed: " << message << '\n';
    }
    return condition;
}

bool run_fp32_lowp_compress(uint8_t op)
{
    using namespace DAYAN;

    std::vector<uint8_t> uram(1024 * 1024);
    std::array<uint32_t, 64> input{};
    for(size_t lane = 0; lane < input.size(); lane++) {
        const float value = static_cast<float>(static_cast<int>(lane) - 31) / 13.0f;
        input[lane] = std::bit_cast<uint32_t>(value);
    }
    std::memcpy(uram.data() + INPUT_ADDR, input.data(), sizeof(input));
    std::array<uint32_t, 64> initial_output;
    initial_output.fill(0xdeadbeefu);
    std::memcpy(uram.data() + OUTPUT_ADDR, initial_output.data(), sizeof(initial_output));

    std::array<vec_inst_t, VEC_INST_QUEUE_CAPACITY> inst_queue{};
    size_t pc = 0;
    inst_queue[pc++] = vec_inst_t{scalar_true_raw()};
    inst_queue[pc++] = vec_inst_t{scalar_add_raw()};
    inst_queue[pc++] = vec_inst_t{regular_load_raw(), INPUT_ADDR};
    inst_queue[pc++] = vec_inst_t{convert_raw(op, 0)};
    inst_queue[pc++] = vec_inst_t{shuffle_raw(0b110), 0x00};
    inst_queue[pc++] = vec_inst_t{masked_store_raw(0), OUTPUT_ADDR};

    MPMCQueue<vec_vfork_req_t, 16> vfork_queue;
    MPMCQueue<vec_resp_t, 16> resp_queue;
    vec_sparse_memory_t sparse_memory;
    vec_executer_t executer(
        &vfork_queue, &resp_queue, inst_queue.data(), uram.data(), &sparse_memory, 64, 1);
    executer.start();

    vec_vfork_req_t req;
    req.block_id = 0;
    req.inst_count = pc;
    req.control.epilogue0_end = pc;
    vfork_queue.enqueue(req);
    const vec_resp_t resp = resp_queue.dequeue();
    executer.stop();

    if(!check(resp.block_id == 0 && resp.inst_count == pc,
              "narrow vector block retirement")) {
        return false;
    }

    std::array<uint32_t, 64> output{};
    std::memcpy(output.data(), uram.data() + OUTPUT_ADDR, sizeof(output));
    for(size_t word = 0; word < output.size() / 2; word++) {
        const uint32_t low = execute_vec_convert_lane(input[2 * word], op, 0);
        const uint32_t high = execute_vec_convert_lane(input[2 * word + 1], op, 0);
        if(!check(output[word] == (low | (high << 16)),
                  "narrow conversion and compression layout")) {
            return false;
        }
    }
    for(size_t word = output.size() / 2; word < output.size(); word++) {
        if(!check(output[word] == initial_output[word],
                  "narrow masked store preserves undefined lanes")) {
            return false;
        }
    }
    return true;
}

bool run_lowp_decompress_fp32(uint8_t op)
{
    using namespace DAYAN;

    std::vector<uint8_t> uram(1024 * 1024);
    std::array<uint32_t, 64> input{};
    for(size_t word = 0; word < input.size() / 2; ++word) {
        const uint16_t low = static_cast<uint16_t>(0x3c00 + word);
        const uint16_t high = static_cast<uint16_t>(0x4000 + word);
        input[word] = low | (static_cast<uint32_t>(high) << 16);
    }
    std::memcpy(uram.data() + INPUT_ADDR, input.data(), sizeof(input));

    std::array<vec_inst_t, VEC_INST_QUEUE_CAPACITY> inst_queue{};
    size_t pc = 0;
    inst_queue[pc++] = vec_inst_t{regular_load_raw(), INPUT_ADDR};
    inst_queue[pc++] = vec_inst_t{shuffle_raw(0b110), 0x20};
    inst_queue[pc++] = vec_inst_t{convert_raw(op, 0)};
    inst_queue[pc++] = vec_inst_t{regular_store_raw(), OUTPUT_ADDR};

    MPMCQueue<vec_vfork_req_t, 16> vfork_queue;
    MPMCQueue<vec_resp_t, 16> resp_queue;
    vec_sparse_memory_t sparse_memory;
    vec_executer_t executer(
        &vfork_queue, &resp_queue, inst_queue.data(), uram.data(), &sparse_memory, 64, 1);
    executer.start();

    vec_vfork_req_t req;
    req.block_id = 0;
    req.inst_count = pc;
    req.control.epilogue0_end = pc;
    vfork_queue.enqueue(req);
    const vec_resp_t resp = resp_queue.dequeue();
    executer.stop();

    if(!check(resp.block_id == 0 && resp.inst_count == pc,
              "widen vector block retirement")) {
        return false;
    }

    std::array<uint32_t, 64> output{};
    std::memcpy(output.data(), uram.data() + OUTPUT_ADDR, sizeof(output));
    for(size_t word = 0; word < input.size() / 2; ++word) {
        if(!check(output[2 * word]
                         == execute_vec_convert_lane(input[word], op, 0),
                  "decompress and widening low element")
           || !check(output[2 * word + 1]
                         == execute_vec_convert_lane(input[word] >> 16, op, 0),
                     "decompress and widening high element")) {
            return false;
        }
    }
    return true;
}

bool run_bf16_mxfp8()
{
    using namespace DAYAN;

    std::vector<uint8_t> uram(1024 * 1024);
    std::array<uint32_t, 64> input{};
    constexpr std::array<uint16_t, 4> values = {
        0x3f80, 0x4000, 0x4080, 0x4100
    };
    constexpr std::array<uint32_t, 4> scales = {
        0x77, 0x78, 0x79, 0x7a
    };
    for(size_t subcore = 0; subcore < values.size(); ++subcore) {
        const uint32_t packed = values[subcore]
            | (static_cast<uint32_t>(values[subcore]) << 16);
        for(size_t word = 0; word < 16; ++word) {
            input[subcore * 16 + word] = packed;
        }
    }
    std::memcpy(uram.data() + INPUT_ADDR, input.data(), sizeof(input));

    std::array<vec_inst_t, VEC_INST_QUEUE_CAPACITY> inst_queue{};
    size_t pc = 0;
    inst_queue[pc++] = vec_inst_t{regular_load_raw(), INPUT_ADDR};
    inst_queue[pc++] = vec_inst_t{convert_raw(6, 0)};
    inst_queue[pc++] = vec_inst_t{regular_store_raw(), OUTPUT_ADDR};

    MPMCQueue<vec_vfork_req_t, 16> vfork_queue;
    MPMCQueue<vec_resp_t, 16> resp_queue;
    vec_sparse_memory_t sparse_memory;
    vec_executer_t executer(
        &vfork_queue, &resp_queue, inst_queue.data(), uram.data(), &sparse_memory, 64, 1);
    executer.start();

    vec_vfork_req_t req;
    req.block_id = 0;
    req.inst_count = pc;
    req.control.epilogue0_end = pc;
    vfork_queue.enqueue(req);
    const vec_resp_t resp = resp_queue.dequeue();
    executer.stop();

    if(!check(resp.block_id == 0 && resp.inst_count == pc,
              "vector block retirement")) {
        return false;
    }

    std::array<uint32_t, 64> output{};
    std::memcpy(output.data(), uram.data() + OUTPUT_ADDR, sizeof(output));
    for(size_t subcore = 0; subcore < values.size(); ++subcore) {
        const size_t base = subcore * 16;
        for(size_t word = 0; word < 8; ++word) {
            if(!check(output[base + word] == 0x78787878,
                      "Sub-Core payload placement")) {
                return false;
            }
        }
        if(!check(output[base + 8] == scales[subcore],
                  "Sub-Core scale placement")) {
            return false;
        }
    }
    return true;
}

bool run_vfork_context()
{
    using namespace DAYAN;

    constexpr uint64_t blocked_input_addr = 0x1000;
    constexpr uint64_t allowed_output_addr = 0x9000;
    std::vector<uint8_t> uram(1024 * 1024);
    std::array<uint32_t, 64> initial_output;
    initial_output.fill(0xdeadbeefu);
    std::memcpy(uram.data() + allowed_output_addr, initial_output.data(),
                sizeof(initial_output));

    std::array<vec_inst_t, VEC_INST_QUEUE_CAPACITY> inst_queue{};
    size_t pc = 0;
    inst_queue[pc++] = vec_inst_t{regular_load_raw(), blocked_input_addr};
    inst_queue[pc++] = vec_inst_t{regular_store_raw(), allowed_output_addr};

    MPMCQueue<vec_vfork_req_t, 16> vfork_queue;
    MPMCQueue<vec_resp_t, 16> resp_queue;
    vec_sparse_memory_t sparse_memory;
    interrupt_controller_t interrupts;
    vec_executer_t executer(&vfork_queue, &resp_queue, inst_queue.data(),
                            uram.data(), &sparse_memory, 64, 1, 4,
                            &interrupts);
    executer.start();

    vec_vfork_req_t req;
    req.block_id = 0;
    req.inst_count = pc;
    req.control.epilogue0_end = pc;
    req.uram_range = uram_range_t{1, 2};
    vfork_queue.enqueue(req);
    const vec_resp_t resp = resp_queue.dequeue();
    executer.stop();

    if(!check(resp.block_id == 0 && resp.inst_count == pc,
              "vfork-context block retirement") ||
       !check(interrupts.count() == 1,
              "vfork Context controls FlexVec URAM range") ||
       !check((interrupts.read(1) >> 32) == static_cast<uint16_t>(
                  interrupt_source_t::flexvec_uram_group_0_range),
              "vfork Context raises the expected FlexVec interrupt")) {
        return false;
    }

    std::array<uint32_t, 64> output{};
    std::memcpy(output.data(), uram.data() + allowed_output_addr,
                sizeof(output));
    for(uint32_t value : output) {
        if(!check(value == 0,
                  "failed load stores zeros inside the vfork range")) {
            return false;
        }
    }
    return true;
}

bool run_flexvec_32_byte_alignment()
{
    using namespace DAYAN;

    constexpr uint64_t uram_window_base = 0x00800000;
    constexpr uint64_t input_offset = 32;
    constexpr uint64_t output_offset = 0x200;
    constexpr uint64_t input_addr = uram_window_base + input_offset;
    constexpr uint64_t output_addr = uram_window_base + output_offset;
    std::vector<uint8_t> uram(1024 * 1024);
    std::array<uint32_t, 64> input{};
    for(size_t lane = 0; lane < input.size(); ++lane) {
        input[lane] = 0x10000u + lane;
    }
    std::memcpy(uram.data() + input_offset, input.data(), sizeof(input));

    std::array<vec_inst_t, VEC_INST_QUEUE_CAPACITY> inst_queue{};
    size_t pc = 0;
    inst_queue[pc++] = vec_inst_t{regular_load_raw(), input_addr};
    inst_queue[pc++] = vec_inst_t{regular_store_raw(), output_addr};

    MPMCQueue<vec_vfork_req_t, 16> vfork_queue;
    MPMCQueue<vec_resp_t, 16> resp_queue;
    vec_sparse_memory_t sparse_memory;
    interrupt_controller_t interrupts;
    vec_executer_t executer(&vfork_queue, &resp_queue, inst_queue.data(),
                            uram.data(), &sparse_memory, 64, 1, 4,
                            &interrupts);
    executer.start();

    vec_vfork_req_t req;
    req.block_id = 0;
    req.inst_count = pc;
    req.control.epilogue0_end = pc;
    req.uram_range = uram_range_t{0, 48};
    vfork_queue.enqueue(req);
    const vec_resp_t resp = resp_queue.dequeue();
    executer.stop();

    std::array<uint32_t, 64> output{};
    std::memcpy(output.data(), uram.data() + output_offset, sizeof(output));
    return check(resp.block_id == 0 && resp.inst_count == pc,
                 "window-addressed FlexVec block retirement")
        && check(interrupts.count() == 0,
                 "window-addressed 32-byte-aligned access does not interrupt")
        && check(output == input,
                 "window-addressed FlexVec load/store preserves data");
}

}

int main()
{
    if(!run_fp32_lowp_compress(4) || !run_fp32_lowp_compress(5)
       || !run_lowp_decompress_fp32(2) || !run_lowp_decompress_fp32(3)
       || !run_bf16_mxfp8() || !run_vfork_context()
       || !run_flexvec_32_byte_alignment()) {
        return 1;
    }
    std::cout << "vec_convert_execution_test passed\n";
    return 0;
}
