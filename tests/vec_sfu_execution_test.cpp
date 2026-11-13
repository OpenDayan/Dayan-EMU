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
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr uint32_t CUSTOM_32_OPCODE = 0x0b;
constexpr uint64_t INPUT_ADDR = 0x1000;
constexpr uint64_t OUTPUT_ADDR = 0x2000;

constexpr uint32_t regular_load_raw()
{
    return (0b010000u << 26) | CUSTOM_32_OPCODE;
}

constexpr uint32_t regular_store_raw()
{
    return (0b010001u << 26) | CUSTOM_32_OPCODE;
}

constexpr uint32_t sfu_raw(uint8_t precision, uint8_t op)
{
    return (0b100000u << 26) | (0b00010u << 20)
        | (static_cast<uint32_t>(precision & 1) << 17)
        | (static_cast<uint32_t>(op & 31) << 12)
        | CUSTOM_32_OPCODE;
}

bool check(bool condition, const char* message)
{
    if(!condition) {
        std::cerr << "vec_sfu_execution_test failed: " << message << '\n';
    }
    return condition;
}

bool run_fast_sfu(uint8_t op)
{
    using namespace DAYAN;

    std::vector<uint8_t> uram(1024 * 1024);
    std::array<uint32_t, 64> input{};
    for(size_t lane = 0; lane < input.size(); lane++) {
        const float value = op == 0
            ? static_cast<float>(static_cast<int>(lane) - 32) / 16.0f
            : static_cast<float>(lane + 1) / 8.0f;
        input[lane] = std::bit_cast<uint32_t>(value);
    }
    std::memcpy(uram.data() + INPUT_ADDR, input.data(), sizeof(input));

    std::array<vec_inst_t, VEC_INST_QUEUE_CAPACITY> inst_queue{};
    size_t pc = 0;
    inst_queue[pc++] = vec_inst_t{regular_load_raw(), INPUT_ADDR};
    inst_queue[pc++] = vec_inst_t{sfu_raw(1, op)};
    inst_queue[pc++] = vec_inst_t{regular_store_raw(), OUTPUT_ADDR};

    MPMCQueue<vec_vfork_req_t, 16> vfork_queue;
    MPMCQueue<vec_resp_t, 16> resp_queue;
    vec_sparse_memory_t sparse_memory;
    vec_executer_t executer(
        &vfork_queue, &resp_queue, inst_queue.data(), uram.data(),
        &sparse_memory, 64, 1);
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
    for(size_t lane = 0; lane < output.size(); lane++) {
        const float x = std::bit_cast<float>(input[lane]);
        const float expected = op == 0 ? std::exp2(x) : 1.0f / x;
        if(!check(output[lane] == std::bit_cast<uint32_t>(expected),
                  op == 0 ? "fast exp2 execution" : "fast reciprocal execution")) {
            return false;
        }
    }
    return true;
}

}

int main()
{
    if(!run_fast_sfu(0) || !run_fast_sfu(1)) {
        return 1;
    }

    std::cout << "vec_sfu_execution_test passed\n";
    return 0;
}
