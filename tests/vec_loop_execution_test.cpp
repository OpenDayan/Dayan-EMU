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
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr uint64_t output_base = 0x1000;
constexpr uint64_t phase_stride = 2048;

constexpr uint32_t iterator_raw(bool vector, uint8_t selector, uint8_t sd)
{
    return (0b111110u << 26) | (uint32_t(sd) << 23)
        | (uint32_t(vector ? 3 : 2) << 15) | (uint32_t(selector) << 10) | 0x0b;
}

constexpr uint32_t scalar_store_raw(uint8_t source)
{
    return (0b010001u << 26) | (3u << 20) | (uint32_t(source) << 15)
        | (2u << 12) | 0x0b;
}

void write_u64(std::vector<uint8_t>& memory, uint64_t address, uint64_t value)
{
    std::memcpy(memory.data() + address, &value, sizeof(value));
}

bool check_loop(uint64_t dim0, uint64_t dim1, unsigned phases, uint8_t selector)
{
    using namespace DAYAN;
    std::vector<uint8_t> uram(65536, 0xa5);
    auto expected = uram;
    vec_tcsr_t tcsr{};
    tcsr.dim0 = dim0;
    tcsr.dim1 = dim1;
    tcsr.dim2 = 1;
    tcsr.dim3 = 1;
    for(unsigned i = 0; i < 4; ++i) {
        tcsr.s[i].base = 0x10000 + 0x100 * i;
        tcsr.s[i].dim0Stride = 7 + i;
        tcsr.s[i].dim1Stride = 101 + i;
        tcsr.s[i].dim2Stride = 1009;
        tcsr.s[i].dim3Stride = 10007;
        tcsr.v[i].base = 0x20000 + 0x100 * i;
        tcsr.v[i].dim0Stride = 13 + i;
        tcsr.v[i].dim1Stride = 211 + i;
        tcsr.v[i].dim2Stride = 2003;
        tcsr.v[i].dim3Stride = 20011;
    }
    tcsr.v[3].base = output_base;
    tcsr.v[3].dim0Stride = 128;
    tcsr.v[3].dim1Stride = 512;
    tcsr.v[3].dim2Stride = 16384;
    tcsr.v[3].dim3Stride = 32768;

    std::array<vec_inst_t, VEC_INST_QUEUE_CAPACITY> instructions{};
    std::array<uint8_t, 5> boundaries{};
    size_t count = 0;
    for(unsigned phase = 0; phase < 5; ++phase) {
        if(phases & (1u << phase)) {
            const uint64_t address = phase * phase_stride;
            instructions[count++] = {iterator_raw(false, selector, 2), 0};
            instructions[count++] = {scalar_store_raw(2), address};
            instructions[count++] = {iterator_raw(true, selector, 3), 0};
            instructions[count++] = {scalar_store_raw(3), address + 32};
            // Increment a scalar counter to record execution order at each coordinate.
            instructions[count++] = {(0b010000u << 26) | (1u << 23) | (1u << 17) | 0x0b, 1};
            instructions[count++] = {(0b111110u << 26) | (1u << 15) | (1u << 12) | 0x0b, 0};
            instructions[count++] = {scalar_store_raw(0), address + 64};
        }
        boundaries[phase] = count;
    }

    for(uint64_t d3 = 0; d3 < 2; ++d3) {
        for(uint64_t d2 = 0; d2 < 2; ++d2) {
            uint64_t sequence = 0;
            auto record = [&](unsigned phase, uint64_t d0, uint64_t d1) {
                if(!(phases & (1u << phase))) {
                    return;
                }
                const uint64_t address = output_base + phase * phase_stride
                    + 128 * d0 + 512 * d1 + 16384 * d2 + 32768 * d3;
                const auto& s = tcsr.s[selector];
                const auto& v = tcsr.v[selector];
                write_u64(expected, address, s.base + d0 * s.dim0Stride + d1 * s.dim1Stride
                    + d2 * s.dim2Stride + d3 * s.dim3Stride);
                write_u64(expected, address + 32, v.base + d0 * v.dim0Stride + d1 * v.dim1Stride
                    + d2 * v.dim2Stride + d3 * v.dim3Stride);
                write_u64(expected, address + 64, ++sequence);
            };
            record(0, 0, 0);
            for(uint64_t row = 0; row <= dim1; ++row) {
                record(1, 0, row);
                for(uint64_t column = 0; column <= dim0; ++column) {
                    record(2, column, row);
                }
                record(3, dim0 + 1, row);
            }
            record(4, dim0 + 1, dim1 + 1);
        }
    }

    MPMCQueue<vec_vfork_req_t, 16> requests;
    MPMCQueue<vec_resp_t, 16> responses;
    vec_sparse_memory_t sparse_memory;
    vec_executer_t executer(&requests, &responses, instructions.data(), uram.data(),
                           &sparse_memory, 64, 1);
    executer.start();
    vec_vfork_req_t request{};
    request.inst_count = count;
    request.control = {boundaries[0], boundaries[1], boundaries[2], boundaries[3], boundaries[4], false};
    request.tcsr_snapshot = tcsr;
    requests.enqueue(request);
    const auto response = responses.dequeue();
    executer.stop();
    if(response.block_id != 0 || response.inst_count != count || uram != expected) {
        std::cerr << "vec_loop_execution_test failed: Dim0=" << dim0 << " Dim1=" << dim1
                  << " phases=" << phases << " selector=" << unsigned(selector) << '\n';
        return false;
    }
    return true;
}

}

int main()
{
    for(const auto& dimensions : {std::array<uint64_t, 2>{0, 0}, {2, 0}, {0, 1}, {2, 1}}) {
        for(unsigned phases = 1; phases < 32; ++phases) {
            for(uint8_t selector = 0; selector < 4; ++selector) {
                if(!check_loop(dimensions[0], dimensions[1], phases, selector)) {
                    return 1;
                }
            }
        }
    }
    std::cout << "vec_loop_execution_test passed (496 cases)\n";
    return 0;
}
