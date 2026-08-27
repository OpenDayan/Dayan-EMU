#include "vec.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr uint32_t CUSTOM_32_OPCODE = 0x0b;
constexpr uint64_t INPUT_ADDR = 0x1000;
constexpr uint64_t OUTPUT_ADDR = 0x5000;

constexpr uint32_t regular_load_raw(uint8_t vb, uint8_t sel)
{
    return (0b010000u << 26) | (static_cast<uint32_t>(vb & 1) << 25)
        | (static_cast<uint32_t>(sel & 3) << 15) | CUSTOM_32_OPCODE;
}

constexpr uint32_t masked_load_raw(uint8_t vb, uint8_t vrs1, uint8_t s2)
{
    return (0b010000u << 26) | (static_cast<uint32_t>(vb & 1) << 25)
        | (1u << 20) | (static_cast<uint32_t>(vrs1 & 31) << 15)
        | (static_cast<uint32_t>(s2 & 7) << 12) | CUSTOM_32_OPCODE;
}

constexpr uint32_t regular_store_raw(uint8_t sel, uint8_t vrs1)
{
    return (0b010001u << 26) | (static_cast<uint32_t>(sel & 3) << 20)
        | (static_cast<uint32_t>(vrs1 & 31) << 15) | CUSTOM_32_OPCODE;
}

constexpr uint32_t masked_store_raw(uint8_t sel, uint8_t vrs1, uint8_t s2)
{
    return (0b010001u << 26) | (1u << 22)
        | (static_cast<uint32_t>(sel & 3) << 20)
        | (static_cast<uint32_t>(vrs1 & 31) << 15)
        | (static_cast<uint32_t>(s2 & 7) << 12) | CUSTOM_32_OPCODE;
}

constexpr uint32_t g2s_raw(uint8_t sd)
{
    return (0b010000u << 26) | (static_cast<uint32_t>(sd & 7) << 23)
        | (1u << 17) | CUSTOM_32_OPCODE;
}

bool check(bool condition, const char* message)
{
    if(!condition) {
        std::cerr << "vec_channel_stride_test failed: " << message << '\n';
    }
    return condition;
}

void write_word(std::vector<uint8_t>& uram, uint64_t address, uint32_t value)
{
    std::memcpy(uram.data() + address, &value, sizeof(value));
}

uint32_t read_word(const std::vector<uint8_t>& uram, uint64_t address)
{
    uint32_t value = 0;
    std::memcpy(&value, uram.data() + address, sizeof(value));
    return value;
}

void execute(std::vector<uint8_t>& uram,
             const std::vector<RVTNG::vec_inst_t>& instructions,
             const RVTNG::vec_tcsr_t& tcsr)
{
    using namespace RVTNG;

    std::array<vec_inst_t, VEC_INST_QUEUE_CAPACITY> inst_queue{};
    std::copy(instructions.begin(), instructions.end(), inst_queue.begin());
    MPMCQueue<vec_vfork_req_t, 16> vfork_queue;
    MPMCQueue<vec_resp_t, 16> resp_queue;
    vec_sparse_memory_t sparse_memory;
    vec_executer_t executer(
        &vfork_queue, &resp_queue, inst_queue.data(), uram.data(),
        &sparse_memory, 64, 1);
    executer.start();

    vec_vfork_req_t req{};
    req.block_id = 0;
    req.inst_count = instructions.size();
    req.control.epilogue0_end = instructions.size();
    req.tcsr_snapshot = tcsr;
    vfork_queue.enqueue(req);
    const vec_resp_t resp = resp_queue.dequeue();
    executer.stop();
    check(resp.block_id == 0 && resp.inst_count == instructions.size(),
          "vector block retirement");
}

bool check_tcsr_layout_and_defaults()
{
    RVTNG::vec_tcsr_t tcsr{};
    const uint64_t* raw = reinterpret_cast<const uint64_t*>(&tcsr);
    for(size_t selector = 0; selector < 4; ++selector) {
        if(!check(raw[65 + selector] == 32,
                  "channel stride reset value and linear TCSR offset")) {
            return false;
        }
    }
    return true;
}

bool check_strided_load()
{
    constexpr uint64_t stride = 96;
    std::vector<uint8_t> uram(1024 * 1024);
    std::array<uint32_t, 64> expected{};
    for(size_t channel = 0; channel < 8; ++channel) {
        for(size_t lane = 0; lane < 8; ++lane) {
            const size_t vector_lane = channel * 8 + lane;
            expected[vector_lane] = 0x10000000u + vector_lane;
            write_word(uram, INPUT_ADDR + channel * stride + lane * 4,
                       expected[vector_lane]);
        }
    }

    RVTNG::vec_tcsr_t tcsr{};
    tcsr.uramChannelStride[2] = stride;
    execute(uram, {
        {regular_load_raw(0, 2), INPUT_ADDR},
        {regular_store_raw(0, 0), OUTPUT_ADDR}
    }, tcsr);

    std::array<uint32_t, 64> output{};
    std::memcpy(output.data(), uram.data() + OUTPUT_ADDR, sizeof(output));
    return check(output == expected, "u2v selected channel stride and lane layout");
}

bool check_strided_store()
{
    constexpr uint64_t stride = 128;
    constexpr uint32_t sentinel = 0xdeadbeef;
    std::vector<uint8_t> uram(1024 * 1024);
    std::array<uint32_t, 64> input{};
    for(size_t lane = 0; lane < input.size(); ++lane) {
        input[lane] = 0x20000000u + lane;
    }
    std::memcpy(uram.data() + INPUT_ADDR, input.data(), sizeof(input));
    for(uint64_t offset = 0; offset < 8 * stride; offset += 4) {
        write_word(uram, OUTPUT_ADDR + offset, sentinel);
    }

    RVTNG::vec_tcsr_t tcsr{};
    tcsr.uramChannelStride[3] = stride;
    execute(uram, {
        {regular_load_raw(0, 0), INPUT_ADDR},
        {regular_store_raw(3, 0), OUTPUT_ADDR}
    }, tcsr);

    for(size_t channel = 0; channel < 8; ++channel) {
        for(size_t word = 0; word < stride / 4; ++word) {
            const uint32_t expected = word < 8
                ? input[channel * 8 + word] : sentinel;
            if(!check(read_word(uram, OUTPUT_ADDR + channel * stride + word * 4)
                          == expected,
                      "v2u channel blocks and gaps")) {
                return false;
            }
        }
    }
    return true;
}

bool check_masked_accesses_stay_contiguous()
{
    std::vector<uint8_t> uram(1024 * 1024);
    std::array<uint32_t, 64> input{};
    for(size_t lane = 0; lane < input.size(); ++lane) {
        input[lane] = 0x30000000u + lane;
    }
    std::memcpy(uram.data() + INPUT_ADDR, input.data(), sizeof(input));

    RVTNG::vec_tcsr_t tcsr{};
    tcsr.uramChannelStride[3] = 256;
    execute(uram, {
        {g2s_raw(0), ~uint64_t{0}},
        {masked_load_raw(0, 0, 0), INPUT_ADDR},
        {masked_store_raw(3, 0, 0), OUTPUT_ADDR}
    }, tcsr);

    std::array<uint32_t, 64> output{};
    std::memcpy(output.data(), uram.data() + OUTPUT_ADDR, sizeof(output));
    return check(output == input,
                 "masked u2v/v2u remain contiguous and ignore channel stride");
}

}

int main()
{
    if(!check_tcsr_layout_and_defaults() || !check_strided_load()
       || !check_strided_store() || !check_masked_accesses_stay_contiguous()) {
        return 1;
    }
    std::cout << "vec_channel_stride_test passed\n";
    return 0;
}
