#include "vec.hpp"
#include <array>
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

bool run_bf16_mxfp8()
{
    using namespace RVTNG;

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

}

int main()
{
    if(!run_bf16_mxfp8()) {
        return 1;
    }
    std::cout << "vec_convert_execution_test passed\n";
    return 0;
}
