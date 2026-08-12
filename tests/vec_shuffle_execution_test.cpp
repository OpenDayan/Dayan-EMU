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
constexpr uint64_t PARAM_ADDR = 0x0800;

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

constexpr uint32_t shuffle_raw(uint8_t subfunc)
{
    return (0b011111u << 26)
        | (static_cast<uint32_t>(subfunc & 7) << 12)
        | CUSTOM_32_OPCODE;
}

bool check(bool condition, const char* message)
{
    if(!condition) {
        std::cerr << "vec_shuffle_execution_test failed: " << message << '\n';
    }
    return condition;
}

bool run_case(bool dynamic, uint64_t amount)
{
    using namespace RVTNG;

    std::vector<uint8_t> uram(1024 * 1024);
    std::array<uint32_t, 64> input;
    for(size_t lane = 0; lane < input.size(); lane++) {
        input[lane] = 0x1000 + lane;
    }
    std::memcpy(uram.data() + INPUT_ADDR, input.data(), sizeof(input));
    const uint32_t scalar_amount = amount;
    std::memcpy(uram.data() + PARAM_ADDR, &scalar_amount, sizeof(scalar_amount));

    std::array<vec_inst_t, VEC_INST_QUEUE_CAPACITY> inst_queue{};
    size_t pc = 0;
    if(dynamic) {
        inst_queue[pc++] = vec_inst_t{regular_load_raw(0b110), PARAM_ADDR};
    }
    inst_queue[pc++] = vec_inst_t{regular_load_raw(), INPUT_ADDR};
    inst_queue[pc++] = vec_inst_t{shuffle_raw(dynamic ? 0b111 : 0b110), dynamic ? 0 : amount};
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

    if(!check(resp.block_id == 0 && resp.inst_count == pc, "vector block retirement")) {
        return false;
    }

    std::array<uint32_t, 64> output;
    std::memcpy(output.data(), uram.data() + OUTPUT_ADDR, sizeof(output));
    const size_t group_size = size_t{1} << amount;
    for(size_t lane = 0; lane < output.size() / group_size; lane++) {
        if(!check(output[lane] == input[lane * group_size], "decoded compress lane mapping")) {
            return false;
        }
    }
    return true;
}

}

int main()
{
    if(!run_case(false, 2) || !run_case(true, 4)) {
        return 1;
    }
    std::cout << "vec_shuffle_execution_test passed\n";
    return 0;
}
