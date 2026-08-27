#include "vec.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr uint32_t CUSTOM_32_OPCODE = 0x0b;
constexpr uint64_t INPUT_A_ADDR = 0x1000;
constexpr uint64_t INPUT_B_ADDR = 0x2000;
constexpr uint64_t OUTPUT_ADDR = 0x3000;

constexpr uint32_t regular_load_raw(uint8_t vb)
{
    return (0b010000u << 26) | (static_cast<uint32_t>(vb & 1) << 25)
        | CUSTOM_32_OPCODE;
}

constexpr uint32_t regular_store_raw()
{
    return (0b010001u << 26) | CUSTOM_32_OPCODE;
}

constexpr uint32_t int32_minmax_raw(uint8_t func8)
{
    return (0b100001u << 26) | (16u << 20)
        | (static_cast<uint32_t>(func8) << 7) | CUSTOM_32_OPCODE;
}

constexpr uint32_t fp32_minmax_raw(uint8_t func5, uint8_t rm)
{
    return (0b100010u << 26) | (16u << 20)
        | (static_cast<uint32_t>(func5) << 10)
        | (static_cast<uint32_t>(rm & 7) << 7) | CUSTOM_32_OPCODE;
}

bool check(bool condition, const char* message)
{
    if(!condition) {
        std::cerr << "vec_minmax_execution_test failed: " << message << '\n';
    }
    return condition;
}

std::array<uint32_t, 64> run_case(
    const std::array<uint32_t, 64>& input_a,
    const std::array<uint32_t, 64>& input_b,
    uint32_t operation)
{
    using namespace RVTNG;

    std::vector<uint8_t> uram(1024 * 1024);
    std::memcpy(uram.data() + INPUT_A_ADDR, input_a.data(), sizeof(input_a));
    std::memcpy(uram.data() + INPUT_B_ADDR, input_b.data(), sizeof(input_b));

    std::array<vec_inst_t, VEC_INST_QUEUE_CAPACITY> inst_queue{};
    inst_queue[0] = vec_inst_t{regular_load_raw(0), INPUT_A_ADDR};
    inst_queue[1] = vec_inst_t{regular_load_raw(1), INPUT_B_ADDR};
    inst_queue[2] = vec_inst_t{operation};
    inst_queue[3] = vec_inst_t{regular_store_raw(), OUTPUT_ADDR};

    MPMCQueue<vec_vfork_req_t, 16> vfork_queue;
    MPMCQueue<vec_resp_t, 16> resp_queue;
    vec_sparse_memory_t sparse_memory;
    vec_executer_t executer(
        &vfork_queue, &resp_queue, inst_queue.data(), uram.data(),
        &sparse_memory, 64, 1);
    executer.start();

    vec_vfork_req_t req{};
    req.block_id = 0;
    req.inst_count = 4;
    req.control.epilogue0_end = 4;
    vfork_queue.enqueue(req);
    const vec_resp_t resp = resp_queue.dequeue();
    executer.stop();
    check(resp.block_id == 0 && resp.inst_count == 4,
          "vector block retirement");

    std::array<uint32_t, 64> output{};
    std::memcpy(output.data(), uram.data() + OUTPUT_ADDR, sizeof(output));
    return output;
}

bool check_int32_minmax()
{
    constexpr std::array<int32_t, 8> a_pattern = {
        0, 1, -1, std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::max(), -100, 100, 42
    };
    constexpr std::array<int32_t, 8> b_pattern = {
        0, -1, 1, std::numeric_limits<int32_t>::max(),
        std::numeric_limits<int32_t>::min(), -200, 200, 42
    };
    std::array<uint32_t, 64> input_a{};
    std::array<uint32_t, 64> input_b{};
    for(size_t lane = 0; lane < input_a.size(); ++lane) {
        input_a[lane] = std::bit_cast<uint32_t>(a_pattern[lane % a_pattern.size()]);
        input_b[lane] = std::bit_cast<uint32_t>(b_pattern[lane % b_pattern.size()]);
    }

    const auto maximum_u = run_case(input_a, input_b, int32_minmax_raw(0x08));
    const auto minimum_u = run_case(input_a, input_b, int32_minmax_raw(0x09));
    const auto maximum = run_case(input_a, input_b, int32_minmax_raw(0x0a));
    const auto minimum = run_case(input_a, input_b, int32_minmax_raw(0x0b));
    for(size_t lane = 0; lane < input_a.size(); ++lane) {
        const int32_t a = std::bit_cast<int32_t>(input_a[lane]);
        const int32_t b = std::bit_cast<int32_t>(input_b[lane]);
        if(!check(maximum_u[lane] == (input_a[lane] > input_b[lane]
                                         ? input_a[lane] : input_b[lane]),
                  "int32 maxu result") ||
           !check(minimum_u[lane] == (input_a[lane] < input_b[lane]
                                         ? input_a[lane] : input_b[lane]),
                  "int32 minu result") ||
           !check(maximum[lane] == (a > b ? input_a[lane] : input_b[lane]),
                  "int32 max result") ||
           !check(minimum[lane] == (a < b ? input_a[lane] : input_b[lane]),
                  "int32 min result")) {
            return false;
        }
    }
    return true;
}

bool check_fp32_minmax()
{
    constexpr std::array<uint32_t, 12> input_a_pattern = {
        0x3fc00000, 0xc0000000, 0x7f800000, 0xff800000,
        0x7fc12345, 0x40400000, 0x00000000, 0x80000000,
        0x40a00000, 0x00000001, 0x80000001, 0x7f800001
    };
    constexpr std::array<uint32_t, 12> input_b_pattern = {
        0x40200000, 0xc0400000, 0xff800000, 0x7f800000,
        0x40800000, 0x7fc54321, 0x80000000, 0x00000000,
        0x40a00000, 0x00000000, 0x80000000, 0xc0e00000
    };
    constexpr std::array<uint32_t, 12> maximum_pattern = {
        0x40200000, 0xc0000000, 0x7f800000, 0x7f800000,
        0x40800000, 0x7fc54321, 0x80000000, 0x00000000,
        0x40a00000, 0x00000001, 0x80000000, 0xc0e00000
    };
    constexpr std::array<uint32_t, 12> minimum_pattern = {
        0x3fc00000, 0xc0400000, 0xff800000, 0xff800000,
        0x40800000, 0x7fc54321, 0x80000000, 0x00000000,
        0x40a00000, 0x00000000, 0x80000001, 0xc0e00000
    };

    std::array<uint32_t, 64> input_a{};
    std::array<uint32_t, 64> input_b{};
    for(size_t lane = 0; lane < input_a.size(); ++lane) {
        input_a[lane] = input_a_pattern[lane % input_a_pattern.size()];
        input_b[lane] = input_b_pattern[lane % input_b_pattern.size()];
    }

    const auto maximum = run_case(input_a, input_b, fp32_minmax_raw(0b00100, 7));
    const auto minimum = run_case(input_a, input_b, fp32_minmax_raw(0b00101, 7));
    for(size_t lane = 0; lane < input_a.size(); ++lane) {
        if(!check(maximum[lane] == maximum_pattern[lane % maximum_pattern.size()],
                  "fp32 max ordered/NaN/equal result") ||
           !check(minimum[lane] == minimum_pattern[lane % minimum_pattern.size()],
                  "fp32 min ordered/NaN/equal result")) {
            return false;
        }
    }
    return true;
}

}

int main()
{
    if(!check_int32_minmax() || !check_fp32_minmax()) {
        return 1;
    }
    std::cout << "vec_minmax_execution_test passed\n";
    return 0;
}
