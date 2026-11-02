#include "vec.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr uint64_t capacity = 8192;

bool check(bool condition, const char* message)
{
    if(!condition) {
        std::cerr << "vec_sparse_memory_test failed: " << message << '\n';
    }
    return condition;
}

bool check_addresses()
{
    RVTNG::vec_sparse_memory_t memory, peer;
    for(uint64_t addr = 0; addr < capacity; addr += 4) {
        if(!check(memory.read_u32(addr) == 0, "initial contents")) { return false; }
        memory.write_u32(addr, addr + 1);
    }
    for(uint64_t addr = 0; addr < capacity; addr += 4) {
        if(!check(memory.read_u32(addr) == addr + 1, "all banks and rows are distinct") ||
           !check(memory.read_u32(addr + capacity) == addr + 1, "8 KiB address wrap") ||
           !check(memory.read_u32(addr + 0x8000000000000000ULL) == addr + 1, "high address bits are ignored") ||
           !check(peer.read_u32(addr) == 0, "per-core storage is independent")) {
            return false;
        }
    }
    memory.write_u32(capacity + 3, 0x12345678);
    memory.write_u32(UINT64_MAX, 0xabcdef01);
    return check(memory.read_u32(0) == 0x12345678, "byte offset bits are ignored") &&
           check(memory.read_u32(capacity - 4) == 0xabcdef01, "last word alias") &&
           check(memory.read_u32(4) == 5, "neighboring word is unchanged");
}

bool check_atomic_aliases()
{
    RVTNG::vec_sparse_memory_t memory;
    memory.write_u32(64, UINT32_MAX);
    if(!check(memory.atomic_update_u32(capacity + 64, 2, 0) == UINT32_MAX &&
              memory.read_u32(64) == 1, "wrapped add returns the old word")) {
        return false;
    }
    memory.write_u32(64, 0x12f0);
    return check(memory.atomic_update_u32(2 * capacity + 64, 0xff, 1) == 0x12f0 &&
                 memory.read_u32(64) == 0xf0, "wrapped and") &&
           check(memory.atomic_update_u32(3 * capacity + 64, 0x80000000, 2) == 0xf0 &&
                 memory.read_u32(64) == 0x800000f0, "wrapped or") &&
           check(memory.atomic_update_u32(4 * capacity + 64, 0x800000ff, 3) == 0x800000f0 &&
                 memory.read_u32(64) == 0xf, "wrapped xor");
}

bool check_vector_boundary()
{
    using namespace RVTNG;
    constexpr uint32_t load = (0b010000u << 26) | 0x0b;
    constexpr uint32_t store = (0b010001u << 26) | 0x0b;
    constexpr uint32_t sparse = 1u << 12;
    std::array<uint32_t, 64> input{}, output{};
    for(size_t lane = 0; lane < input.size(); ++lane) { input[lane] = lane + 100; }
    std::vector<uint8_t> uram(1024);
    std::memcpy(uram.data(), input.data(), sizeof(input));
    std::array<vec_inst_t, VEC_INST_QUEUE_CAPACITY> instructions{};
    instructions[0] = {load, 0};
    instructions[1] = {store | sparse, capacity - 128};
    instructions[2] = {load | sparse, 2 * capacity - 128};
    instructions[3] = {store, 512};
    MPMCQueue<vec_vfork_req_t, 16> requests;
    MPMCQueue<vec_resp_t, 16> responses;
    vec_sparse_memory_t memory;
    vec_executer_t executer(&requests, &responses, instructions.data(), uram.data(), &memory, 64, 1);
    executer.start();
    vec_vfork_req_t request{};
    request.inst_count = 4;
    request.control.epilogue0_end = 4;
    requests.enqueue(request);
    const auto response = responses.dequeue();
    executer.stop();
    std::memcpy(output.data(), uram.data() + 512, sizeof(output));
    return check(response.inst_count == 4, "vector block retirement") &&
           check(output == input, "vector load/store across the SRAM boundary") &&
           check(memory.read_u32(capacity - 128) == input[0] && memory.read_u32(0) == input[32],
                 "vector lanes wrap after base plus offset");
}

}

int main()
{
    if(!check_addresses() || !check_atomic_aliases() || !check_vector_boundary()) { return 1; }
    std::cout << "vec_sparse_memory_test passed\n";
    return 0;
}
