#include "vec.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool check(bool condition, const char* message)
{
    if(!condition) {
        std::cerr << "vec_block_retirement_test failed: " << message << '\n';
    }
    return condition;
}

}

int main()
{
    RVTNG::vec_tcsr_t outer_tcsr{};
    outer_tcsr.dim2 = 1;
    outer_tcsr.dim3 = 2;
    outer_tcsr.dim4 = 1;
    outer_tcsr.dim5 = 1;
    if(!check(RVTNG::vec_outer_coord_count(outer_tcsr) == 24,
            "outer coordinate count must include dimensions 2 through 5")) {
        return 1;
    }
    std::vector<unsigned> seen(24);
    for(uint64_t job = 0; job < 4; ++job) {
        for(uint64_t index = job; index < seen.size(); index += 4) {
            const RVTNG::vec_coord_t coord = RVTNG::vec_outer_coord(outer_tcsr, index);
            const uint64_t flattened = coord.d2 + 2 * (coord.d3 + 3 * (coord.d4 + 2 * coord.d5));
            ++seen[flattened];
        }
    }
    if(!check(seen == std::vector<unsigned>(24, 1),
            "strided worker jobs must cover every outer coordinate once")) {
        return 1;
    }

    RVTNG::vec_block_retirement_t blocks;
    RVTNG::vec_scoreboard_fences_t fences;
    std::vector<std::string> callbacks;

    auto drain = [&]() {
        fences.drain_ready(
            [&](uint64_t target) {
                return blocks.reached(target);
            },
            [&](const RVTNG::vec_scoreboard_fence_t& fence) {
                callbacks.push_back(
                    fence.kind == RVTNG::vec_scoreboard_fence_kind_t::produce
                        ? "produce:" + std::to_string(fence.data)
                        : "bar:" + std::to_string(fence.rd));
            });
    };

    unsigned polls = 0;
    const uint64_t empty_target = blocks.snapshot();
    blocks.wait_until(empty_target, [&polls]() {
        ++polls;
    });
    if(!check(empty_target == 0 && polls == 0,
            "a scoreboard request without a preceding vFork must not wait")) {
        return 1;
    }
    fences.enqueue({
        empty_target,
        RVTNG::vec_scoreboard_fence_kind_t::produce,
        7,
        0,
        0
    });
    drain();
    if(!check(callbacks == std::vector<std::string>{"produce:7"},
            "a target-zero marker must complete immediately")) {
        return 1;
    }

    const uint64_t block0 = blocks.issue();
    const uint64_t first_target = blocks.snapshot();
    fences.enqueue({
        first_target,
        RVTNG::vec_scoreboard_fence_kind_t::produce,
        11,
        0,
        0
    });
    fences.enqueue({
        first_target,
        RVTNG::vec_scoreboard_fence_kind_t::bar,
        0,
        98,
        3
    });
    const uint64_t block1 = blocks.issue();
    fences.enqueue({
        blocks.snapshot(),
        RVTNG::vec_scoreboard_fence_kind_t::produce,
        13,
        99,
        5
    });
    if(!check(block0 == 0 && block1 == 1 && first_target == 1,
            "a scoreboard target must snapshot only already-issued blocks")
        || !check(!blocks.reached(first_target),
            "a callback must remain blocked before its target retires")) {
        return 1;
    }
    drain();
    if(!check(callbacks == std::vector<std::string>{"produce:7"},
            "pending markers must not callback before retirement")) {
        return 1;
    }

    blocks.wait_until(first_target, [&]() {
        ++polls;
        blocks.retire(block0);
    });
    drain();
    if(!check(blocks.reached(first_target),
            "the first callback must unblock when block zero retires")
        || !check(!blocks.reached(blocks.snapshot()),
            "a later vFork must remain outside the earlier snapshot")
        || !check(callbacks == std::vector<std::string>{
                "produce:7", "produce:11", "bar:3"},
            "same-target markers must callback in order while a later block remains active")) {
        return 1;
    }

    const uint64_t second_target = blocks.snapshot();
    blocks.wait_until(second_target, [&]() {
        ++polls;
        blocks.retire(block1);
    });
    drain();
    if(!check(blocks.retired_blocks() == second_target,
            "all blocks through the second snapshot must retire")
        || !check(callbacks == std::vector<std::string>{
                "produce:7", "produce:11", "bar:3", "produce:13"},
            "produce and bar callbacks must preserve marker order")) {
        return 1;
    }

    bool rejected_out_of_order = false;
    RVTNG::vec_block_retirement_t ordering;
    ordering.issue();
    ordering.issue();
    try {
        ordering.retire(1);
    } catch(const std::logic_error&) {
        rejected_out_of_order = true;
    }
    if(!check(rejected_out_of_order,
            "out-of-order block retirement must be rejected")) {
        return 1;
    }

    std::cout << "vec_block_retirement_test passed\n";
    return 0;
}
