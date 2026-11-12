#include "vec_shuffle.hpp"
#include <array>
#include <cstdint>
#include <iostream>

namespace {

constexpr uint32_t CUSTOM_32_OPCODE = 0x0b;

constexpr uint32_t shuffle_raw(uint8_t subfunc)
{
    return (0b011111u << 26)
        | (static_cast<uint32_t>(subfunc & 7) << 12)
        | CUSTOM_32_OPCODE;
}

bool check(bool condition, const char* message)
{
    if(!condition) {
        std::cerr << "vec_shuffle_test failed: " << message << '\n';
    }
    return condition;
}

}

int main()
{
    std::array<uint32_t, 64> src;
    for(size_t lane = 0; lane < src.size(); lane++) {
        src[lane] = 0x1000 + lane;
    }

    for(uint64_t ctrl = 0; ctrl <= 4; ++ctrl) {
        const auto dst = DAYAN::execute_vec_shuffle_compress(src, ctrl);
        const size_t group_size = size_t{1} << (ctrl + 1);
        const size_t packed_lane_count = dst.size() / group_size;
        for(size_t lane = 0; lane < dst.size(); lane++) {
            const size_t packed_lane = lane % packed_lane_count;
            if(!check(dst[lane] == src[packed_lane * group_size],
                      "compress repeated packed-block mapping")) {
                return 1;
            }
        }

        const auto high_bits_dst = DAYAN::execute_vec_shuffle_compress(
            src, ctrl | 0x40);
        if(!check(high_bits_dst == dst, "compress ignores parameter bits above bit 5")) {
            return 1;
        }

        const uint64_t decompress_ctrl = ctrl | 0x20;
        const auto decompressed = DAYAN::execute_vec_shuffle_compress(
            src, decompress_ctrl);
        for(size_t lane = 0; lane < decompressed.size(); lane++) {
            if(!check(decompressed[lane] == src[lane / group_size],
                      "decompress group broadcast mapping")) {
                return 1;
            }
        }
        const auto decompress_high_bits = DAYAN::execute_vec_shuffle_compress(
            src, decompress_ctrl | 0x40);
        if(!check(decompress_high_bits == decompressed,
                  "decompress ignores parameter bits above bit 5")) {
            return 1;
        }

        const auto decompress_then_compress = DAYAN::execute_vec_shuffle_compress(
            decompressed, ctrl);
        for(size_t lane = 0; lane < decompress_then_compress.size(); lane++) {
            if(!check(decompress_then_compress[lane]
                          == src[lane % packed_lane_count],
                      "decompress-compress preserves and repeats packed block")) {
                return 1;
            }
        }

        const auto compress_then_decompress = DAYAN::execute_vec_shuffle_compress(
            dst, decompress_ctrl);
        for(size_t lane = 0; lane < compress_then_decompress.size(); lane++) {
            if(!check(compress_then_decompress[lane]
                          == src[(lane / group_size) * group_size],
                      "compress-decompress broadcasts each group-zero lane")) {
                return 1;
            }
        }
    }

    for(uint64_t ctrl = 0x10; ctrl <= 0x13; ++ctrl) {
        const auto dst = DAYAN::execute_vec_shuffle_compress(src, ctrl);
        const size_t block_size = size_t{1} << (ctrl - 0x0f);
        for(size_t lane = 0; lane < dst.size(); ++lane) {
            const size_t block = lane / (2 * block_size);
            const size_t block_lane = lane % block_size;
            if(!check(dst[lane] == src[block * block_size + block_lane],
                      "block scatter mapping")) {
                return 1;
            }
        }

        const auto high_bits_dst = DAYAN::execute_vec_shuffle_compress(
            src, ctrl | 0x40);
        if(!check(high_bits_dst == dst,
                  "block scatter ignores parameter bits above bit 5")) {
            return 1;
        }
    }

    const auto block4 = DAYAN::execute_vec_shuffle_compress(src, 0x11);
    for(size_t lane = 0; lane < 4; ++lane) {
        if(!check(block4[lane] == src[lane],
                  "block-4 first packed block front copy")
            || !check(block4[lane + 4] == src[lane],
                      "block-4 first packed block duplicate")
            || !check(block4[lane + 8] == src[lane + 4],
                      "block-4 second packed block front copy")) {
            return 1;
        }
    }

    for(uint64_t ctrl = 0x25; ctrl <= 0x27; ++ctrl) {
        const auto dst = DAYAN::execute_vec_shuffle_compress(src, ctrl);
        const size_t numerator = ctrl - 0x20;
        for(size_t lane = 0; lane < dst.size(); ++lane) {
            if(!check(dst[lane] == src[(numerator * lane) / 8],
                      "fractional scatter mapping")) {
                return 1;
            }
        }
    }

    const std::array<uint32_t, 64> zeros{};
    if(!check(DAYAN::execute_vec_shuffle_compress(src, 0x05) == zeros,
              "reserved compress control has deterministic undefined result")) {
        return 1;
    }

    if(!check(((shuffle_raw(0b110) >> 12) & 7) == 0b110,
            "static compress subfunc encoding")
        || !check(((shuffle_raw(0b111) >> 12) & 7) == 0b111,
            "dynamic compress subfunc encoding")) {
        return 1;
    }

    std::cout << "vec_shuffle_test passed\n";
    return 0;
}
