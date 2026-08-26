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

    for(uint64_t amount = 1; amount <= 5; amount++) {
        const auto dst = RVTNG::execute_vec_shuffle_compress(src, amount);
        const size_t group_size = size_t{1} << amount;
        const size_t packed_lane_count = dst.size() / group_size;
        for(size_t lane = 0; lane < dst.size(); lane++) {
            const size_t packed_lane = lane % packed_lane_count;
            if(!check(dst[lane] == src[packed_lane * group_size],
                      "compress repeated packed-block mapping")) {
                return 1;
            }
        }

        const auto high_bits_dst = RVTNG::execute_vec_shuffle_compress(src, amount | 0x40);
        if(!check(high_bits_dst == dst, "compress ignores parameter bits above bit 5")) {
            return 1;
        }

        const auto decompressed = RVTNG::execute_vec_shuffle_compress(src, amount | 0x20);
        for(size_t lane = 0; lane < decompressed.size(); lane++) {
            if(!check(decompressed[lane] == src[lane / group_size],
                      "decompress group broadcast mapping")) {
                return 1;
            }
        }
        const auto decompress_high_bits = RVTNG::execute_vec_shuffle_compress(
            src, amount | 0x60);
        if(!check(decompress_high_bits == decompressed,
                  "decompress ignores parameter bits above bit 5")) {
            return 1;
        }

        const auto decompress_then_compress = RVTNG::execute_vec_shuffle_compress(
            decompressed, amount);
        for(size_t lane = 0; lane < decompress_then_compress.size(); lane++) {
            if(!check(decompress_then_compress[lane]
                          == src[lane % packed_lane_count],
                      "decompress-compress preserves and repeats packed block")) {
                return 1;
            }
        }

        const auto compress_then_decompress = RVTNG::execute_vec_shuffle_compress(
            dst, amount | 0x20);
        for(size_t lane = 0; lane < compress_then_decompress.size(); lane++) {
            if(!check(compress_then_decompress[lane]
                          == src[(lane / group_size) * group_size],
                      "compress-decompress broadcasts each group-zero lane")) {
                return 1;
            }
        }
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
