#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace RVTNG {

inline std::array<uint32_t, 64> execute_vec_shuffle_compress(
    const std::array<uint32_t, 64>& src, uint64_t amount)
{
    std::array<uint32_t, 64> dst{};
    const uint64_t ctrl = amount & 0x3f;
    const uint64_t p = ctrl & 0x1f;
    if(p == 0 || p >= 6) {
        return dst;
    }

    const size_t group_size = size_t{1} << p;
    const size_t packed_lane_count = dst.size() / group_size;
    for(size_t lane = 0; lane < dst.size(); lane++) {
        if((ctrl & 0x20) != 0) {
            dst[lane] = src[lane / group_size];
        } else {
            dst[lane] = src[(lane % packed_lane_count) * group_size];
        }
    }
    return dst;
}

}
