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

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace DAYAN {

inline std::array<uint32_t, 64> execute_vec_shuffle_compress(
    const std::array<uint32_t, 64>& src, uint64_t amount)
{
    std::array<uint32_t, 64> dst{};
    const uint64_t ctrl = amount & 0x3f;
    if(ctrl <= 0x04) {
        const size_t group_size = size_t{1} << (ctrl + 1);
        const size_t packed_lane_count = dst.size() / group_size;
        for(size_t lane = 0; lane < dst.size(); ++lane) {
            dst[lane] = src[(lane % packed_lane_count) * group_size];
        }
        return dst;
    }
    if(ctrl >= 0x10 && ctrl <= 0x13) {
        const size_t block_size = size_t{1} << (ctrl - 0x0f);
        for(size_t lane = 0; lane < dst.size(); ++lane) {
            const size_t block = lane / (2 * block_size);
            const size_t block_lane = lane % block_size;
            dst[lane] = src[block * block_size + block_lane];
        }
        return dst;
    }
    if(ctrl >= 0x20 && ctrl <= 0x24) {
        const size_t group_size = size_t{1} << ((ctrl & 0x1f) + 1);
        for(size_t lane = 0; lane < dst.size(); ++lane) {
            dst[lane] = src[lane / group_size];
        }
        return dst;
    }
    if(ctrl >= 0x25 && ctrl <= 0x27) {
        const size_t numerator = ctrl - 0x20;
        for(size_t lane = 0; lane < dst.size(); ++lane) {
            dst[lane] = src[(numerator * lane) / 8];
        }
    }
    return dst;
}

}
