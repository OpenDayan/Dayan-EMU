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

#ifndef __SWIZZLE_H__
#define __SWIZZLE_H__

#include "utils.hpp"

#include <cstdint>

namespace DAYAN {

static constexpr uint64_t kSwizzleUramBytes = 1536ull * 1024ull;
static constexpr uint64_t kSwizzleLocalMask = (uint64_t{1} << 10) - 1;
static constexpr unsigned kSwizzleLocalLsb = 5;

template<unsigned maxMask = 8>
static inline uint64_t swizzle_addr_raw(uint64_t addr,
                                        uint64_t swizzleMBS)
{
    static_assert(maxMask > 0 && maxMask <= 10);

    const unsigned s = swizzleMBS & 0xf;
    const unsigned b = (swizzleMBS >> 8) & 0xf;
    const unsigned m = (swizzleMBS >> 16) & 0xf;
    const uint64_t local_addr =
        (addr >> kSwizzleLocalLsb) & kSwizzleLocalMask;
    const uint64_t shifted_addr = local_addr >> s;
    const uint64_t low_mask = ((uint64_t{1} << b) - 1) &
        ((uint64_t{1} << maxMask) - 1);
    const unsigned mask_shift = (m - kSwizzleLocalLsb) & 0xf;
    const uint64_t mask = (low_mask << mask_shift) & kSwizzleLocalMask;
    const uint64_t swizzled_local = local_addr ^ (shifted_addr & mask);
    const uint64_t local_field_mask =
        kSwizzleLocalMask << kSwizzleLocalLsb;
    return (addr & ~local_field_mask) |
        (swizzled_local << kSwizzleLocalLsb);
}

template<unsigned maxMask = 8>
static inline uint64_t swizzle_addr(uint64_t addr, uint64_t swizzleMBS)
{
    addr = swizzle_addr_raw<maxMask>(addr, swizzleMBS);
    if(addr >= 0x800000) {
        addr -= 0x800000;
    }
    if(addr >= kSwizzleUramBytes) {
        debugf("Fault! URAM Overwrite!\n");
        addr = 0;
    }
    return addr;
}

}


#endif
