#ifndef __SWIZZLE_H__
#define __SWIZZLE_H__

#include "utils.hpp"

#include <cstdint>

// object SwizzleAddr {
//   def apply(addr: UInt, S: UInt, B: UInt, M: UInt, maxMask : Int = 8): UInt = {
//     require(addr.getWidth > maxMask)
//     val shiftedAddr = (addr >> S).asUInt
//     val mask = Cat(0.U((addr.getWidth - maxMask).W), ((1.U(maxMask.W) << B).asUInt - 1.U).asUInt(addr.getWidth - 1, 0))
//     val shiftedMask = (mask << M).asUInt
//     addr ^ (shiftedAddr & shiftedMask)
//   }
// }

namespace RVTNG {

    static constexpr uint64_t kSwizzleUramBytes = 1536ull * 1024ull;

    template<long maxMask = 8>
    static inline uint64_t swizzle_addr_raw(uint64_t addr,
                                            uint64_t swizzleMBS) {
        const uint8_t S = (swizzleMBS >> 0) & 0xff;
        const uint8_t B = (swizzleMBS >> 8) & 0xff;
        const uint8_t M = (swizzleMBS >> 16) & 0xff;
        const uint64_t shiftedAddr = S >= 64 ? 0 : addr >> S;
        const uint64_t maskLimit = maxMask >= 64
            ? ~uint64_t{0} : ((uint64_t{1} << maxMask) - 1);
        const uint64_t mask = (B >= 64
            ? ~uint64_t{0} : ((uint64_t{1} << B) - 1)) & maskLimit;
        const uint64_t shiftedMask = M >= 64 ? 0 : mask << M;
        return addr ^ (shiftedAddr & shiftedMask);
    }
    
    template<long maxMask = 8>
    static inline uint64_t swizzle_addr(uint64_t addr, uint64_t swizzleMBS) {
        addr = swizzle_addr_raw<maxMask>(addr, swizzleMBS);
        // 可选地址范围检查
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
