#ifndef __SWIZZLE_H__
#define __SWIZZLE_H__

#include "debugf.hpp"

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
    
    template<long maxMask = 8>
    static inline uint64_t swizzle_addr(uint64_t addr, uint64_t swizzleMBS) {
        uint8_t S = (swizzleMBS >>  0) & 0xff;
        uint8_t B = (swizzleMBS >>  8) & 0xff;
        uint8_t M = (swizzleMBS >> 16) & 0xff;
        uint64_t shiftedAddr = addr >> S;
        uint64_t mask = ((1ull << B) - 1) & ((1 << (maxMask - 1)) - 1);
        uint64_t shiftedMask = mask << M;
        addr = addr ^ (shiftedAddr & shiftedMask);
        // 可选地址范围检查
        if(addr >= 0x10000000) {
            addr -= 0x10000000;
        }
        if(addr >= 1024ul * 1024ul) {
            debugf("Fault! URAM Overwrite!\n");
            addr = 0;           
        }
        return addr;
    }

}


#endif