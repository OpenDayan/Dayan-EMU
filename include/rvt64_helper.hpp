#include "utils.hpp"
#include "rvt64.hpp"
#include "swizzle.hpp"
#include <atomic>
#include <thread>
#include <functional>
#include <cassert>
#include <cstring>


namespace DAYAN {

    static inline bool myisnan(float x) {
        return ((*(int*)&x) & 0x7fffffff) > 0x7f800000;
    }

    template<typename T, long src>
    static inline void* rvt64_get_uram_ptr(uint8_t* uram, ulong base, ulong dim0, ulong dim1, ulong dim2, ulong dim3, const rvt64_tcsr_t& tcsr) {
        ulong stride0, stride1, stride2, stride3, swizzle;
        stride0 = sizeof(T);
        switch (src) {
            case 0:
                stride1 = tcsr.Src0Dim1Stride;
                stride2 = tcsr.Src0Dim2Stride;
                stride3 = tcsr.Src0Dim3Stride;
                swizzle = tcsr.Src0SwizzleMBS;
                break;
            case 1:
                stride1 = tcsr.Src1Dim1Stride;
                stride2 = tcsr.Src1Dim2Stride;
                stride3 = tcsr.Src1Dim3Stride;
                swizzle = tcsr.Src1SwizzleMBS;
                break;
            case 2:
                stride1 = tcsr.DstDim1Stride;
                stride2 = tcsr.DstDim2Stride;
                stride3 = tcsr.DstDim3Stride;
                swizzle = tcsr.DstSwizzleMBS;
                break;
            default: 
                assert(0);
        }

        auto offset =
              dim0 * stride0
            + dim1 * stride1
            + dim2 * stride2
            + dim3 * stride3;
        base = base & 0x1fffff; // Architectural URAM address width is 21 bits.
        auto addr = swizzle_addr_raw(offset + base, swizzle);
        return (void*)(uram + addr);
    }

    template<typename T, ulong src>
    static inline void rvt64_write_uram(uint8_t* uram, ulong base, ulong dim0, ulong dim1, ulong dim2, ulong dim3, const rvt64_tcsr_t& tcsr, const T& data) {
        std::memcpy(rvt64_get_uram_ptr<T, src>(uram, base, dim0, dim1, dim2, dim3, tcsr), &data, sizeof(T));
    }

    template<typename T, ulong src>
    static inline T rvt64_read_uram(uint8_t* uram, ulong base, ulong dim0, ulong dim1, ulong dim2, ulong dim3, const rvt64_tcsr_t& tcsr) {
        T data;
        std::memcpy(&data, rvt64_get_uram_ptr<T, src>(uram, base, dim0, dim1, dim2, dim3, tcsr), sizeof(T));
        return data;
    }

    template<typename T>
    static inline T long_bits_to_T(long bits) {
        T f;
        memcpy(&f, &bits, sizeof(f));
        return f;
    }

    template<typename TIN, typename TOUT, typename FUNC>
    static inline void rvt64_handle_binary_operation(
        FUNC func,
        uint8_t* uram, ulong broadcast,
        ulong a_base, ulong b_base, ulong c_base, const rvt64_tcsr_t& tcsr) {
        for(ulong dim3 = 0; dim3 <= tcsr.dim3; dim3++) {
            for(ulong dim2 = 0; dim2 <= tcsr.dim2; dim2++) {
                for(ulong dim1 = 0; dim1 <= tcsr.dim1; dim1++) {
                    for(ulong dim0 = 0; dim0 <= tcsr.dim0; dim0++) {
                        TIN a = broadcast == 0b10 ? long_bits_to_T<TIN>(a_base) :
                                                    rvt64_read_uram<TIN, 0>(uram, a_base, broadcast == 0b01 ? 0 : dim0, dim1, dim2, dim3, tcsr);

                        TIN b = rvt64_read_uram<TIN, 1>(uram, b_base, dim0, dim1, dim2, dim3, tcsr);
                        TOUT c = func(a, b);
                        rvt64_write_uram<TOUT, 2>(uram, c_base, dim0, dim1, dim2, dim3, tcsr, c);
                    }
                }
            }
        }
    }

    
    template<typename TIN, typename TOUT, typename FUNC>
    static inline void rvt64_handle_unary_operation(
        FUNC func,
        uint8_t* uram,
        ulong x_base, ulong y_base, const rvt64_tcsr_t& tcsr)
    {
        for(ulong dim3 = 0; dim3 <= tcsr.dim3; dim3++) {
            for(ulong dim2 = 0; dim2 <= tcsr.dim2; dim2++) {
                for(ulong dim1 = 0; dim1 <= tcsr.dim1; dim1++) {
                    for(ulong dim0 = 0; dim0 <= tcsr.dim0; dim0++) {
                        TIN x = rvt64_read_uram<TIN, 0>(uram, x_base, dim0, dim1, dim2, dim3, tcsr);
                        TOUT y = func(x);
                        if (std::is_invocable_v<TOUT, float>) {
                            if(myisnan(y)) {
                                debugf("[ERROR] RVT64: Encounter NAN in unary operation!\n");
                            }
                        }
                        rvt64_write_uram<TOUT, 2>(uram, y_base, dim0, dim1, dim2, dim3, tcsr, y);
                    }
                }
            }
        }
    }

    template<typename TIN, typename TOUT, typename FUNC>
    static inline void rvt64_handle_reduce_operation(
        FUNC func, uint8_t* uram, ulong x_base, ulong y_base, const rvt64_tcsr_t& tcsr)
    {
        for(ulong dim3 = 0; dim3 <= tcsr.dim3; dim3++) {
            for(ulong dim2 = 0; dim2 <= tcsr.dim2; dim2++) {
                for(ulong dim1 = 0; dim1 <= tcsr.dim1; dim1++) {
                    TIN acc = rvt64_read_uram<TIN, 0>(uram, x_base, 0, dim1, dim2, dim3, tcsr);
                    for(ulong dim0 = 1; dim0 <= tcsr.dim0; dim0++) {
                        TIN x = rvt64_read_uram<TIN, 0>(uram, x_base, dim0, dim1, dim2, dim3, tcsr);
                        acc = func(acc, x);
                    }
                    rvt64_write_uram<TOUT, 2>(uram, y_base, 0, dim1, dim2, dim3, tcsr, acc);
                }
            }
        }
    }

    template<typename TIN, typename TOUT, typename FUNC>
    static inline void rvt64_handle_reduce_index_operation(
        FUNC cmp,
        uint8_t* uram, ulong need_idx,
        ulong x_base, ulong i_base, ulong y_base, const rvt64_tcsr_t& tcsr)
    {
        for(ulong dim3 = 0; dim3 <= tcsr.dim3; dim3++) {
            for(ulong dim2 = 0; dim2 <= tcsr.dim2; dim2++) {
                for(ulong dim1 = 0; dim1 <= tcsr.dim1; dim1++) {
                    TIN acc = rvt64_read_uram<TIN, 0>(uram, x_base, 0, dim1, dim2, dim3, tcsr);
                    ulong idx = 0;
                    for(ulong dim0 = 1; dim0 <= tcsr.dim0; dim0++) {
                        TIN x = rvt64_read_uram<TIN, 0>(uram, x_base, dim0, dim1, dim2, dim3, tcsr);
                        if(cmp(x, acc)) { // cmp ? x : acc
                            acc = x;
                            idx = dim0;
                        }
                    }
                    if(need_idx) rvt64_write_uram<uint32_t, 2>(uram, i_base, 0, dim1, dim2, dim3, tcsr, (uint32_t)idx);
                    rvt64_write_uram<TOUT, 2>(uram, y_base, 0, dim1, dim2, dim3, tcsr, acc);
                }
            }
        }
    }
} // namespace DAYAN
