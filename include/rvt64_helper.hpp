#include "utils.hpp"
#include "rvt64.hpp"
#include "swizzle.hpp"
#include <atomic>
#include <thread>
#include <functional>
#include <cassert>


namespace RVTNG {

    template<typename T>
    static inline void* rvt64_get_uram_ptr(uint8_t* uram, ulong base, ulong dim0, ulong dim1, ulong dim2, ulong dim3, const rvt64_tcsr_t& tcsr, int src, uint32_t type_size) {
        int stride0, stride1, stride2, stride3;
        stride0 = type_size;
        switch (src) {
            case 0:
                stride1 = tcsr.Src0Dim1Stride;
                stride2 = tcsr.Src0Dim2Stride;
                stride3 = tcsr.Src0Dim3Stride;
                break;
            case 1:
                stride1 = tcsr.Src1Dim1Stride;
                stride2 = tcsr.Src1Dim2Stride;
                stride3 = tcsr.Src1Dim3Stride;
                break;
            case 2:
                stride1 = tcsr.DstDim1Stride;
                stride2 = tcsr.DstDim2Stride;
                stride3 = tcsr.DstDim3Stride;
                break;
            default: 
                assert(0);
        }

        auto offset =
              dim0 * stride0
            + dim1 * stride1
            + dim2 * stride2
            + dim3 * stride3;
        base = base & 0xfffff;// 1M
        auto addr = offset + base;
        return (void*)(uram + addr);
    }

    template<typename T>
    static inline void rvt64_write_uram(uint8_t* uram, ulong base, ulong dim0, ulong dim1, ulong dim2, ulong dim3, const rvt64_tcsr_t& tcsr, int src, uint32_t type_size, const T& data) {
        // memcpy(
        //     rvt64_get_uram_ptr<T>(uram, base, dim0, dim1, dim2, dim3, tcsr), 
        //     (void*)&data,
        //     sizeof(T)
        // );
        *(T*)rvt64_get_uram_ptr<T>(uram, base, dim0, dim1, dim2, dim3, tcsr, src, type_size) = data;
    }

    template<typename T>
    static inline T rvt64_read_uram(uint8_t* uram, ulong base, ulong dim0, ulong dim1, ulong dim2, ulong dim3, const rvt64_tcsr_t& tcsr, int src, uint32_t type_size) {
        // memcpy(
        //     (void*)&data,
        //     rvt64_get_uram_ptr<T>(uram, base, dim0, dim1, dim2, dim3, tcsr),
        //     sizeof(T)
        // );
        return  *(T*)rvt64_get_uram_ptr<T>(uram, base, dim0, dim1, dim2, dim3, tcsr, src, type_size);;
    }

    template<typename TIN, typename TOUT, typename FUNC>
    static inline void rvt64_handle_binary_operation(
        FUNC func,
        uint8_t* uram, ulong broadcast,
        ulong a_base, ulong b_base, ulong c_base, const rvt64_tcsr_t& tcsr, uint32_t data_size) {
        // Check broadcast
        {
            if(broadcast) {
                if(tcsr.dim0 != 1) {
                    debugf("* Error: Broadcast 0 is not supported for non-1D arrays.\n");
                    return;
                }
            }
        }
        for(ulong dim3 = 0; dim3 <= tcsr.dim3; dim3++) {
            for(ulong dim2 = 0; dim2 <= tcsr.dim2; dim2++) {
                for(ulong dim1 = 0; dim1 <= tcsr.dim1; dim1++) {
                    for(ulong dim0 = 0; dim0 <= tcsr.dim0; dim0++) {
                        TIN a = rvt64_read_uram<TIN>(uram, a_base, dim0, dim1, dim2, dim3, tcsr, 0, data_size);
                        TIN b = rvt64_read_uram<TIN>(uram, b_base, 
                            broadcast ? 0 : dim0, 
                            dim1,
                            dim2,
                            dim3, tcsr, 1, data_size);
                        TOUT c = func(a, b);
                        // printf("%f %f %f\n", (float)a, (float)b, (float)c);
                        rvt64_write_uram<TOUT>(uram, c_base, dim0, dim1, dim2, dim3, tcsr, 2, data_size, c);    
                    }
                }
            }
        }
    }

    
    template<typename TIN, typename TOUT, typename FUNC>
    static inline void rvt64_handle_unary_operation(
        FUNC func,
        uint8_t* uram,
        ulong x_base, ulong y_base, const rvt64_tcsr_t& tcsr, uint32_t src_data_size, uint32_t dst_data_size)
    {
        for(ulong dim3 = 0; dim3 <= tcsr.dim3; dim3++) {
            for(ulong dim2 = 0; dim2 <= tcsr.dim2; dim2++) {
                for(ulong dim1 = 0; dim1 <= tcsr.dim1; dim1++) {
                    for(ulong dim0 = 0; dim0 <= tcsr.dim0; dim0++) {
                        TIN x = rvt64_read_uram<TIN>(uram, x_base, dim0, dim1, dim2, dim3, tcsr, 0, src_data_size);
                        TOUT y = func(x);
                        // printf("%f %f\n", (float)x, (float)y);
                        rvt64_write_uram<TOUT>(uram, y_base, dim0, dim1, dim2, dim3, tcsr, 2, dst_data_size, y);
                    }
                }
            }
        }
    }

    
    template<typename TIN, typename TOUT, typename FUNC>
    static inline void rvt64_handle_reduce_operation(
        FUNC func, TIN init_value,
        uint8_t* uram, ulong x_base, ulong y_base, const rvt64_tcsr_t& tcsr, uint32_t data_size)
    {
        for(ulong dim3 = 0; dim3 <= tcsr.dim3; dim3++) {
            for(ulong dim2 = 0; dim2 <= tcsr.dim2; dim2++) {
                for(ulong dim1 = 0; dim1 <= tcsr.dim1; dim1++) {
                    TIN acc = init_value;
                    for(ulong dim0 = 0; dim0 <= tcsr.dim0; dim0++) {
                        TIN x = rvt64_read_uram<TIN>(uram, x_base, dim0, dim1, dim2, dim3, tcsr, 0, data_size);
                        acc = func(acc, x);
                    }
                    rvt64_write_uram<TOUT>(uram, y_base, 0, dim1, dim2, dim3, tcsr, 2, data_size, acc);
                }
            }
        }
    }
} // namespace RVTNG

