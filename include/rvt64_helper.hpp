#include "debugf.hpp"
#include "rvt64.hpp"
#include "swizzle.hpp"
#include <atomic>
#include <thread>
#include <functional>


namespace RVTNG {

    template<typename T>
    static inline void* rvt64_get_uram_ptr(uint8_t* uram, ulong base, ulong dim0, ulong dim1, ulong dim2, ulong dim3, const rvt64_tcsr_t& tcsr) {
        auto offset =
              dim0 * tcsr.stride0
            + dim1 * tcsr.stride1
            + dim2 * tcsr.stride2
            + dim3 * tcsr.stride3;
        base = base & 0xfffff;// 1M
        auto addr = offset * sizeof(T) + base;
        return (void*)(uram + addr);
    }

    template<typename T>
    static inline void rvt64_write_uram(uint8_t* uram, ulong base, ulong dim0, ulong dim1, ulong dim2, ulong dim3, const rvt64_tcsr_t& tcsr, const T& data) {
        // memcpy(
        //     rvt64_get_uram_ptr<T>(uram, base, dim0, dim1, dim2, dim3, tcsr), 
        //     (void*)&data,
        //     sizeof(T)
        // );
        *(T*)rvt64_get_uram_ptr<T>(uram, base, dim0, dim1, dim2, dim3, tcsr) = data;
    }

    template<typename T>
    static inline T rvt64_read_uram(uint8_t* uram, ulong base, ulong dim0, ulong dim1, ulong dim2, ulong dim3, const rvt64_tcsr_t& tcsr) {
        // memcpy(
        //     (void*)&data,
        //     rvt64_get_uram_ptr<T>(uram, base, dim0, dim1, dim2, dim3, tcsr),
        //     sizeof(T)
        // );
        return  *(T*)rvt64_get_uram_ptr<T>(uram, base, dim0, dim1, dim2, dim3, tcsr);;
    }

    template<typename TIN, typename TOUT, typename FUNC>
    static inline void rvt64_handle_binary_operation(
        FUNC func,
        uint8_t* uram, ulong broadcast,
        ulong a_base, const rvt64_tcsr_t& a_tcsr,
        ulong b_base, const rvt64_tcsr_t& b_tcsr,
        ulong c_base, const rvt64_tcsr_t& c_tcsr) {
        bool broadcast_0 = (broadcast & 0b00000001) != 0;
        bool broadcast_1 = (broadcast & 0b00000010) != 0;
        bool broadcast_2 = (broadcast & 0b00000100) != 0;
        bool broadcast_3 = (broadcast & 0b00001000) != 0;
        // Check broadcast
        {
            if(broadcast_0) {
                if(a_tcsr.dim0 != 1 && b_tcsr.dim0 != 1) {
                    debugf("* Error: Broadcast 0 is not supported for non-1D arrays.\n");
                    return;
                }
            }
            if(broadcast_1) {
                if(a_tcsr.dim1 != 1 && b_tcsr.dim1 != 1) {
                    debugf("* Error: Broadcast 1 is not supported for non-1D arrays.\n");
                    return;
                }
            }
            if(broadcast_2) {
                if(a_tcsr.dim2 != 1 && b_tcsr.dim2 != 1) {
                    debugf("* Error: Broadcast 2 is not supported for non-1D arrays.\n");
                    return;
                }
            }
            if(broadcast_3) {
                if(a_tcsr.dim3 != 1 && b_tcsr.dim3 != 1) {
                    debugf("* Error: Broadcast 3 is not supported for non-1D arrays.\n");
                    return;
                }
            }
        }
        for(ulong dim3 = 0; dim3 < c_tcsr.dim3; dim3++) {
            for(ulong dim2 = 0; dim2 < c_tcsr.dim2; dim2++) {
                for(ulong dim1 = 0; dim1 < c_tcsr.dim1; dim1++) {
                    for(ulong dim0 = 0; dim0 < c_tcsr.dim0; dim0++) {
                        TIN a = rvt64_read_uram<TIN>(uram, a_base, dim0, dim1, dim2, dim3, a_tcsr);
                        TIN b = rvt64_read_uram<TIN>(uram, b_base, 
                            broadcast_0 ? 0 : dim0,
                            broadcast_1 ? 0 : dim1,
                            broadcast_2 ? 0 : dim2,
                            broadcast_3 ? 0 : dim3, b_tcsr);
                        TOUT c = func(a, b);
                        rvt64_write_uram<TOUT>(uram, c_base, dim0, dim1, dim2, dim3, c_tcsr, c);    
                    }
                }
            }
        }
    }

    
    template<typename TIN, typename TOUT, typename FUNC>
    static inline void rvt64_handle_unary_operation(
        FUNC func,
        uint8_t* uram,
        ulong x_base, const rvt64_tcsr_t& x_tcsr,
        ulong y_base, const rvt64_tcsr_t& y_tcsr)
    {
        for(ulong dim3 = 0; dim3 < y_tcsr.dim3; dim3++) {
            for(ulong dim2 = 0; dim2 < y_tcsr.dim2; dim2++) {
                for(ulong dim1 = 0; dim1 < y_tcsr.dim1; dim1++) {
                    for(ulong dim0 = 0; dim0 < y_tcsr.dim0; dim0++) {
                        TIN x = rvt64_read_uram<TIN>(uram, x_base, dim0, dim1, dim2, dim3, x_tcsr);
                        TOUT y = func(x);
                        rvt64_write_uram<TOUT>(uram, y_base, dim0, dim1, dim2, dim3, y_tcsr, y);
                    }
                }
            }
        }
    }

    
    template<typename TIN, typename TOUT, typename FUNC>
    static inline void rvt64_handle_reduce_operation(
        FUNC func, TIN init_value,
        uint8_t* uram, uint8_t reduce_dim,
        ulong x_base, const rvt64_tcsr_t& x_tcsr,
        ulong y_base, const rvt64_tcsr_t& y_tcsr)
    {
        switch (reduce_dim) {
        case 0:
            for(ulong dim3 = 0; dim3 < x_tcsr.dim3; dim3++) {
                for(ulong dim2 = 0; dim2 < x_tcsr.dim2; dim2++) {
                    for(ulong dim1 = 0; dim1 < x_tcsr.dim1; dim1++) {
                        TIN acc = init_value;
                        for(ulong dim0 = 0; dim0 < x_tcsr.dim0; dim0++) {
                            TIN x = rvt64_read_uram<TIN>(uram, x_base, dim0, dim1, dim2, dim3, x_tcsr);
                            acc = func(acc, x);
                        }
                        rvt64_write_uram<TOUT>(uram, y_base, 0, dim1, dim2, dim3, y_tcsr, acc);
                    }
                }
            }
            break;
        case 1:
            for(ulong dim3 = 0; dim3 < x_tcsr.dim3; dim3++) {
                for(ulong dim2 = 0; dim2 < x_tcsr.dim2; dim2++) {
                    for(ulong dim0 = 0; dim0 < x_tcsr.dim0; dim0++) {
                        TIN acc = init_value;
                        for(ulong dim1 = 0; dim1 < x_tcsr.dim1; dim1++) {
                            TIN x = rvt64_read_uram<TIN>(uram, x_base, dim0, dim1, dim2, dim3, x_tcsr);
                            acc = func(acc, x);
                        }
                        rvt64_write_uram<TOUT>(uram, y_base, dim0, 0, dim2, dim3, y_tcsr, acc);
                    }
                }
            }
            break;
        case 2:
            for(ulong dim3 = 0; dim3 < x_tcsr.dim3; dim3++) {
                for(ulong dim1 = 0; dim1 < x_tcsr.dim1; dim1++) {
                    for(ulong dim0 = 0; dim0 < x_tcsr.dim0; dim0++) {
                        TIN acc = init_value;
                        for(ulong dim2 = 0; dim2 < x_tcsr.dim2; dim2++) {
                            TIN x = rvt64_read_uram<TIN>(uram, x_base, dim0, dim1, dim2, dim3, x_tcsr);
                            acc = func(acc, x);
                        }
                        rvt64_write_uram<TOUT>(uram, y_base, dim0, dim1, 0, dim3, y_tcsr, acc);
                    }
                }
            }
            break;
        case 3:
            for(ulong dim2 = 0; dim2 < x_tcsr.dim2; dim2++) {
                for(ulong dim1 = 0; dim1 < x_tcsr.dim1; dim1++) {
                    for(ulong dim0 = 0; dim0 < x_tcsr.dim0; dim0++) {
                        TIN acc = init_value;
                        for(ulong dim3 = 0; dim3 < x_tcsr.dim3; dim3++) {
                            TIN x = rvt64_read_uram<TIN>(uram, x_base, dim0, dim1, dim2, dim3, x_tcsr);
                            acc = func(acc, x);
                        }
                        rvt64_write_uram<TOUT>(uram, y_base, dim0, dim1, dim2, 0, y_tcsr, acc);
                    }
                }
            }
            break;
        default:
            debugf("* Error: Reduce dimension %d is not supported.\n", reduce_dim);
            return;
        }
    }

} // namespace RVTNG

