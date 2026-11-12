#ifndef DAYAN_VEC_SCALAR_HPP
#define DAYAN_VEC_SCALAR_HPP

#include "vec.hpp"
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace DAYAN {

enum class vec_scalar_op_t : uint8_t {
    rol,
    sll,
    sra,
    srl,
    add,
    sub,
    mul,
    get_dim,
    get_s,
    get_v,
    bitwise,
    beq,
    bne,
    blt,
    bge,
    invalid,
};

struct vec_scalar_fields_t {
    vec_scalar_op_t op = vec_scalar_op_t::invalid;
    uint8_t sd = 0;
    uint8_t s1 = 0;
    uint8_t s2 = 0;
    uint8_t selector = 0;
    uint8_t logic_lut = 0;
    int16_t branch_offset = 0;
};

inline vec_scalar_fields_t decode_vec_scalar(uint32_t raw)
{
    vec_scalar_fields_t fields;
    const uint8_t op6 = (raw >> 26) & 0x3f;
    const uint8_t func8 = (raw >> 15) & 0xff;
    const uint8_t func2 = (raw >> 10) & 0x3;

    fields.sd = (raw >> 23) & 0x7;
    fields.s2 = (raw >> 12) & 0x7;
    fields.s1 = (raw >> 7) & 0x7;
    fields.selector = (func8 == 1 && func2 == 3) ? (fields.s1 & 0x3) : func2;
    fields.logic_lut = (((raw >> 15) & 0x3) << 2) | func2;
    fields.branch_offset = (func8 & 0x80) ? static_cast<int16_t>(func8) - 0x100 : func8;

    if(op6 == 0b111111) {
        constexpr vec_scalar_op_t branch_ops[] = {
            vec_scalar_op_t::beq,
            vec_scalar_op_t::bne,
            vec_scalar_op_t::blt,
            vec_scalar_op_t::bge,
        };
        fields.op = branch_ops[func2];
        return fields;
    }
    if(op6 != 0b111110) {
        return fields;
    }

    if(func8 == 0) {
        constexpr vec_scalar_op_t shift_ops[] = {
            vec_scalar_op_t::rol,
            vec_scalar_op_t::sll,
            vec_scalar_op_t::sra,
            vec_scalar_op_t::srl,
        };
        fields.op = shift_ops[func2];
    } else if(func8 == 1) {
        constexpr vec_scalar_op_t arithmetic_ops[] = {
            vec_scalar_op_t::add,
            vec_scalar_op_t::sub,
            vec_scalar_op_t::mul,
            vec_scalar_op_t::get_dim,
        };
        fields.op = arithmetic_ops[func2];
    } else if(func8 == 2) {
        fields.op = vec_scalar_op_t::get_s;
    } else if(func8 == 3) {
        fields.op = vec_scalar_op_t::get_v;
    } else if((func8 >> 2) == 1) {
        fields.op = vec_scalar_op_t::bitwise;
    }
    return fields;
}

inline bool vec_scalar_is_int32(vec_scalar_op_t op)
{
    return op >= vec_scalar_op_t::rol && op <= vec_scalar_op_t::mul;
}

inline uint32_t execute_vec_scalar_int32(vec_scalar_op_t op, uint32_t src1, uint32_t src2)
{
    const uint32_t shamt = src2 & 31;
    switch(op) {
    case vec_scalar_op_t::rol:
        return std::rotl(src1, static_cast<int>(shamt));
    case vec_scalar_op_t::sll:
        return src1 << shamt;
    case vec_scalar_op_t::sra:
        return static_cast<uint32_t>(static_cast<int32_t>(src1) >> shamt);
    case vec_scalar_op_t::srl:
        return src1 >> shamt;
    case vec_scalar_op_t::add:
        return src1 + src2;
    case vec_scalar_op_t::sub:
        return src1 - src2;
    case vec_scalar_op_t::mul:
        return src1 * src2;
    default:
        return 0;
    }
}

template <typename UInt>
inline UInt execute_vec_bitwise_lut(UInt src1, UInt src2, uint8_t logic_lut)
{
    static_assert(std::is_unsigned_v<UInt>);
    UInt result = 0;
    for(unsigned bit = 0; bit < std::numeric_limits<UInt>::digits; bit++) {
        const unsigned lut_index = (((src1 >> bit) & 1) << 1) | ((src2 >> bit) & 1);
        result |= static_cast<UInt>((logic_lut >> lut_index) & 1) << bit;
    }
    return result;
}

inline bool vec_vv_int32_is_bitwise(uint8_t func8)
{
    return (func8 >> 4) == 1;
}

inline uint8_t vec_vv_int32_logic_lut(uint8_t func8)
{
    return func8 & 0xf;
}

inline bool execute_vec_scalar_branch(vec_scalar_op_t op, uint64_t src1, uint64_t src2)
{
    const int32_t src1_lo = std::bit_cast<int32_t>(static_cast<uint32_t>(src1));
    const int32_t src2_lo = std::bit_cast<int32_t>(static_cast<uint32_t>(src2));
    switch(op) {
    case vec_scalar_op_t::beq:
        return src1 == src2;
    case vec_scalar_op_t::bne:
        return src1 != src2;
    case vec_scalar_op_t::blt:
        return src1_lo < src2_lo;
    case vec_scalar_op_t::bge:
        return src1_lo >= src2_lo;
    default:
        return false;
    }
}

inline int64_t vec_scalar_branch_target(uint64_t pc, int16_t offset)
{
    return static_cast<int64_t>(pc) + offset;
}

inline uint64_t vec_outer_coord_value(const vec_coord_t& coord, uint8_t selector)
{
    const uint64_t dims[] = {coord.d2, coord.d3, coord.d4, coord.d5};
    return dims[selector & 0x3];
}

inline const vec_iterator_t& vec_select_iter(const vec_tcsr_t& tcsr, bool vector_iter, uint8_t selector)
{
    return vector_iter ? tcsr.v[selector & 0x3] : tcsr.s[selector & 0x3];
}

inline uint64_t vec_iter_value(const vec_iterator_t& iter, const vec_coord_t& coord)
{
    return iter.base
        + coord.d0 * iter.dim0Stride
        + coord.d1 * iter.dim1Stride
        + coord.d2 * iter.dim2Stride
        + coord.d3 * iter.dim3Stride
        + coord.d4 * iter.dim4Stride
        + coord.d5 * iter.dim5Stride;
}

}

#endif
