#ifndef DAYAN_VEC_UNARY_HPP
#define DAYAN_VEC_UNARY_HPP

#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace DAYAN {

enum class vec_unary_kind_t : uint8_t {
    invalid,
    shift,
    sfu,
    convert,
};

struct vec_unary_fields_t {
    vec_unary_kind_t kind = vec_unary_kind_t::invalid;
    uint8_t op = 0;
    uint8_t shift_amount = 0;
    uint8_t rounding_mode = 0;
    uint8_t precision = 0;
};

constexpr vec_unary_fields_t decode_vec_unary(uint32_t raw)
{
    if(((raw >> 26) & 0x3f) != 0b100000) {
        return {};
    }

    const uint8_t unit = (raw >> 20) & 0x1f;
    if(unit == 0b00001) {
        const uint8_t op = (raw >> 17) & 0x7;
        if(op >= 1 && op <= 3) {
            return {vec_unary_kind_t::shift, op,
                    static_cast<uint8_t>((raw >> 12) & 0x1f), 0};
        }
        return {};
    }

    if(unit == 0b00010) {
        const uint8_t subclass = (raw >> 18) & 0x3;
        if(subclass == 0b00) {
            return {vec_unary_kind_t::sfu,
                    static_cast<uint8_t>((raw >> 12) & 0x1f), 0, 0,
                    static_cast<uint8_t>((raw >> 17) & 0x1)};
        }
        if(subclass == 0b01) {
            return {vec_unary_kind_t::convert,
                    static_cast<uint8_t>((raw >> 15) & 0x7), 0,
                    static_cast<uint8_t>((raw >> 12) & 0x7)};
        }
    }
    return {};
}

inline bool vec_sfu_op_supported(uint8_t precision, uint8_t op)
{
    return precision == 0 ? op <= 5 : precision == 1 && op <= 1;
}

inline uint32_t execute_vec_sfu_lane(uint32_t input, uint8_t precision,
                                    uint8_t op)
{
    const float x = std::bit_cast<float>(input);
    float result;

    // The ISA leaves ERU approximation accuracy implementation-defined.  The
    // functional model uses the same deterministic FP32 semantics as SFU.
    if(precision == 1) {
        switch(op) {
        case 0:
            result = std::exp2(x);
            break;
        case 1:
            result = 1.0f / x;
            break;
        default:
            throw std::invalid_argument("unsupported fast VEC SFU operation");
        }
        return std::bit_cast<uint32_t>(result);
    }

    if(precision != 0) {
        throw std::invalid_argument("unsupported VEC SFU precision");
    }

    switch(op) {
    case 0:
        result = std::log2(x);
        break;
    case 1:
        result = std::exp2(x);
        break;
    case 2:
        result = std::sin(x);
        break;
    case 3:
        result = std::cos(x);
        break;
    case 4:
        result = 1.0f / std::sqrt(x);
        break;
    case 5:
        result = 1.0f / x;
        break;
    default:
        throw std::invalid_argument("unsupported VEC SFU operation");
    }
    return std::bit_cast<uint32_t>(result);
}

}

#endif
