#ifndef RVTNG_VEC_CONVERT_HPP
#define RVTNG_VEC_CONVERT_HPP

#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace RVTNG {

enum class vec_rounding_mode_t : uint8_t {
    rne = 0,
    rtz = 1,
    rdn = 2,
    rup = 3,
    rmm = 4,
};

enum class vec_convert_op_t : uint8_t {
    s32_to_fp32 = 0,
    fp32_to_s32 = 1,
    fp16_to_fp32 = 2,
    bf16_to_fp32 = 3,
    fp32_to_fp16 = 4,
    fp32_to_bf16 = 5,
    bf16_to_mxfp8 = 6,
    bf16_to_nvfp4 = 7,
};

constexpr bool vec_rounding_mode_supported(uint8_t mode)
{
    return mode <= static_cast<uint8_t>(vec_rounding_mode_t::rmm);
}

constexpr bool vec_convert_op_supported(uint8_t op)
{
    return op <= static_cast<uint8_t>(vec_convert_op_t::fp32_to_bf16);
}

namespace vec_convert_detail {

constexpr bool round_increment(bool sign, vec_rounding_mode_t mode,
                               bool inexact, bool above_half, bool tie,
                               bool odd)
{
    switch(mode) {
    case vec_rounding_mode_t::rne:
        return above_half || (tie && odd);
    case vec_rounding_mode_t::rtz:
        return false;
    case vec_rounding_mode_t::rdn:
        return sign && inexact;
    case vec_rounding_mode_t::rup:
        return !sign && inexact;
    case vec_rounding_mode_t::rmm:
        return above_half || tie;
    }
    return false;
}

constexpr uint64_t round_shift_right(uint64_t value, unsigned shift,
                                     bool sign, vec_rounding_mode_t mode)
{
    if(shift == 0) {
        return value;
    }

    uint64_t truncated = 0;
    bool inexact = value != 0;
    bool above_half = false;
    bool tie = false;
    if(shift < 64) {
        truncated = value >> shift;
        const uint64_t remainder = value & ((uint64_t{1} << shift) - 1);
        const uint64_t half = uint64_t{1} << (shift - 1);
        inexact = remainder != 0;
        above_half = remainder > half;
        tie = remainder == half;
    } else if(shift == 64) {
        const uint64_t half = uint64_t{1} << 63;
        above_half = value > half;
        tie = value == half;
    }

    return truncated + round_increment(sign, mode, inexact, above_half,
                                       tie, (truncated & 1) != 0);
}

constexpr uint16_t fp16_overflow(bool sign, vec_rounding_mode_t mode)
{
    const bool infinity = mode == vec_rounding_mode_t::rne
        || mode == vec_rounding_mode_t::rmm
        || (mode == vec_rounding_mode_t::rdn && sign)
        || (mode == vec_rounding_mode_t::rup && !sign);
    return static_cast<uint16_t>((sign ? 0x8000u : 0u)
                                 | (infinity ? 0x7c00u : 0x7bffu));
}

} // namespace vec_convert_detail

constexpr uint32_t vec_s32_to_fp32(uint32_t input, vec_rounding_mode_t mode)
{
    const bool sign = (input >> 31) != 0;
    const uint32_t magnitude = sign ? uint32_t{0} - input : input;
    if(magnitude == 0) {
        return 0;
    }

    unsigned exponent = 31u - std::countl_zero(magnitude);
    uint64_t significand;
    if(exponent <= 23) {
        significand = static_cast<uint64_t>(magnitude) << (23 - exponent);
    } else {
        significand = vec_convert_detail::round_shift_right(
            magnitude, exponent - 23, sign, mode);
        if(significand == (uint64_t{1} << 24)) {
            significand >>= 1;
            ++exponent;
        }
    }

    return (sign ? 0x80000000u : 0u)
        | ((exponent + 127) << 23)
        | (static_cast<uint32_t>(significand) & 0x007fffffu);
}

constexpr uint32_t vec_fp32_to_s32(uint32_t input, vec_rounding_mode_t mode)
{
    const bool sign = (input >> 31) != 0;
    const uint32_t exponent = (input >> 23) & 0xff;
    const uint32_t fraction = input & 0x007fffff;

    if(exponent == 0xff) {
        if(fraction != 0) {
            return 0x7fffffffu;
        }
        return sign ? 0x80000000u : 0x7fffffffu;
    }

    const uint64_t significand = exponent == 0
        ? fraction
        : (uint64_t{1} << 23) | fraction;
    const int unbiased_exponent = exponent == 0
        ? -126
        : static_cast<int>(exponent) - 127;

    uint64_t magnitude;
    if(unbiased_exponent >= 23) {
        const unsigned shift = static_cast<unsigned>(unbiased_exponent - 23);
        magnitude = shift >= 40 ? (uint64_t{1} << 63)
                                : significand << shift;
    } else {
        magnitude = vec_convert_detail::round_shift_right(
            significand, static_cast<unsigned>(23 - unbiased_exponent),
            sign, mode);
    }

    if(sign) {
        if(magnitude >= (uint64_t{1} << 31)) {
            return 0x80000000u;
        }
        return static_cast<uint32_t>(-static_cast<int64_t>(magnitude));
    }
    if(magnitude > 0x7fffffffu) {
        return 0x7fffffffu;
    }
    return static_cast<uint32_t>(magnitude);
}

constexpr uint32_t vec_fp16_to_fp32(uint16_t input)
{
    const uint32_t sign = static_cast<uint32_t>(input & 0x8000u) << 16;
    const uint32_t exponent = (input >> 10) & 0x1f;
    const uint32_t fraction = input & 0x03ff;

    if(exponent == 0) {
        if(fraction == 0) {
            return sign;
        }
        const unsigned leading = 31u - std::countl_zero(fraction);
        const uint32_t fp32_exponent = static_cast<uint32_t>(
            static_cast<int>(leading) - 24 + 127);
        const uint32_t fp32_fraction =
            (fraction << (23 - leading)) & 0x007fffffu;
        return sign | (fp32_exponent << 23) | fp32_fraction;
    }
    if(exponent == 0x1f) {
        if(fraction == 0) {
            return sign | 0x7f800000u;
        }
        return sign | 0x7f800000u | (fraction << 13) | 0x00400000u;
    }
    return sign | ((exponent + 112) << 23) | (fraction << 13);
}

constexpr uint32_t vec_bf16_to_fp32(uint16_t input)
{
    return static_cast<uint32_t>(input) << 16;
}

constexpr uint16_t vec_fp32_to_fp16(uint32_t input, vec_rounding_mode_t mode)
{
    const bool sign = (input >> 31) != 0;
    const uint16_t sign_bits = sign ? 0x8000u : 0u;
    const uint32_t exponent = (input >> 23) & 0xff;
    const uint32_t fraction = input & 0x007fffff;

    if(exponent == 0xff) {
        if(fraction == 0) {
            return static_cast<uint16_t>(sign_bits | 0x7c00u);
        }
        uint16_t payload = static_cast<uint16_t>(fraction >> 13);
        payload = static_cast<uint16_t>((payload | 0x0200u) & 0x03ffu);
        return static_cast<uint16_t>(sign_bits | 0x7c00u | payload);
    }

    const uint64_t significand = exponent == 0
        ? fraction
        : (uint64_t{1} << 23) | fraction;
    const int unbiased_exponent = exponent == 0
        ? -126
        : static_cast<int>(exponent) - 127;

    if(unbiased_exponent > 15) {
        return vec_convert_detail::fp16_overflow(sign, mode);
    }
    if(unbiased_exponent >= -14) {
        uint64_t rounded = vec_convert_detail::round_shift_right(
            significand, 13, sign, mode);
        int rounded_exponent = unbiased_exponent;
        if(rounded == 0x0800) {
            rounded >>= 1;
            ++rounded_exponent;
        }
        if(rounded_exponent > 15) {
            return vec_convert_detail::fp16_overflow(sign, mode);
        }
        return static_cast<uint16_t>(
            sign_bits | ((rounded_exponent + 15) << 10)
            | (rounded & 0x03ff));
    }

    const uint64_t rounded = vec_convert_detail::round_shift_right(
        significand, static_cast<unsigned>(-unbiased_exponent - 1),
        sign, mode);
    return static_cast<uint16_t>(sign_bits | static_cast<uint16_t>(rounded));
}

constexpr uint16_t vec_fp32_to_bf16(uint32_t input, vec_rounding_mode_t mode)
{
    const uint32_t exponent = (input >> 23) & 0xff;
    const uint32_t fraction = input & 0x007fffff;
    if(exponent == 0xff && fraction != 0) {
        return static_cast<uint16_t>((input >> 16) | 0x0040u);
    }
    return static_cast<uint16_t>(vec_convert_detail::round_shift_right(
        input, 16, (input >> 31) != 0, mode));
}

inline uint32_t execute_vec_convert_lane(uint32_t input, uint8_t op,
                                         uint8_t rounding_mode,
                                         size_t lane)
{
    if(!vec_rounding_mode_supported(rounding_mode)) {
        throw std::invalid_argument("unsupported VEC rounding mode");
    }
    const auto mode = static_cast<vec_rounding_mode_t>(rounding_mode);
    switch(static_cast<vec_convert_op_t>(op)) {
    case vec_convert_op_t::s32_to_fp32:
        return vec_s32_to_fp32(input, mode);
    case vec_convert_op_t::fp32_to_s32:
        return vec_fp32_to_s32(input, mode);
    case vec_convert_op_t::fp16_to_fp32:
        return vec_fp16_to_fp32(static_cast<uint16_t>(input));
    case vec_convert_op_t::bf16_to_fp32:
        return vec_bf16_to_fp32(static_cast<uint16_t>(input));
    case vec_convert_op_t::fp32_to_fp16:
        return static_cast<uint32_t>(vec_fp32_to_fp16(input, mode))
            << (16 * (lane & 1));
    case vec_convert_op_t::fp32_to_bf16:
        return static_cast<uint32_t>(vec_fp32_to_bf16(input, mode))
            << (16 * (lane & 1));
    case vec_convert_op_t::bf16_to_mxfp8:
    case vec_convert_op_t::bf16_to_nvfp4:
        break;
    }
    throw std::invalid_argument("unsupported VEC convert operation");
}

} // namespace RVTNG

#endif
