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

#ifndef DAYAN_VEC_CONVERT_HPP
#define DAYAN_VEC_CONVERT_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace DAYAN {

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
    bf16_to_mxfp4 = 7,
};

constexpr bool vec_rounding_mode_supported(uint8_t mode)
{
    return mode <= static_cast<uint8_t>(vec_rounding_mode_t::rmm);
}

constexpr bool vec_convert_op_supported(uint8_t op)
{
    return op <= static_cast<uint8_t>(vec_convert_op_t::bf16_to_mxfp8);
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

struct vec_mxfp8_block_t {
    std::array<uint8_t, 32> data{};
    uint8_t scale = 0;
};

constexpr void vec_normalize_bf16(uint16_t input, int& exponent,
                                  uint16_t& significand)
{
    const uint16_t raw_exponent = (input >> 7) & 0xff;
    const uint16_t fraction = input & 0x7f;
    if(raw_exponent != 0) {
        exponent = static_cast<int>(raw_exponent) - 127;
        significand = static_cast<uint16_t>(0x80 | fraction);
        return;
    }

    const unsigned leading = 31u - std::countl_zero(
        static_cast<uint32_t>(fraction));
    const unsigned shift = 7u - leading;
    exponent = -126 - static_cast<int>(shift);
    significand = static_cast<uint16_t>(fraction << shift);
}

constexpr uint8_t vec_bf16_block_to_e8m0(
    const std::array<uint16_t, 32>& input)
{
    int max_scale = -128;
    bool any_finite_nonzero = false;
    for(uint16_t value : input) {
        const uint16_t exponent = (value >> 7) & 0xff;
        const uint16_t fraction = value & 0x7f;
        if(exponent == 0xff || (exponent == 0 && fraction == 0)) {
            continue;
        }
        int normalized_exponent;
        uint16_t significand;
        vec_normalize_bf16(value, normalized_exponent, significand);
        const int candidate = normalized_exponent
            - (significand > 0xe0 ? 7 : 8);
        max_scale = max_scale > candidate ? max_scale : candidate;
        any_finite_nonzero = true;
    }
    if(!any_finite_nonzero) {
        return 127;
    }
    const int scale = max_scale < -127 ? -127
        : (max_scale > 127 ? 127 : max_scale);
    return static_cast<uint8_t>(scale + 127);
}

constexpr uint8_t vec_bf16_to_fp8_e4m3(uint16_t input, uint8_t e8m0,
                                        vec_rounding_mode_t mode)
{
    const bool sign = (input >> 15) != 0;
    const uint8_t sign_bits = sign ? 0x80 : 0;
    const uint16_t exponent = (input >> 7) & 0xff;
    const uint16_t fraction = input & 0x7f;
    if(exponent == 0 && fraction == 0) {
        return sign_bits;
    }
    if(exponent == 0xff) {
        return static_cast<uint8_t>(sign_bits | (fraction == 0 ? 0x7e : 0x7f));
    }

    int normalized_exponent;
    uint16_t significand;
    vec_normalize_bf16(input, normalized_exponent, significand);
    int scaled_exponent = normalized_exponent
        - (static_cast<int>(e8m0) - 127);
    if(scaled_exponent > 8) {
        return static_cast<uint8_t>(sign_bits | 0x7e);
    }

    if(scaled_exponent >= -6) {
        uint16_t rounded = static_cast<uint16_t>(
            vec_convert_detail::round_shift_right(significand, 4, sign, mode));
        if(rounded >= 0x10) {
            rounded = 0x8;
            ++scaled_exponent;
        }
        const int output_exponent = scaled_exponent + 7;
        if(output_exponent > 15
           || (output_exponent == 15 && (rounded & 0x7) == 0x7)) {
            return static_cast<uint8_t>(sign_bits | 0x7e);
        }
        return static_cast<uint8_t>(sign_bits
            | (static_cast<uint8_t>(output_exponent) << 3)
            | (rounded & 0x7));
    }

    const unsigned shift = static_cast<unsigned>(-(scaled_exponent + 2));
    const uint16_t rounded = static_cast<uint16_t>(
        vec_convert_detail::round_shift_right(significand, shift, sign, mode));
    if(rounded >= 8) {
        return static_cast<uint8_t>(sign_bits | 0x08);
    }
    return static_cast<uint8_t>(sign_bits | rounded);
}

constexpr vec_mxfp8_block_t vec_bf16_to_mxfp8(
    const std::array<uint16_t, 32>& input, vec_rounding_mode_t mode)
{
    vec_mxfp8_block_t output;
    output.scale = vec_bf16_block_to_e8m0(input);
    for(size_t i = 0; i < input.size(); ++i) {
        output.data[i] = vec_bf16_to_fp8_e4m3(input[i], output.scale, mode);
    }
    return output;
}

inline uint32_t execute_vec_convert_lane(uint32_t input, uint8_t op,
                                         uint8_t rounding_mode)
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
        return vec_fp32_to_fp16(input, mode);
    case vec_convert_op_t::fp32_to_bf16:
        return vec_fp32_to_bf16(input, mode);
    case vec_convert_op_t::bf16_to_mxfp8:
    case vec_convert_op_t::bf16_to_mxfp4:
        break;
    }
    throw std::invalid_argument("unsupported VEC convert operation");
}

template <size_t LaneCount>
inline std::array<uint32_t, LaneCount> execute_vec_convert(
    const std::array<uint32_t, LaneCount>& src, uint8_t op,
    uint8_t rounding_mode)
{
    static_assert(LaneCount % 2 == 0);
    std::array<uint32_t, LaneCount> dst{};
    if(op == static_cast<uint8_t>(vec_convert_op_t::bf16_to_mxfp8)) {
        static_assert(LaneCount % 16 == 0);
        if(!vec_rounding_mode_supported(rounding_mode)) {
            throw std::invalid_argument("unsupported VEC rounding mode");
        }
        for(size_t base = 0; base < LaneCount; base += 16) {
            std::array<uint16_t, 32> block_input{};
            for(size_t word = 0; word < 16; ++word) {
                block_input[2 * word] = static_cast<uint16_t>(src[base + word]);
                block_input[2 * word + 1] = static_cast<uint16_t>(src[base + word] >> 16);
            }
            const auto block = vec_bf16_to_mxfp8(
                block_input, static_cast<vec_rounding_mode_t>(rounding_mode));
            for(size_t word = 0; word < 8; ++word) {
                dst[base + word] = static_cast<uint32_t>(block.data[4 * word])
                    | (static_cast<uint32_t>(block.data[4 * word + 1]) << 8)
                    | (static_cast<uint32_t>(block.data[4 * word + 2]) << 16)
                    | (static_cast<uint32_t>(block.data[4 * word + 3]) << 24);
            }
            dst[base + 8] = block.scale;
        }
        return dst;
    }
    const bool widen = op == static_cast<uint8_t>(vec_convert_op_t::fp16_to_fp32)
        || op == static_cast<uint8_t>(vec_convert_op_t::bf16_to_fp32);
    if(widen) {
        for(size_t lane = 0; lane < LaneCount; lane += 2) {
            dst[lane] = execute_vec_convert_lane(
                src[lane], op, rounding_mode);
            dst[lane + 1] = execute_vec_convert_lane(
                src[lane] >> 16, op, rounding_mode);
        }
        return dst;
    }
    const bool narrow = op == static_cast<uint8_t>(vec_convert_op_t::fp32_to_fp16)
        || op == static_cast<uint8_t>(vec_convert_op_t::fp32_to_bf16);
    if(narrow) {
        for(size_t lane = 0; lane < LaneCount; lane += 2) {
            const uint32_t low = execute_vec_convert_lane(
                src[lane], op, rounding_mode);
            const uint32_t high = execute_vec_convert_lane(
                src[lane + 1], op, rounding_mode);
            dst[lane] = low | (high << 16);
        }
        return dst;
    }

    for(size_t lane = 0; lane < LaneCount; lane++) {
        dst[lane] = execute_vec_convert_lane(src[lane], op, rounding_mode);
    }
    return dst;
}

} // namespace DAYAN

#endif
