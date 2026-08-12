#include "vec_convert.hpp"
#include "vec_shuffle.hpp"
#include "vec_unary.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace RVTNG;

namespace {

bool check(bool condition, const char* message)
{
    if(!condition) {
        std::cerr << "vec_convert_test failed: " << message << '\n';
    }
    return condition;
}

constexpr uint32_t convert_raw(uint8_t op, uint8_t rm, uint8_t vrs1 = 0,
                               uint8_t vb = 0)
{
    return (0b100000u << 26) | (static_cast<uint32_t>(vb) << 25)
        | (0b00010u << 20) | (0b01u << 18)
        | (static_cast<uint32_t>(op) << 15)
        | (static_cast<uint32_t>(rm) << 12)
        | (static_cast<uint32_t>(vrs1) << 7) | 0x0b;
}

bool check_decode()
{
    for(uint8_t op = 0; op < 8; ++op) {
        for(uint8_t rm = 0; rm < 8; ++rm) {
            const auto fields = decode_vec_unary(convert_raw(op, rm, 17, 1));
            if(!check(fields.kind == vec_unary_kind_t::convert,
                      "convert class decode")
               || !check(fields.op == op, "convert op decode")
               || !check(fields.rounding_mode == rm, "convert rm decode")) {
                return false;
            }
        }
    }
    return true;
}

bool check_s32_fp32()
{
    constexpr auto rne = vec_rounding_mode_t::rne;
    constexpr auto rtz = vec_rounding_mode_t::rtz;
    constexpr auto rdn = vec_rounding_mode_t::rdn;
    constexpr auto rup = vec_rounding_mode_t::rup;
    constexpr auto rmm = vec_rounding_mode_t::rmm;

    if(!check(vec_s32_to_fp32(0, rne) == 0, "s32 zero to fp32")
        || !check(vec_s32_to_fp32(0x00000001, rne) == 0x3f800000,
                 "s32 one to fp32")
        || !check(vec_s32_to_fp32(0xffffffff, rne) == 0xbf800000,
                 "s32 minus one to fp32")
        || !check(vec_s32_to_fp32(0x01000001, rne) == 0x4b800000,
                 "s32 RNE tie-to-even")
        || !check(vec_s32_to_fp32(0x01000001, rtz) == 0x4b800000,
                 "s32 positive RTZ")
        || !check(vec_s32_to_fp32(0x01000001, rup) == 0x4b800001,
                 "s32 positive RUP")
        || !check(vec_s32_to_fp32(0xfeffffff, rdn) == 0xcb800001,
                 "s32 negative RDN")
        || !check(vec_s32_to_fp32(0xfeffffff, rup) == 0xcb800000,
                 "s32 negative RUP")
        || !check(vec_s32_to_fp32(0x01000001, rmm) == 0x4b800001,
                 "s32 RMM tie-away")
        || !check(vec_s32_to_fp32(0x80000000, rne) == 0xcf000000,
                 "s32 minimum to fp32")) {
        return false;
    }
    for(int32_t value = -100000; value <= 100000; ++value) {
        const uint32_t encoded = vec_s32_to_fp32(static_cast<uint32_t>(value), rne);
        if(!check(vec_fp32_to_s32(encoded, rne) == static_cast<uint32_t>(value),
                  "exact small integer round trip")) {
            return false;
        }
    }
    return true;
}

bool check_fp32_s32()
{
    constexpr std::array<uint32_t, 5> positive = {2, 1, 1, 2, 2};
    constexpr std::array<uint32_t, 5> negative = {
        0xfffffffeu, 0xffffffffu, 0xfffffffeu, 0xffffffffu, 0xfffffffeu};
    for(uint8_t rm = 0; rm < 5; ++rm) {
        const auto mode = static_cast<vec_rounding_mode_t>(rm);
        if(!check(vec_fp32_to_s32(0x3fc00000, mode) == positive[rm],
                  "positive fp32 to s32 rounding")
           || !check(vec_fp32_to_s32(0xbfc00000, mode) == negative[rm],
                     "negative fp32 to s32 rounding")) {
            return false;
        }
    }
    return check(vec_fp32_to_s32(0x40200000, vec_rounding_mode_t::rne) == 2,
                 "fp32 RNE tie-to-even")
        && check(vec_fp32_to_s32(0x40200000, vec_rounding_mode_t::rmm) == 3,
                 "fp32 RMM tie-away")
        && check(vec_fp32_to_s32(0x7f800000, vec_rounding_mode_t::rne)
                     == 0x7fffffff,
                 "positive infinity saturation")
        && check(vec_fp32_to_s32(0xff800000, vec_rounding_mode_t::rne)
                     == 0x80000000,
                 "negative infinity saturation")
        && check(vec_fp32_to_s32(0x7fc00000, vec_rounding_mode_t::rne)
                     == 0x7fffffff,
                 "NaN integer result");
}

bool check_fp16_fp32()
{
    if(!check(vec_fp16_to_fp32(0x3c00) == 0x3f800000,
                 "fp16 one to fp32")
        || !check(vec_fp16_to_fp32(0x0001) == 0x33800000,
                 "fp16 minimum subnormal to fp32")
        || !check(vec_fp16_to_fp32(0x8000) == 0x80000000,
                 "fp16 negative zero to fp32")
        || !check(vec_fp16_to_fp32(0x7c00) == 0x7f800000,
                 "fp16 infinity to fp32")
        || !check((vec_fp16_to_fp32(0x7c01) & 0x7f800000) == 0x7f800000,
                 "fp16 NaN exponent to fp32")
        || !check((vec_fp16_to_fp32(0x7c01) & 0x007fffff) != 0,
                 "fp16 NaN payload to fp32")) {
        return false;
    }
    for(uint32_t input = 0; input <= 0xffff; ++input) {
        const bool is_nan = (input & 0x7c00) == 0x7c00
            && (input & 0x03ff) != 0;
        if(!is_nan
           && !check(vec_fp32_to_fp16(vec_fp16_to_fp32(input),
                                     vec_rounding_mode_t::rne) == input,
                     "exact fp16 round trip")) {
            return false;
        }
    }
    return true;
}

bool check_fp32_fp16()
{
    return check(vec_fp32_to_fp16(0x3f800000, vec_rounding_mode_t::rne)
                     == 0x3c00,
                 "fp32 one to fp16")
        && check(vec_fp32_to_fp16(0x3f801000, vec_rounding_mode_t::rne)
                     == 0x3c00,
                 "fp16 RNE tie-to-even")
        && check(vec_fp32_to_fp16(0x3f801000, vec_rounding_mode_t::rmm)
                     == 0x3c01,
                 "fp16 RMM tie-away")
        && check(vec_fp32_to_fp16(0x3f801000, vec_rounding_mode_t::rup)
                     == 0x3c01,
                 "fp16 positive RUP")
        && check(vec_fp32_to_fp16(0xbf801000, vec_rounding_mode_t::rdn)
                     == 0xbc01,
                 "fp16 negative RDN")
        && check(vec_fp32_to_fp16(0x33000000, vec_rounding_mode_t::rne)
                     == 0,
                 "fp16 minimum-subnormal halfway RNE")
        && check(vec_fp32_to_fp16(0x33000000, vec_rounding_mode_t::rmm)
                     == 1,
                 "fp16 minimum-subnormal halfway RMM")
        && check(vec_fp32_to_fp16(0x47800000, vec_rounding_mode_t::rtz)
                     == 0x7bff,
                 "fp16 positive overflow RTZ")
        && check(vec_fp32_to_fp16(0x47800000, vec_rounding_mode_t::rne)
                     == 0x7c00,
                 "fp16 positive overflow RNE");
}

bool check_bf16_fp32()
{
    if(!check(vec_bf16_to_fp32(0x3f80) == 0x3f800000,
                 "bf16 one to fp32")
        || !check(vec_bf16_to_fp32(0x8000) == 0x80000000,
                 "bf16 negative zero to fp32")
        || !check(vec_fp32_to_bf16(0x3f808000, vec_rounding_mode_t::rne)
                     == 0x3f80,
                 "bf16 RNE tie-to-even")
        || !check(vec_fp32_to_bf16(0x3f808000, vec_rounding_mode_t::rmm)
                     == 0x3f81,
                 "bf16 RMM tie-away")
        || !check(vec_fp32_to_bf16(0xbf808000, vec_rounding_mode_t::rdn)
                     == 0xbf81,
                 "bf16 negative RDN")) {
        return false;
    }
    for(uint32_t input = 0; input <= 0xffff; ++input) {
        const bool is_nan = (input & 0x7f80) == 0x7f80
            && (input & 0x007f) != 0;
        if(!is_nan
           && !check(vec_fp32_to_bf16(vec_bf16_to_fp32(input),
                                     vec_rounding_mode_t::rne) == input,
                     "exact bf16 round trip")) {
            return false;
        }
    }
    return true;
}

bool check_bf16_mxfp8()
{
    std::array<uint16_t, 32> ones{};
    ones.fill(0x3f80);
    const auto rne = vec_bf16_to_mxfp8(ones, vec_rounding_mode_t::rne);
    if(!check(rne.scale == 0x77, "MXFP8 one block scale")) {
        return false;
    }
    for(uint8_t value : rne.data) {
        if(!check(value == 0x78, "MXFP8 one block payload")) {
            return false;
        }
    }

    std::array<uint16_t, 32> special{};
    special[0] = 0x43e0;
    special[1] = 0xc3e0;
    special[2] = 0x3f80;
    special[3] = 0x7f80;
    special[4] = 0x7fc1;
    const auto converted = vec_bf16_to_mxfp8(
        special, vec_rounding_mode_t::rne);
    if(!check(converted.scale == 0x7f, "MXFP8 finite maximum scale")
       || !check(converted.data[0] == 0x7e, "MXFP8 positive maximum")
       || !check(converted.data[1] == 0xfe, "MXFP8 negative maximum")
       || !check(converted.data[2] == 0x38, "MXFP8 one with unit scale")
       || !check(converted.data[3] == 0x7e, "MXFP8 infinity saturation")
       || !check(converted.data[4] == 0x7f, "MXFP8 NaN encoding")) {
        return false;
    }

    std::array<uint16_t, 32> tie{};
    tie.fill(0x3f88);
    const auto tie_rne = vec_bf16_to_mxfp8(tie, vec_rounding_mode_t::rne);
    const auto tie_rmm = vec_bf16_to_mxfp8(tie, vec_rounding_mode_t::rmm);
    if(!check(tie_rne.data[0] == 0x78, "MXFP8 RNE tie-to-even")
       || !check(tie_rmm.data[0] == 0x79, "MXFP8 RMM tie-away")) {
        return false;
    }

    std::array<uint32_t, 64> src{};
    for(size_t subcore = 0; subcore < 4; ++subcore) {
        const uint16_t value = static_cast<uint16_t>(0x3f80 + subcore * 0x80);
        for(size_t word = 0; word < 16; ++word) {
            src[subcore * 16 + word] = value | (static_cast<uint32_t>(value) << 16);
        }
    }
    const auto dst = execute_vec_convert(src, 6, 0);
    return check(dst[8] == 0x77, "MXFP8 subcore 0 scale")
        && check(dst[24] == 0x78, "MXFP8 subcore 1 scale")
        && check(dst[40] == 0x79, "MXFP8 subcore 2 scale")
        && check(dst[56] == 0x7a, "MXFP8 subcore 3 scale")
        && check(dst[0] == 0x78787878, "MXFP8 subcore 0 payload placement")
        && check(dst[16] == 0x78787878, "MXFP8 subcore 1 payload placement")
        && check(dst[32] == 0x78787878, "MXFP8 subcore 2 payload placement")
        && check(dst[48] == 0x78787878, "MXFP8 subcore 3 payload placement");
}

bool check_lane_semantics_and_unsupported()
{
    if(!check(execute_vec_convert_lane(0x00003c00, 2, 0)
                  == 0x3f800000,
              "fp16 source uses low 16 bits")
       || !check(execute_vec_convert_lane(0x00003f80, 3, 0)
                     == 0x3f800000,
                 "bf16 source uses low 16 bits")
       || !check(execute_vec_convert_lane(0x3f800000, 4, 0)
                     == 0x00003c00,
                 "fp16 narrowing result uses low half")
       || !check(execute_vec_convert_lane(0x3f800000, 5, 0)
                     == 0x00003f80,
                 "bf16 narrowing result uses low half")) {
        return false;
    }

    std::array<uint32_t, 64> src{};
    for(size_t lane = 0; lane < src.size(); lane++) {
        src[lane] = 0x3f800000u + static_cast<uint32_t>(lane << 16);
    }
    for(uint8_t op : {uint8_t{4}, uint8_t{5}}) {
        const auto dst = execute_vec_convert(src, op, 0);
        for(size_t lane = 0; lane < src.size(); lane += 2) {
            const uint32_t low = execute_vec_convert_lane(src[lane], op, 0);
            const uint32_t high = execute_vec_convert_lane(src[lane + 1], op, 0);
            if(!check(dst[lane] == (low | (high << 16)),
                      "adjacent narrowing results share even lane")) {
                return false;
            }
        }
        const auto compact = execute_vec_shuffle_compress(dst, 1);
        for(size_t lane = 0; lane < compact.size() / 2; lane++) {
            if(!check(compact[lane] == dst[2 * lane],
                      "compress preserves packed narrowing order")) {
                return false;
            }
        }
    }

    bool reserved_rm_rejected = false;
    bool mxfp8_lane_rejected = false;
    bool nvfp4_rejected = false;
    try {
        (void)execute_vec_convert_lane(0, 0, 5);
    } catch(const std::invalid_argument&) {
        reserved_rm_rejected = true;
    }
    try {
        (void)execute_vec_convert_lane(0, 6, 0);
    } catch(const std::invalid_argument&) {
        mxfp8_lane_rejected = true;
    }
    try {
        (void)execute_vec_convert_lane(0, 7, 0);
    } catch(const std::invalid_argument&) {
        nvfp4_rejected = true;
    }
    return check(reserved_rm_rejected, "reserved rm must be rejected")
        && check(mxfp8_lane_rejected, "MXFP8 has no single-lane operation")
        && check(nvfp4_rejected, "NVFP4 must remain explicitly unsupported");
}

} // namespace

int main()
{
    if(!check_decode() || !check_s32_fp32() || !check_fp32_s32()
       || !check_fp16_fp32() || !check_fp32_fp16() || !check_bf16_fp32()
       || !check_bf16_mxfp8() || !check_lane_semantics_and_unsupported()) {
        return 1;
    }
    std::cout << "vec_convert_test passed\n";
    return 0;
}
