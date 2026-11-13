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

#include "vec_unary.hpp"
#include <bit>
#include <cstdint>
#include <iostream>

namespace {

constexpr uint32_t CUSTOM_32_OPCODE = 0x0b;

constexpr uint32_t shift_raw(uint8_t op, uint8_t amount)
{
    return (0b100000u << 26) | (0b00001u << 20)
        | (static_cast<uint32_t>(op & 7) << 17)
        | (static_cast<uint32_t>(amount & 31) << 12)
        | CUSTOM_32_OPCODE;
}

constexpr uint32_t sfu_raw(uint8_t precision, uint8_t op)
{
    return (0b100000u << 26) | (0b00010u << 20)
        | (static_cast<uint32_t>(precision & 1) << 17)
        | (static_cast<uint32_t>(op & 31) << 12)
        | CUSTOM_32_OPCODE;
}

bool check(bool condition, const char* message)
{
    if(!condition) {
        std::cerr << "vec_unary_test failed: " << message << '\n';
    }
    return condition;
}

}

int main()
{
    using namespace DAYAN;

    for(uint8_t op = 1; op <= 3; op++) {
        const auto fields = decode_vec_unary(shift_raw(op, 31));
        if(!check(fields.kind == vec_unary_kind_t::shift, "shift class decode")
            || !check(fields.op == op, "three-bit shift operation decode")
            || !check(fields.shift_amount == 31, "five-bit shift amount decode")) {
            return 1;
        }
    }

    const auto exp2 = decode_vec_unary(sfu_raw(0, 1));
    const auto recip = decode_vec_unary(sfu_raw(0, 5));
    const auto fast_exp2 = decode_vec_unary(sfu_raw(1, 0));
    const auto fast_recip = decode_vec_unary(sfu_raw(1, 1));
    const auto convert = decode_vec_unary(0x8024308b);
    const auto obsolete_shift = decode_vec_unary(0x8011b00b);
    const auto shuffle = decode_vec_unary(0b011111u << 26);
    if(!check(exp2.kind == vec_unary_kind_t::sfu && exp2.precision == 0
                  && exp2.op == 1,
            "exp2 SFU decode")
        || !check(recip.kind == vec_unary_kind_t::sfu && recip.precision == 0
                      && recip.op == 5,
            "reciprocal SFU decode")
        || !check(fast_exp2.kind == vec_unary_kind_t::sfu
                      && fast_exp2.precision == 1 && fast_exp2.op == 0,
                  "fast exp2 SFU decode")
        || !check(fast_recip.kind == vec_unary_kind_t::sfu
                      && fast_recip.precision == 1 && fast_recip.op == 1,
                  "fast reciprocal SFU decode")
        || !check(convert.kind == vec_unary_kind_t::convert && convert.op == 0,
            "convert must be distinguished from SFU and shift")
        || !check(obsolete_shift.kind == vec_unary_kind_t::invalid,
            "obsolete overlapping shift encoding must be rejected")
        || !check(shuffle.kind == vec_unary_kind_t::invalid,
            "shuffle must not decode as a unary operation")) {
        return 1;
    }

    const uint32_t three = std::bit_cast<uint32_t>(3.0f);
    const uint32_t four = std::bit_cast<uint32_t>(4.0f);
    if(!check(std::bit_cast<float>(execute_vec_sfu_lane(three, 0, 1)) == 8.0f,
            "exp2 semantics")
        || !check(std::bit_cast<float>(execute_vec_sfu_lane(four, 0, 5)) == 0.25f,
            "reciprocal semantics")
        || !check(std::bit_cast<float>(execute_vec_sfu_lane(three, 1, 0)) == 8.0f,
            "fast exp2 semantics")
        || !check(std::bit_cast<float>(execute_vec_sfu_lane(four, 1, 1)) == 0.25f,
            "fast reciprocal semantics")
        || !check(vec_sfu_op_supported(0, 5), "regular reciprocal support")
        || !check(!vec_sfu_op_supported(0, 6), "regular reserved op rejection")
        || !check(vec_sfu_op_supported(1, 1), "fast reciprocal support")
        || !check(!vec_sfu_op_supported(1, 2), "fast reserved op rejection")) {
        return 1;
    }

    std::cout << "vec_unary_test passed\n";
    return 0;
}
