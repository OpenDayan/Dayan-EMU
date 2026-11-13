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

#include "vec_scalar.hpp"
#include <array>
#include <cstdint>
#include <iostream>

using DAYAN::vec_scalar_op_t;

namespace {

constexpr uint32_t CUSTOM_32_OPCODE = 0x0b;

constexpr uint32_t scalar_raw(
    uint8_t sd, uint8_t func8, uint8_t s2, uint8_t func2, uint8_t s1)
{
    return (0b111110u << 26)
        | (static_cast<uint32_t>(sd & 7) << 23)
        | (static_cast<uint32_t>(func8) << 15)
        | (static_cast<uint32_t>(s2 & 7) << 12)
        | (static_cast<uint32_t>(func2 & 3) << 10)
        | (static_cast<uint32_t>(s1 & 7) << 7)
        | CUSTOM_32_OPCODE;
}

constexpr uint32_t branch_raw(int8_t offset, uint8_t s2, uint8_t func2, uint8_t s1)
{
    return (0b111111u << 26)
        | (static_cast<uint32_t>(static_cast<uint8_t>(offset)) << 15)
        | (static_cast<uint32_t>(s2 & 7) << 12)
        | (static_cast<uint32_t>(func2 & 3) << 10)
        | (static_cast<uint32_t>(s1 & 7) << 7)
        | CUSTOM_32_OPCODE;
}

constexpr uint32_t vv_int32_bitwise_raw(uint8_t vb, uint8_t vrs2, uint8_t vrs1, uint8_t lut)
{
    return (0b100001u << 26)
        | (static_cast<uint32_t>(vb & 1) << 25)
        | (static_cast<uint32_t>(vrs2 & 31) << 20)
        | (static_cast<uint32_t>(vrs1 & 31) << 15)
        | (1u << 11)
        | (static_cast<uint32_t>(lut & 15) << 7)
        | CUSTOM_32_OPCODE;
}

bool check(bool condition, const char* message)
{
    if(!condition) {
        std::cerr << "vec_scalar_test failed: " << message << '\n';
    }
    return condition;
}

}

int main()
{
    using namespace DAYAN;

    const std::array<std::pair<uint32_t, vec_scalar_op_t>, 11> scalar_decodes = {{
        {scalar_raw(1, 0, 2, 0, 3), vec_scalar_op_t::rol},
        {scalar_raw(1, 0, 2, 1, 3), vec_scalar_op_t::sll},
        {scalar_raw(1, 0, 2, 2, 3), vec_scalar_op_t::sra},
        {scalar_raw(1, 0, 2, 3, 3), vec_scalar_op_t::srl},
        {scalar_raw(1, 1, 2, 0, 3), vec_scalar_op_t::add},
        {scalar_raw(1, 1, 2, 1, 3), vec_scalar_op_t::sub},
        {scalar_raw(1, 1, 2, 2, 3), vec_scalar_op_t::mul},
        {scalar_raw(1, 1, 2, 3, 3), vec_scalar_op_t::get_dim},
        {scalar_raw(1, 2, 2, 2, 3), vec_scalar_op_t::get_s},
        {scalar_raw(1, 3, 2, 1, 3), vec_scalar_op_t::get_v},
        {scalar_raw(1, 5, 2, 2, 3), vec_scalar_op_t::bitwise},
    }};
    for(const auto& [raw, expected] : scalar_decodes) {
        if(!check(decode_vec_scalar(raw).op == expected, "scalar opcode decode")) {
            return 1;
        }
    }
    const auto bitwise = decode_vec_scalar(scalar_raw(6, 5, 4, 2, 3));
    if(!check(bitwise.sd == 6 && bitwise.s2 == 4 && bitwise.s1 == 3,
            "scalar register decode")
        || !check(bitwise.logic_lut == 0b0110, "bitwise LUT decode")
        || !check(decode_vec_scalar(0).op == vec_scalar_op_t::invalid,
            "non-vector instruction stays invalid")) {
        return 1;
    }

    constexpr vec_scalar_op_t branch_ops[] = {
        vec_scalar_op_t::beq,
        vec_scalar_op_t::bne,
        vec_scalar_op_t::blt,
        vec_scalar_op_t::bge,
    };
    for(uint8_t func2 = 0; func2 < 4; func2++) {
        const auto decoded = decode_vec_scalar(branch_raw(-2, 4, func2, 5));
        if(!check(decoded.op == branch_ops[func2], "branch opcode decode")
            || !check(decoded.branch_offset == -2, "signed branch offset")) {
            return 1;
        }
    }

    if(!check(execute_vec_scalar_int32(vec_scalar_op_t::rol, 0x80000001u, 1) == 3,
            "rol semantics")
        || !check(execute_vec_scalar_int32(vec_scalar_op_t::sll, 3, 4) == 48,
            "sll semantics")
        || !check(execute_vec_scalar_int32(vec_scalar_op_t::sra, 0x80000000u, 4) == 0xf8000000u,
            "sra semantics")
        || !check(execute_vec_scalar_int32(vec_scalar_op_t::srl, 0x80000000u, 4) == 0x08000000u,
            "srl semantics")
        || !check(execute_vec_scalar_int32(vec_scalar_op_t::add, 0xffffffffu, 2) == 1,
            "wrapping add")
        || !check(execute_vec_scalar_int32(vec_scalar_op_t::sub, 1, 2) == 0xffffffffu,
            "wrapping sub")
        || !check(execute_vec_scalar_int32(vec_scalar_op_t::mul, 0x80000000u, 3) == 0x80000000u,
            "low-32 multiply")) {
        return 1;
    }

    const uint64_t logic_a = 0x80000000ffff0000ull;
    const uint64_t logic_b = 0x00000001ff00ff00ull;
    if(!check(execute_vec_bitwise_lut<uint64_t>(logic_a, logic_b, 0b0110) == (logic_a ^ logic_b),
            "64-bit LUT xor")
        || !check(execute_vec_bitwise_lut<uint64_t>(logic_a, logic_b, 0b1000) == (logic_a & logic_b),
            "64-bit LUT and")
        || !check(execute_vec_bitwise_lut<uint64_t>(logic_a, logic_b, 0b1110) == (logic_a | logic_b),
            "64-bit LUT or")) {
        return 1;
    }

    const uint32_t vv_raw = vv_int32_bitwise_raw(1, 17, 9, 0b0110);
    const uint8_t vv_func8 = (vv_raw >> 7) & 0xff;
    const uint32_t vv_a = 0x8000ffffu;
    const uint32_t vv_b = 0x00ff00ffu;
    if(!check(((vv_raw >> 26) & 0x3f) == 0b100001, "vv.int32 bitwise op6")
        || !check(vv_func8 == 0x16, "vv.int32 bitwise func8 encoding")
        || !check(vec_vv_int32_is_bitwise(vv_func8), "vv.int32 bitwise decode")
        || !check(vec_vv_int32_logic_lut(vv_func8) == 0b0110, "vv.int32 LUT decode")
        || !check(execute_vec_bitwise_lut<uint32_t>(vv_a, vv_b,
                vec_vv_int32_logic_lut(vv_func8)) == (vv_a ^ vv_b),
            "vv.int32 per-lane LUT semantics")) {
        return 1;
    }

    vec_coord_t coord;
    coord.d0 = 2;
    coord.d1 = 3;
    coord.d2 = 5;
    coord.d3 = 7;
    coord.d4 = 11;
    coord.d5 = 13;
    vec_tcsr_t tcsr{};
    tcsr.s[2] = {17, 19, 23, 29, 31, 37, 41};
    tcsr.v[1] = {43, 47, 53, 59, 61, 67, 71};
    const uint64_t expected_s = 17 + 2 * 19 + 3 * 23 + 5 * 29 + 7 * 31 + 11 * 37 + 13 * 41;
    const uint64_t expected_v = 43 + 2 * 47 + 3 * 53 + 5 * 59 + 7 * 61 + 11 * 67 + 13 * 71;
    const uint64_t expected_dims[] = {5, 7, 11, 13};
    for(uint8_t selector = 0; selector < 4; selector++) {
        const auto getdim = decode_vec_scalar(scalar_raw(1, 1, 0, 3, selector));
        if(!check(vec_outer_coord_value(coord, getdim.selector) == expected_dims[selector],
                "getdim s1 selector")) {
            return 1;
        }
    }
    if(!check(vec_iter_value(vec_select_iter(tcsr, false, 2), coord) == expected_s,
            "gets six-dimensional iterator")
        || !check(vec_iter_value(vec_select_iter(tcsr, true, 1), coord) == expected_v,
            "getv six-dimensional iterator")) {
        return 1;
    }

    if(!check(execute_vec_scalar_branch(vec_scalar_op_t::beq, 0x100000000ull, 0x100000000ull),
            "beq compares 64 bits")
        || !check(execute_vec_scalar_branch(vec_scalar_op_t::bne, 0x100000000ull, 0),
            "bne compares 64 bits")
        || !check(execute_vec_scalar_branch(vec_scalar_op_t::blt, 0xffffffffull, 1),
            "blt compares signed low 32 bits")
        || !check(execute_vec_scalar_branch(vec_scalar_op_t::bge, 0xffffffffull, 0xfffffffeull),
            "bge compares signed low 32 bits")
        || !check(vec_scalar_branch_target(1, -2) == -1, "relative negative branch target")) {
        return 1;
    }

    std::cout << "vec_scalar_test passed\n";
    return 0;
}
