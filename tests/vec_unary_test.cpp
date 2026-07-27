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
    using namespace RVTNG;

    for(uint8_t op = 1; op <= 3; op++) {
        const auto fields = decode_vec_unary(shift_raw(op, 31));
        if(!check(fields.kind == vec_unary_kind_t::shift, "shift class decode")
            || !check(fields.op == op, "three-bit shift operation decode")
            || !check(fields.shift_amount == 31, "five-bit shift amount decode")) {
            return 1;
        }
    }

    const auto exp2 = decode_vec_unary(0x8020100b);
    const auto recip = decode_vec_unary(0x8220580b);
    const auto convert = decode_vec_unary(0x8024308b);
    const auto obsolete_shift = decode_vec_unary(0x8011b00b);
    const auto shuffle = decode_vec_unary(0b011111u << 26);
    if(!check(exp2.kind == vec_unary_kind_t::sfu && exp2.op == 1,
            "exp2 SFU decode")
        || !check(recip.kind == vec_unary_kind_t::sfu && recip.op == 5,
            "reciprocal SFU decode")
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
    if(!check(std::bit_cast<float>(execute_vec_sfu_lane(three, 1)) == 8.0f,
            "exp2 semantics")
        || !check(std::bit_cast<float>(execute_vec_sfu_lane(four, 5)) == 0.25f,
            "reciprocal semantics")) {
        return 1;
    }

    std::cout << "vec_unary_test passed\n";
    return 0;
}
