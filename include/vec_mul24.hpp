#ifndef DAYAN_VEC_MUL24_HPP
#define DAYAN_VEC_MUL24_HPP

#include <cstdint>
#include <stdexcept>

namespace DAYAN {

constexpr uint32_t execute_vec_uint24_mul_lane(uint32_t lhs, uint32_t rhs)
{
    const uint64_t product = static_cast<uint64_t>(lhs & 0x00ffffffu)
        * static_cast<uint64_t>(rhs & 0x00ffffffu);
    return static_cast<uint32_t>(product);
}

constexpr uint32_t execute_vec_int24_mul_lane(uint32_t lhs, uint32_t rhs)
{
    const auto sign_extend_24 = [](uint32_t value) constexpr -> int32_t {
        value &= 0x00ffffffu;
        return (value & 0x00800000u) != 0
            ? static_cast<int32_t>(static_cast<int64_t>(value) - 0x01000000)
            : static_cast<int32_t>(value);
    };
    const int64_t product = static_cast<int64_t>(sign_extend_24(lhs))
        * static_cast<int64_t>(sign_extend_24(rhs));
    return static_cast<uint32_t>(product);
}

constexpr bool vec_mul24_func_supported(uint8_t func8)
{
    return func8 == 0x06 || func8 == 0x07;
}

inline uint32_t execute_vec_mul24_lane(uint8_t func8, uint32_t lhs,
                                       uint32_t rhs)
{
    switch(func8) {
    case 0x06:
        return execute_vec_uint24_mul_lane(lhs, rhs);
    case 0x07:
        return execute_vec_int24_mul_lane(lhs, rhs);
    default:
        throw std::invalid_argument("unsupported VEC mul24 operation");
    }
}

} // namespace DAYAN

#endif
