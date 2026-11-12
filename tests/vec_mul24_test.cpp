#include "vec_mul24.hpp"

#include <cstdint>
#include <iostream>

using namespace DAYAN;

namespace {

bool check(bool condition, const char* message)
{
    if(!condition) {
        std::cerr << "vec_mul24_test failed: " << message << '\n';
    }
    return condition;
}

} // namespace

int main()
{
    if(!check(vec_mul24_func_supported(0x06), "func8 0x06 uint24 decode")
       || !check(vec_mul24_func_supported(0x07), "func8 0x07 int24 decode")
       || !check(!vec_mul24_func_supported(0x05), "func8 0x05 is not mul24")
       || !check(!vec_mul24_func_supported(0x08), "func8 0x08 is not mul24")
       || !check(execute_vec_mul24_lane(0x06, 0x00ffffff, 2) == 0x01fffffe,
                 "func8 0x06 executes uint24")
       || !check(execute_vec_mul24_lane(0x07, 0x00ffffff, 2) == 0xfffffffe,
                 "func8 0x07 executes int24")
       || !check(execute_vec_uint24_mul_lane(3, 7) == 21,
              "uint24 ordinary product")
       || !check(execute_vec_int24_mul_lane(3, 7) == 21,
                 "int24 ordinary product")
       || !check(execute_vec_uint24_mul_lane(0x01ffffff, 2) == 0x01fffffe,
                 "uint24 masks high input bits")
       || !check(execute_vec_int24_mul_lane(0x00ffffff, 2) == 0xfffffffe,
                 "int24 sign extension")
       || !check(execute_vec_uint24_mul_lane(0x00ffffff, 2) == 0x01fffffe,
                 "uint24 does not sign extend")
       || !check(execute_vec_int24_mul_lane(0x00800000, 2) == 0xff000000,
                 "int24 minimum product low 32 bits")
       || !check(execute_vec_uint24_mul_lane(0x00ffffff, 0x00ffffff)
                     == 0xfe000001,
                 "uint24 product truncates to low 32 bits")
       || !check(execute_vec_int24_mul_lane(0x00ffffff, 0x00ffffff) == 1,
                 "int24 negative-one product")) {
        return 1;
    }
    std::cout << "vec_mul24_test passed\n";
    return 0;
}
