#ifdef NDEBUG
#undef NDEBUG
#endif

#include "fputils.hpp"
#include "mat.hpp"
#include "rvt_decoder.hpp"
#include "swizzle.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

constexpr size_t kUramSize = 1024 * 1024;
constexpr uint64_t kABase = 0x01000;
constexpr uint64_t kBBase = 0x20000;
constexpr uint64_t kCBase = 0x40000;
constexpr uint64_t kAScaleBase = 0x50000;
constexpr uint64_t kBScaleBase = 0x60000;

size_t align32(size_t value) {
    return (value + 31) & ~size_t{31};
}

template<typename T>
void store(std::vector<uint8_t>& memory, uint64_t address, T value) {
    std::memcpy(memory.data() + address, &value, sizeof(value));
}

template<typename T>
T load(const std::vector<uint8_t>& memory, uint64_t address) {
    T value;
    std::memcpy(&value, memory.data() + address, sizeof(value));
    return value;
}

void copy_to(std::vector<uint8_t>& memory, uint64_t address,
             const std::vector<uint8_t>& values) {
    std::copy(values.begin(), values.end(), memory.begin() + address);
}

void copy_to_swizzled(std::vector<uint8_t>& memory, uint64_t address,
                      uint64_t swizzle, const std::vector<uint8_t>& values) {
    for (size_t index = 0; index < values.size(); ++index) {
        memory[RVTNG::swizzle_addr(address + index, swizzle)] = values[index];
    }
}

std::vector<uint8_t> pack_u16(std::initializer_list<uint16_t> values) {
    std::vector<uint8_t> packed(values.size() * sizeof(uint16_t), 0);
    size_t index = 0;
    for (const uint16_t value : values) {
        packed[index * 2] = static_cast<uint8_t>(value);
        packed[index * 2 + 1] = static_cast<uint8_t>(value >> 8);
        ++index;
    }
    return packed;
}

std::vector<uint8_t> pack_f32(std::initializer_list<uint32_t> bitPatterns) {
    std::vector<uint8_t> packed;
    packed.reserve(bitPatterns.size() * sizeof(uint32_t));
    for (const uint32_t bits : bitPatterns) {
        packed.push_back(static_cast<uint8_t>(bits));
        packed.push_back(static_cast<uint8_t>(bits >> 8));
        packed.push_back(static_cast<uint8_t>(bits >> 16));
        packed.push_back(static_cast<uint8_t>(bits >> 24));
    }
    return packed;
}

std::vector<uint8_t> pack_nibbles(const std::vector<uint8_t>& values) {
    std::vector<uint8_t> packed((values.size() + 1) / 2, 0);
    for (size_t index = 0; index < values.size(); ++index) {
        const unsigned shift = (index & 1) == 0 ? 0 : 4;
        packed[index / 2] |= static_cast<uint8_t>((values[index] & 0x0f) << shift);
    }
    return packed;
}

[[noreturn]] void fail(std::string_view name, float actual, float expected) {
    std::cerr << name << ": got " << actual << ", expected " << expected << '\n';
    std::abort();
}

void expect_close(std::string_view name, float actual, float expected,
                  float tolerance = 1e-6f) {
    const float allowed = tolerance * std::max(1.0f, std::fabs(expected));
    if (std::isnan(actual) || std::fabs(actual - expected) > allowed) {
        fail(name, actual, expected);
    }
}

float run_one(uint64_t dataType, uint64_t k,
              const std::vector<uint8_t>& aPayload,
              const std::vector<uint8_t>& bPayload,
              const std::vector<uint8_t>& aScales = {},
              const std::vector<uint8_t>& bScales = {},
              float initialC = 0.0f, bool acc = false) {
    std::vector<uint8_t> memory(kUramSize, 0);
    copy_to(memory, kABase, aPayload);
    copy_to(memory, kBBase, bPayload);
    copy_to(memory, kAScaleBase, aScales);
    copy_to(memory, kBScaleBase, bScales);
    store(memory, kCBase, initialC);

    RVTNG::mat_tcsr_t control{};
    control.k = k - 1;
    control.aStride = align32(aPayload.size());
    control.bStride = align32(bPayload.size());
    control.cStride = 32;
    control.dataType = dataType;
    control.aScaleStride = 32;
    control.bScaleStride = 32;

    assert(RVTNG::execute_matmul(memory.data(), control,
                                kABase, kBBase, kCBase,
                                kAScaleBase, kBScaleBase, acc));
    return load<float>(memory, kCBase);
}

void test_instruction_decoder() {
    constexpr uint32_t mm = 0x2000000bu | (7u << 7) | (3u << 15) |
        (5u << 20) | (6u << 25);
    constexpr uint32_t mma = mm | (1u << 12);
    constexpr uint32_t setptr = 0x2000400bu | (3u << 15) |
        (5u << 20) | (6u << 25);

    static_assert(RVT_TYPE_TF32 == RVT_TYPE_FP32);
    assert(IS_RVT_MAT_MM(mm));
    assert(IS_RVT_MAT_MMA(mma));
    assert(!IS_RVT_MAT_MMA(mm));
    assert(!IS_RVT_MAT_MM_FAMILY(mm | (1u << 28)));
    assert(IS_RVT_MAT_SETPTR(setptr));
    assert(!IS_RVT_MAT_MM_FAMILY(setptr));
    assert(!IS_RVT_MAT_SETPTR(setptr | (1u << 7)));
    assert(RVT_MAT_GROUP(setptr) == 6);
    assert(rvt::is_mat(mm));
    assert(rvt::is_mat(setptr));
}

void test_scalar_decoders() {
    expect_close("FP16 smallest subnormal", FP16ToFP32(0x0001),
                 std::ldexp(1.0f, -24), 0.0f);
    expect_close("FP8 E4M3 subnormal", FP8ToFP32<4>(0x01),
                 std::ldexp(1.0f, -9), 0.0f);
    expect_close("FP8 E5M2 subnormal", FP8ToFP32<5>(0x01),
                 std::ldexp(1.0f, -16), 0.0f);
    expect_close("FP4 E2M1 maximum", FP4E2M1ToFP32(0x07), 6.0f, 0.0f);
    expect_close("FP4 E2M1 negative", FP4E2M1ToFP32(0x0d), -3.0f, 0.0f);
    assert(E8M0ToExponent(0x7f) == 0);
    assert(E8M0ToExponent(0x80) == 1);
}

void test_tf32() {
    // 0x3f801001 is just above the halfway point between adjacent TF32
    // values, so round-to-nearest-even produces 0x3f802000.
    const float result = run_one(RVT_TYPE_TF32, 1,
                                 pack_f32({0x3f801001u}),
                                 pack_f32({0x3f800000u}));
    const uint32_t resultBits = std::bit_cast<uint32_t>(result);
    assert(resultBits == 0x3f802000u);
}

void test_bf16_and_fp16() {
    const auto bf16A = pack_u16({0x3fc0, 0xc000}); // 1.5, -2
    const auto bf16B = pack_u16({0x4000, 0x4040}); // 2, 3
    expect_close("BF16", run_one(RVT_TYPE_BF16, 2, bf16A, bf16B), -3.0f);

    const auto fp16A = pack_u16({0x3e00, 0xc000}); // 1.5, -2
    const auto fp16B = pack_u16({0x4000, 0x4200}); // 2, 3
    expect_close("FP16", run_one(RVT_TYPE_FP16, 2, fp16A, fp16B), -3.0f);
}

void test_fp8() {
    expect_close("FP8 E4M3",
                 run_one(RVT_TYPE_FP8_E4M3, 2,
                         {0x3c, 0xc0}, {0x40, 0x44}),
                 -3.0f);
    expect_close("FP8 E5M2",
                 run_one(RVT_TYPE_FP8_E5M2, 2,
                         {0x3e, 0xc0}, {0x40, 0x42}),
                 -3.0f);
}

void test_mxfp8() {
    constexpr size_t k = 33;
    for (const auto& [dataType, one] : {
             std::pair<uint64_t, uint8_t>{RVT_TYPE_MX_FP8_E4M3, 0x38},
             std::pair<uint64_t, uint8_t>{RVT_TYPE_MX_FP8_E5M2, 0x3c}}) {
        const std::vector<uint8_t> payload(k, one);
        expect_close(dataType == RVT_TYPE_MX_FP8_E4M3
                         ? "MXFP8 E4M3" : "MXFP8 E5M2",
                     run_one(dataType, k, payload, payload,
                             {0x7f, 0x80}, {0x7f, 0x7f}),
                     34.0f);
    }
}

void test_int8_encodings() {
    const std::vector<uint8_t> a = {
        static_cast<uint8_t>(-3), static_cast<uint8_t>(4)};
    const std::vector<uint8_t> b = {
        static_cast<uint8_t>(5), static_cast<uint8_t>(-2)};
    for (const uint64_t dataType : {
             uint64_t{RVT_TYPE_RTL_S8}, uint64_t{RVT_TYPE_S8},
             uint64_t{RVT_TYPE_COMPAT_S8}}) {
        expect_close("INT8 encoding", run_one(dataType, 2, a, b), -23.0f);
    }

    for (const uint64_t dataType : {
             uint64_t{RVT_TYPE_RTL_U8}, uint64_t{RVT_TYPE_U8},
             uint64_t{RVT_TYPE_COMPAT_U8}}) {
        expect_close("UINT8 encoding", run_one(dataType, 1, {250}, {2}), 500.0f);
    }
}

void test_fp4_and_int4() {
    expect_close("FP4 E2M1",
                 run_one(RVT_TYPE_FP4_E2M1, 2,
                         pack_nibbles({0x3, 0xc}),
                         pack_nibbles({0x4, 0x5})),
                 -3.0f);
    expect_close("INT4 low/high nibble order",
                 run_one(RVT_TYPE_S4, 2,
                         pack_nibbles({0xd, 0x4}),
                         pack_nibbles({0x5, 0xe})),
                 -23.0f);
}

void test_mxfp4_scale_layout() {
    constexpr size_t k = 33;
    const auto payload = pack_nibbles(std::vector<uint8_t>(k, 0x2)); // 1.0
    std::vector<uint8_t> memory(kUramSize, 0);
    copy_to(memory, kABase, payload);
    copy_to(memory, kABase + 32, payload);
    copy_to(memory, kBBase, payload);
    copy_to(memory, kBBase + 32, payload);

    // Each row/column has one E8M0 byte per K32 group.
    copy_to(memory, kAScaleBase, {0x7f, 0x80});
    copy_to(memory, kAScaleBase + 32, {0x80, 0x81});
    copy_to(memory, kBScaleBase, {0x7f, 0x7f});
    copy_to(memory, kBScaleBase + 32, {0x7e, 0x7f});

    RVTNG::mat_tcsr_t control{};
    control.k = k - 1;
    control.m = 1;
    control.n = 1;
    control.aStride = 32;
    control.bStride = 32;
    control.cStride = 32;
    control.dataType = RVT_TYPE_MX_FP4_E2M1;
    control.aScaleStride = 32;
    control.bScaleStride = 32;

    assert(RVTNG::execute_matmul(memory.data(), control,
                                kABase, kBBase, kCBase,
                                kAScaleBase, kBScaleBase, false));
    expect_close("MXFP4 row0 col0", load<float>(memory, kCBase), 34.0f);
    expect_close("MXFP4 row0 col1", load<float>(memory, kCBase + 4), 18.0f);
    expect_close("MXFP4 row1 col0", load<float>(memory, kCBase + 32), 68.0f);
    expect_close("MXFP4 row1 col1", load<float>(memory, kCBase + 36), 36.0f);
}

void test_layout_mma_and_k_blocking() {
    std::vector<uint8_t> memory(kUramSize, 0);
    copy_to(memory, kABase, pack_u16({0x3f80, 0x4000}));       // [1, 2]
    copy_to(memory, kABase + 32, pack_u16({0x4040, 0x4080}));  // [3, 4]
    copy_to(memory, kBBase, pack_u16({0x40a0, 0x40c0}));       // col [5, 6]
    copy_to(memory, kBBase + 32, pack_u16({0x40e0, 0x4100}));  // col [7, 8]
    store(memory, kCBase, 100.0f);
    store(memory, kCBase + 4, 100.0f);
    store(memory, kCBase + 32, 100.0f);
    store(memory, kCBase + 36, 100.0f);

    RVTNG::mat_tcsr_t control{};
    control.k = 1;
    control.m = 1;
    control.n = 1;
    control.aStride = 32;
    control.bStride = 32;
    control.cStride = 32;
    control.dataType = RVT_TYPE_BF16;

    assert(RVTNG::execute_matmul(memory.data(), control,
                                kABase, kBBase, kCBase, 0, 0, false));
    const float expected[2][2] = {{17.0f, 23.0f}, {39.0f, 53.0f}};
    for (size_t m = 0; m < 2; ++m) {
        for (size_t n = 0; n < 2; ++n) {
            expect_close("matrix layout",
                         load<float>(memory, kCBase + m * 32 + n * 4),
                         expected[m][n]);
        }
    }

    assert(RVTNG::execute_matmul(memory.data(), control,
                                kABase, kBBase, kCBase, 0, 0, true));
    for (size_t m = 0; m < 2; ++m) {
        for (size_t n = 0; n < 2; ++n) {
            expect_close("MMA accumulation",
                         load<float>(memory, kCBase + m * 32 + n * 4),
                         2.0f * expected[m][n]);
        }
    }

    const std::vector<uint8_t> ones(129, 1);
    expect_close("K128 block accumulation",
                 run_one(RVT_TYPE_RTL_S8, 129, ones, ones), 129.0f);
}

void test_swizzled_addresses() {
    constexpr uint64_t swizzle = (uint64_t{8} << 16) | (uint64_t{2} << 8) | 5;
    std::vector<uint8_t> memory(kUramSize, 0);
    copy_to_swizzled(memory, kABase, swizzle, pack_u16({0x3f80, 0x4000}));
    copy_to_swizzled(memory, kABase + 32, swizzle, pack_u16({0x4040, 0x4080}));
    copy_to_swizzled(memory, kBBase, swizzle, pack_u16({0x40a0, 0x40c0}));
    copy_to_swizzled(memory, kBBase + 32, swizzle, pack_u16({0x40e0, 0x4100}));

    RVTNG::mat_tcsr_t control{};
    control.k = 1;
    control.m = 1;
    control.n = 1;
    control.aStride = 32;
    control.bStride = 32;
    control.cStride = 32;
    control.aSwizzleMBS = swizzle;
    control.bSwizzleMBS = swizzle;
    control.cSwizzleMBS = swizzle;
    control.dataType = RVT_TYPE_BF16;

    assert(RVTNG::execute_matmul(memory.data(), control,
                                kABase, kBBase, kCBase, 0, 0, false));
    const float expected[2][2] = {{17.0f, 23.0f}, {39.0f, 53.0f}};
    for (size_t m = 0; m < 2; ++m) {
        for (size_t n = 0; n < 2; ++n) {
            const uint64_t logical = kCBase + m * 32 + n * sizeof(float);
            const float actual = load<float>(memory, RVTNG::swizzle_addr(logical, swizzle));
            expect_close("swizzled matrix address", actual, expected[m][n]);
        }
    }
}

void test_unsupported_type() {
    std::vector<uint8_t> memory(kUramSize, 0);
    RVTNG::mat_tcsr_t control{};
    control.dataType = 0xff;
    assert(!RVTNG::execute_matmul(memory.data(), control,
                                 kABase, kBBase, kCBase, 0, 0, false));
}

} // namespace

int main() {
    test_instruction_decoder();
    test_scalar_decoders();
    test_tf32();
    test_bf16_and_fp16();
    test_fp8();
    test_mxfp8();
    test_int8_encodings();
    test_fp4_and_int4();
    test_mxfp4_scale_layout();
    test_layout_mma_and_k_blocking();
    test_swizzled_addresses();
    test_unsupported_type();
    std::cout << "MAT datatype tests passed\n";
    return 0;
}
