#ifdef NDEBUG
#undef NDEBUG
#endif

#include "fputils.hpp"
#include "mat.hpp"
#include "dayan_decoder.hpp"
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

constexpr uint16_t paired_type(uint8_t aBase, uint8_t bBase,
                               bool aMX = false, bool bMX = false) {
    return DAYAN_MAT_PAIRED_DATATYPE(
        DAYAN_MAT_OPERAND(aBase, aMX),
        DAYAN_MAT_OPERAND(bBase, bMX));
}

template<typename T>
void store(std::vector<uint8_t>& memory, uint64_t address, T value) {
    std::memcpy(memory.data() + address, &value, sizeof(value));
}

template<typename T>
void store_swizzled(std::vector<uint8_t>& memory, uint64_t address, uint64_t swizzle, T value) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
    for (size_t index = 0; index < sizeof(value); ++index) {
        memory[DAYAN::swizzle_addr(address + index, swizzle)] = bytes[index];
    }
}

template<typename T>
T load(const std::vector<uint8_t>& memory, uint64_t address) {
    T value;
    std::memcpy(&value, memory.data() + address, sizeof(value));
    return value;
}

template<typename T>
T load_swizzled(const std::vector<uint8_t>& memory, uint64_t address, uint64_t swizzle) {
    T value;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
    for (size_t index = 0; index < sizeof(value); ++index) {
        bytes[index] = memory[DAYAN::swizzle_addr(address + index, swizzle)];
    }
    return value;
}

void copy_to(std::vector<uint8_t>& memory, uint64_t address,
             const std::vector<uint8_t>& values) {
    std::copy(values.begin(), values.end(), memory.begin() + address);
}

void copy_to_swizzled(std::vector<uint8_t>& memory, uint64_t address,
                      uint64_t swizzle, const std::vector<uint8_t>& values) {
    for (size_t index = 0; index < values.size(); ++index) {
        memory[DAYAN::swizzle_addr(address + index, swizzle)] = values[index];
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

std::vector<uint8_t> pack_u32(std::initializer_list<uint32_t> values) {
    std::vector<uint8_t> packed(values.size() * sizeof(uint32_t));
    size_t index = 0;
    for (const uint32_t value : values) {
        std::memcpy(packed.data() + index * sizeof(uint32_t), &value, sizeof(value));
        ++index;
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

    DAYAN::mat_tcsr_t control{};
    control.k = k - 1;
    control.aStride = align32(aPayload.size());
    control.bStride = align32(bPayload.size());
    control.cStride = 32;
    control.dataType = dataType;
    control.aScaleStride = 32;
    control.bScaleStride = 32;

    assert(DAYAN::execute_matmul(memory.data(), control,
                                kABase, kBBase, kCBase,
                                kAScaleBase, kBScaleBase, acc));
    return load<float>(memory, kCBase);
}

void test_instruction_decoder() {
    constexpr uint32_t mm = 0x2000000bu | (7u << 7) | (3u << 15) |
        (5u << 20) | (6u << 25);
    constexpr uint32_t mma = mm | (1u << 12);
    constexpr uint32_t conv2d = mm | (2u << 12);
    constexpr uint32_t conv2da = mm | (3u << 12);
    constexpr uint32_t setptr = 0x2000400bu | (3u << 15) |
        (5u << 20) | (6u << 25);

    static_assert(DAYAN_TYPE_S4 == 0x18);
    static_assert(DAYAN_TYPE_U4 == 0x19);
    static_assert(paired_type(DAYAN_MAT_TYPE_BF16, DAYAN_MAT_TYPE_FP16)
                  == 0x8452);
    assert(IS_RVT_MAT_MM(mm));
    assert(IS_RVT_MAT_MMA(mma));
    assert(!IS_RVT_MAT_MMA(mm));
    assert(IS_RVT_MAT_CONV2D(conv2d));
    assert(IS_RVT_MAT_CONV2DA(conv2da));
    assert(!IS_RVT_MAT_MM_FAMILY(conv2d));
    assert(IS_RVT_MAT_MM_FAMILY(mm | (1u << 28)));
    assert(IS_RVT_TOP_MAT(mm | (1u << 28)));
    assert(!IS_RVT_TOP_VEC(mm | (1u << 28)));
    assert(IS_RVT_MAT_SETPTR(setptr));
    assert(!IS_RVT_MAT_MM_FAMILY(setptr));
    assert(IS_RVT_MAT_SETPTR(setptr | (1u << 7)));
    assert(DAYAN_MAT_GROUP(setptr) == 6);
    assert(dayan::is_mat(mm));
    assert(dayan::is_mat(conv2d));
    assert(dayan::is_mat(setptr));
}

void test_bf16_conv2d() {
    std::vector<uint8_t> memory(kUramSize, 0);
    copy_to(memory, kABase, pack_u16({0x3f80, 0x4000}));
    copy_to(memory, kABase + 32, pack_u16({0x4040, 0x4080}));
    copy_to(memory, kBBase, pack_u16({0x40a0, 0x40c0}));
    copy_to(memory, kBBase + 32, pack_u16({0x40e0, 0x4100}));

    DAYAN::conv_tcsr_t control{};
    control.c = 2;
    control.wi = 2;
    control.hi = 1;
    control.wo = 2;
    control.ho = 1;
    control.wk = 1;
    control.hk = 1;
    control.co = 2;
    control.wiStride = 32;
    control.hiStride = 64;
    control.wkStride = 32;
    control.hkStride = 32;
    control.coStride = 32;
    control.woStride = 32;
    control.hoStride = 64;
    control.dataType = DAYAN_TYPE_BF16;
    control.dilationW = 1;
    control.dilationH = 1;
    control.strideW = 1;
    control.strideH = 1;

    assert(DAYAN::execute_conv2d(memory.data(), control, kABase, kBBase, kCBase, false));
    const float expected[2][2] = {{17.0f, 23.0f}, {39.0f, 53.0f}};
    for (size_t w = 0; w < 2; ++w) {
        for (size_t co = 0; co < 2; ++co) {
            expect_close("Conv2D", load<float>(memory, kCBase + w * 32 + co * sizeof(float)), expected[w][co]);
        }
    }

    assert(DAYAN::execute_conv2d(memory.data(), control, kABase, kBBase, kCBase, true));
    for (size_t w = 0; w < 2; ++w) {
        for (size_t co = 0; co < 2; ++co) {
            expect_close("Conv2DA", load<float>(memory, kCBase + w * 32 + co * sizeof(float)), 2.0f * expected[w][co]);
        }
    }
}

void test_bf16_conv2d_layout() {
    constexpr uint64_t inputSwizzle = 5 | (1 << 8) | (6 << 16);
    constexpr uint64_t kernelSwizzle = 6 | (1 << 8) | (7 << 16);
    constexpr uint64_t outputSwizzle = 5 | (1 << 8) | (8 << 16);
    constexpr uint64_t hiCount = 3;
    constexpr uint64_t wiCount = 4;
    constexpr uint64_t ciCount = 3;
    constexpr uint64_t hkCount = 2;
    constexpr uint64_t wkCount = 2;
    constexpr uint64_t coCount = 2;
    constexpr uint64_t hoCount = 3;
    constexpr uint64_t woCount = 3;

    std::vector<uint8_t> memory(kUramSize, 0);
    std::vector<float> inputValues(hiCount * wiCount * ciCount);
    std::vector<float> kernelValues(hkCount * wkCount * coCount * ciCount);
    for (uint64_t hi = 0; hi < hiCount; ++hi) {
        for (uint64_t wi = 0; wi < wiCount; ++wi) {
            for (uint64_t ci = 0; ci < ciCount; ++ci) {
                const size_t index = (hi * wiCount + wi) * ciCount + ci;
                const uint16_t value = FP32ToBF16(static_cast<float>(1 + index) / 8.0f);
                inputValues[index] = BF16ToFP32(value);
                store_swizzled(memory, kABase + hi * 96 + wi * 16 + ci * sizeof(uint16_t), inputSwizzle, value);
            }
        }
    }
    for (uint64_t hk = 0; hk < hkCount; ++hk) {
        for (uint64_t wk = 0; wk < wkCount; ++wk) {
            for (uint64_t co = 0; co < coCount; ++co) {
                for (uint64_t ci = 0; ci < ciCount; ++ci) {
                    const size_t index = ((hk * wkCount + wk) * coCount + co) * ciCount + ci;
                    const uint16_t value = FP32ToBF16(static_cast<float>(1 + index) / 16.0f);
                    kernelValues[index] = BF16ToFP32(value);
                    store_swizzled(memory, kBBase + hk * 160 + wk * 64 + co * 16 + ci * sizeof(uint16_t),
                                   kernelSwizzle, value);
                }
            }
        }
    }

    DAYAN::conv_tcsr_t control{};
    control.c = ciCount;
    control.wi = wiCount;
    control.hi = hiCount;
    control.wo = woCount;
    control.ho = hoCount;
    control.wk = wkCount;
    control.hk = hkCount;
    control.co = coCount;
    control.wiStride = 16;
    control.hiStride = 96;
    control.wkStride = 64;
    control.hkStride = 160;
    control.coStride = 16;
    control.woStride = 32;
    control.hoStride = 128;
    control.iSwizzleMBS = inputSwizzle;
    control.kSwizzleMBS = kernelSwizzle;
    control.oSwizzleMBS = outputSwizzle;
    control.dataType = DAYAN_TYPE_BF16;
    control.padL = 1;
    control.padU = 1;
    control.dilationW = 1;
    control.dilationH = 1;
    control.strideW = 2;
    control.strideH = 1;

    assert(DAYAN::execute_conv2d(memory.data(), control, kABase, kBBase, kCBase, false));
    for (uint64_t ho = 0; ho < hoCount; ++ho) {
        for (uint64_t wo = 0; wo < woCount; ++wo) {
            for (uint64_t co = 0; co < coCount; ++co) {
                float expected = 0.0f;
                for (uint64_t hk = 0; hk < hkCount; ++hk) {
                    const int64_t hi = static_cast<int64_t>(ho + hk) - 1;
                    if (hi < 0 || static_cast<uint64_t>(hi) >= hiCount) {
                        continue;
                    }
                    for (uint64_t wk = 0; wk < wkCount; ++wk) {
                        const int64_t wi = static_cast<int64_t>(wo * 2 + wk) - 1;
                        if (wi < 0 || static_cast<uint64_t>(wi) >= wiCount) {
                            continue;
                        }
                        for (uint64_t ci = 0; ci < ciCount; ++ci) {
                            const size_t inputIndex =
                                (static_cast<uint64_t>(hi) * wiCount + static_cast<uint64_t>(wi)) * ciCount + ci;
                            const size_t kernelIndex = ((hk * wkCount + wk) * coCount + co) * ciCount + ci;
                            expected += inputValues[inputIndex] * kernelValues[kernelIndex];
                        }
                    }
                }
                const uint64_t address = kCBase + ho * 128 + wo * 32 + co * sizeof(float);
                const float actual = load_swizzled<float>(memory, address, outputSwizzle);
                expect_close("Conv2D layout", actual, expected);
            }
        }
    }
}

void test_paired_scaled_conv2d() {
    constexpr uint64_t channels = 33;
    std::vector<uint8_t> memory(kUramSize, 0);
    copy_to(memory, kABase, std::vector<uint8_t>(channels, 0x38));
    copy_to(memory, kBBase, std::vector<uint8_t>(channels, 1));
    copy_to(memory, kAScaleBase, {0x7f, 0x80});

    DAYAN::conv_tcsr_t control{};
    control.c = channels;
    control.wi = 1;
    control.hi = 1;
    control.wo = 1;
    control.ho = 1;
    control.wk = 1;
    control.hk = 1;
    control.co = 1;
    control.wiStride = 64;
    control.hiStride = 64;
    control.wkStride = 64;
    control.hkStride = 64;
    control.coStride = 64;
    control.woStride = 32;
    control.hoStride = 32;
    control.scaleWiStride = 32;
    control.scaleHiStride = 32;
    control.scaleWkStride = 32;
    control.scaleHkStride = 32;
    control.scaleCoStride = 32;
    control.dataType = paired_type(
        DAYAN_MAT_TYPE_E4M3, DAYAN_MAT_TYPE_S8, true, false);
    control.dilationW = 1;
    control.dilationH = 1;
    control.strideW = 1;
    control.strideH = 1;

    assert(DAYAN::execute_conv2d(
        memory.data(), control, kABase, kBBase, kCBase,
        kAScaleBase, kBScaleBase, false));
    expect_close("paired scaled Conv2D", load<float>(memory, kCBase), 34.0f);
}

void test_scalar_decoders() {
    expect_close("FP16 smallest subnormal", FP16ToFP32(0x0001),
                 std::ldexp(1.0f, -24), 0.0f);
    expect_close("FP8 E0M7 fraction", FP8ToFP32<0>(0x60), 0.75f, 0.0f);
    expect_close("FP8 E0M7 minimum", FP8ToFP32<0>(0x01),
                 std::ldexp(1.0f, -7), 0.0f);
    expect_close("FP8 E1M6 subnormal", FP8ToFP32<1>(0x01),
                 std::ldexp(1.0f, -5), 0.0f);
    expect_close("FP8 E2M5 subnormal", FP8ToFP32<2>(0x01),
                 std::ldexp(1.0f, -5), 0.0f);
    expect_close("FP8 E3M4 subnormal", FP8ToFP32<3>(0x01),
                 std::ldexp(1.0f, -6), 0.0f);
    expect_close("FP8 E4M3 subnormal", FP8ToFP32<4>(0x01),
                 std::ldexp(1.0f, -9), 0.0f);
    expect_close("FP8 E5M2 subnormal", FP8ToFP32<5>(0x01),
                 std::ldexp(1.0f, -16), 0.0f);
    expect_close("FP8 E6M1 subnormal", FP8ToFP32<6>(0x01),
                 std::ldexp(1.0f, -31), 0.0f);
    expect_close("FP8 E6M1 maximum finite", FP8ToFP32<6>(0x7f),
                 std::ldexp(3.0f, 31), 0.0f);
    expect_close("FP4 E2M1 maximum", FP4E2M1ToFP32(0x07), 6.0f, 0.0f);
    expect_close("FP4 E2M1 negative", FP4E2M1ToFP32(0x0d), -3.0f, 0.0f);
    assert(E8M0ToExponent(0x7f) == 0);
    assert(E8M0ToExponent(0x80) == 1);
}

void test_bf16_and_fp16() {
    const auto bf16A = pack_u16({0x3fc0, 0xc000}); // 1.5, -2
    const auto bf16B = pack_u16({0x4000, 0x4040}); // 2, 3
    expect_close("BF16", run_one(DAYAN_TYPE_BF16, 2, bf16A, bf16B), -3.0f);

    const auto fp16A = pack_u16({0x3e00, 0xc000}); // 1.5, -2
    const auto fp16B = pack_u16({0x4000, 0x4200}); // 2, 3
    expect_close("FP16", run_one(DAYAN_TYPE_FP16, 2, fp16A, fp16B), -3.0f);
}

void test_paired_16bit_formats() {
    const uint16_t dataType = paired_type(
        DAYAN_MAT_TYPE_BF16, DAYAN_MAT_TYPE_FP16);
    expect_close("paired BF16 x FP16",
                 run_one(dataType, 2,
                         pack_u16({0x3fc0, 0xc000}),
                         pack_u16({0x4000, 0x4200})),
                 -3.0f);

    // FP16 fraction bits [2:0] are discarded before the common FP32 kernel.
    expect_close("FP16 low fraction truncation",
                 run_one(paired_type(DAYAN_MAT_TYPE_FP16,
                                     DAYAN_MAT_TYPE_BF16),
                         1, pack_u16({0x3c07}), pack_u16({0x3f80})),
                 1.0f);
}

void test_tf32() {
    const uint32_t conversions[][2] = {
        {0x00000000, 0x00000000}, {0x80000000, 0x80000000},
        {0x3f801fff, 0x3f800000}, {0xbf801fff, 0xbf800000},
        {0x3f802000, 0x3f802000}, {0x3fffffff, 0x3fffe000},
        {0x00001fff, 0x00000000}, {0x00002000, 0x00002000},
        {0x80001fff, 0x80000000}, {0x007fffff, 0x007fe000},
        {0x7f7fffff, 0x7f7fe000}, {0x7f800000, 0x7f800000},
        {0xff800000, 0xff800000}, {0x7f800001, 0x7fc00000},
    };
    for (const auto& entry : conversions) {
        const float actual = TF32Clip(std::bit_cast<float>(entry[0]));
        assert(std::bit_cast<uint32_t>(actual) == entry[1]);
    }

    const auto one = pack_u32({0x3f800000});
    {
        constexpr uint16_t type = paired_type(DAYAN_MAT_TYPE_TF32, DAYAN_MAT_TYPE_TF32);
        expect_close("TF32 truncates low fraction", run_one(type, 1, pack_u32({0x3f801fff}), one), 1.0f, 0.0f);
        expect_close("TF32 retains ten fraction bits", run_one(type, 1, pack_u32({0x3f802000}), one), 1.0f + 1.0f / 1024, 0.0f);
        expect_close("TF32 negative truncation", run_one(type, 1, pack_u32({0xbf801fff}), one), -1.0f, 0.0f);
        expect_close("TF32 full exponent", run_one(type, 1, pack_u32({0x00800000}), pack_u32({0x7d800000})), 0.25f, 0.0f);
        const float initialC = 1.0f / (1 << 20);
        expect_close("TF32 FP32 accumulation", run_one(type, 1, one, one, {}, {}, initialC, true), 1.0f + initialC, 0.0f);
        const float inf = run_one(type, 1, pack_u32({0x7f800000}), one);
        assert(std::bit_cast<uint32_t>(inf) == 0x7f800000);
        const float nan = run_one(type, 1, pack_u32({0x7f800001}), one);
        assert((std::bit_cast<uint32_t>(nan) & 0x7fffffffu) > 0x7f800000u);
    }

    constexpr uint64_t swizzle = (uint64_t{7} << 16) | (uint64_t{2} << 8) | 3;
    for (const uint8_t aType : {DAYAN_MAT_TYPE_TF32, DAYAN_MAT_TYPE_BF16, DAYAN_MAT_TYPE_FP16}) {
        for (const uint8_t bType : {DAYAN_MAT_TYPE_TF32, DAYAN_MAT_TYPE_BF16, DAYAN_MAT_TYPE_FP16}) {
            if (aType != DAYAN_MAT_TYPE_TF32 && bType != DAYAN_MAT_TYPE_TF32) {
                continue;
            }
            for (const uint64_t k : {127, 128, 129}) {
                DAYAN::mat_tcsr_t control{};
                control.m = 2;
                control.n = 4;
                control.k = k - 1;
                control.aStride = align32(k * (aType == DAYAN_MAT_TYPE_TF32 ? 4 : 2)) + 32;
                control.bStride = align32(k * (bType == DAYAN_MAT_TYPE_TF32 ? 4 : 2)) + 64;
                control.cStride = 64;
                control.aSwizzleMBS = swizzle;
                control.bSwizzleMBS = swizzle;
                control.cSwizzleMBS = swizzle;
                control.dataType = paired_type(aType, bType);
                std::vector<uint8_t> memory(kUramSize, 0xa5);
                const auto write = [&](uint64_t base, uint64_t stride, uint8_t type, uint64_t rows, int sign) {
                    for (uint64_t row = 0; row < rows; ++row) {
                        for (uint64_t inner = 0; inner < k; ++inner) {
                            const float value = sign * float(int((row * 3 + inner) % 13) - 6) / 16;
                            if (type == DAYAN_MAT_TYPE_TF32) {
                                store_swizzled(memory, base + row * stride + inner * 4, swizzle, value);
                            } else {
                                const uint16_t raw = type == DAYAN_MAT_TYPE_BF16 ? FP32ToBF16(value) : FP32ToFP16(value);
                                store_swizzled(memory, base + row * stride + inner * 2, swizzle, raw);
                            }
                        }
                    }
                };
                write(kABase, control.aStride, aType, 3, 1);
                write(kBBase, control.bStride, bType, 5, -1);
                for (const bool acc : {false, true}) {
                    assert(DAYAN::execute_matmul(memory.data(), control, kABase, kBBase, kCBase, 0, 0, acc));
                    for (uint64_t m = 0; m < 3; ++m) {
                        for (uint64_t n = 0; n < 5; ++n) {
                            float expected = 0;
                            for (uint64_t inner = 0; inner < k; ++inner) {
                                expected += -float(int((m * 3 + inner) % 13) - 6) * float(int((n * 3 + inner) % 13) - 6) / 256;
                            }
                            expect_close("TF32 mixed strided swizzled MM/MMA", load_swizzled<float>(memory, kCBase + m * control.cStride + n * 4, swizzle), expected * (acc ? 2 : 1), 0.0f);
                        }
                        assert(load_swizzled<uint32_t>(memory, kCBase + m * control.cStride + 5 * 4, swizzle) == 0xa5a5a5a5u);
                    }
                }
            }
        }
    }
}

void test_fp8() {
    expect_close("FP8 E4M3",
                 run_one(DAYAN_TYPE_FP8_E4M3, 2,
                         {0x3c, 0xc0}, {0x40, 0x44}),
                 -3.0f);
    expect_close("FP8 E5M2",
                 run_one(DAYAN_TYPE_FP8_E5M2, 2,
                         {0x3e, 0xc0}, {0x40, 0x42}),
                 -3.0f);
}

void test_paired_8bit_formats() {
    struct format_case_t {
        uint8_t base;
        uint8_t sample;
        float value;
        std::string_view name;
    };
    const format_case_t formats[] = {
        {DAYAN_MAT_TYPE_E0M7, 0x60, 0.75f, "E0M7"},
        {DAYAN_MAT_TYPE_E1M6, 0x30, 1.5f, "E1M6"},
        {DAYAN_MAT_TYPE_E2M5, 0x30, 1.5f, "E2M5"},
        {DAYAN_MAT_TYPE_E3M4, 0x38, 1.5f, "E3M4"},
        {DAYAN_MAT_TYPE_E4M3, 0x3c, 1.5f, "E4M3"},
        {DAYAN_MAT_TYPE_E5M2, 0x3e, 1.5f, "E5M2"},
        {DAYAN_MAT_TYPE_E6M1, 0x3f, 1.5f, "E6M1"},
    };
    for(const format_case_t& format : formats) {
        expect_close(format.name,
                     run_one(paired_type(format.base, DAYAN_MAT_TYPE_S8),
                             1, {format.sample}, {2}),
                     2.0f * format.value);
        expect_close(format.name,
                     run_one(paired_type(DAYAN_MAT_TYPE_S8, format.base),
                             1, {2}, {format.sample}),
                     2.0f * format.value);

        constexpr size_t k = 33;
        expect_close(format.name,
                     run_one(paired_type(format.base, DAYAN_MAT_TYPE_S8,
                                         true, false),
                             k, std::vector<uint8_t>(k, format.sample),
                             std::vector<uint8_t>(k, 1), {0x7f, 0x80}),
                     34.0f * format.value);
        expect_close(format.name,
                     run_one(paired_type(DAYAN_MAT_TYPE_S8, format.base,
                                         false, true),
                             k, std::vector<uint8_t>(k, 1),
                             std::vector<uint8_t>(k, format.sample),
                             {}, {0x7f, 0x80}),
                     34.0f * format.value);
    }

    expect_close("paired E0M7 x E6M1",
                 run_one(paired_type(DAYAN_MAT_TYPE_E0M7,
                                     DAYAN_MAT_TYPE_E6M1),
                         1, {0x60}, {0x3f}),
                 1.125f);
    expect_close("paired U8 x E5M2",
                 run_one(paired_type(DAYAN_MAT_TYPE_U8,
                                     DAYAN_MAT_TYPE_E5M2),
                         2, {3, 4}, {0x40, 0xc0}),
                 -2.0f);
}

void test_mxfp8() {
    constexpr size_t k = 33;
    for (const auto& [dataType, one] : {
             std::pair<uint64_t, uint8_t>{DAYAN_TYPE_MX_FP8_E4M3, 0x38},
             std::pair<uint64_t, uint8_t>{DAYAN_TYPE_MX_FP8_E5M2, 0x3c}}) {
        const std::vector<uint8_t> payload(k, one);
        expect_close(dataType == DAYAN_TYPE_MX_FP8_E4M3
                         ? "MXFP8 E4M3" : "MXFP8 E5M2",
                     run_one(dataType, k, payload, payload,
                             {0x7f, 0x80}, {0x7f, 0x7f}),
                     34.0f);
    }
}

void test_independent_mx_operands() {
    constexpr size_t k = 33;
    const uint16_t dataType = paired_type(
        DAYAN_MAT_TYPE_E4M3, DAYAN_MAT_TYPE_S8, true, false);
    expect_close("paired one-sided MX scale",
                 run_one(dataType, k,
                         std::vector<uint8_t>(k, 0x38),
                         std::vector<uint8_t>(k, 1),
                         {0x7f, 0x80}),
                 34.0f);
}

void test_int8_encodings() {
    const std::vector<uint8_t> a = {
        static_cast<uint8_t>(-3), static_cast<uint8_t>(4)};
    const std::vector<uint8_t> b = {
        static_cast<uint8_t>(5), static_cast<uint8_t>(-2)};
    expect_close("S8 legacy encoding",
                 run_one(DAYAN_TYPE_S8, 2, a, b), -23.0f);
    expect_close("U8 legacy encoding",
                 run_one(DAYAN_TYPE_U8, 1, {250}, {2}), 500.0f);
}

void test_fp4_and_int4() {
    expect_close("FP4 E2M1",
                 run_one(DAYAN_TYPE_FP4_E2M1, 2,
                         pack_nibbles({0x3, 0xc}),
                         pack_nibbles({0x4, 0x5})),
                 -3.0f);
    expect_close("INT4 low/high nibble order",
                 run_one(DAYAN_TYPE_S4, 2,
                         pack_nibbles({0xd, 0x4}),
                         pack_nibbles({0x5, 0xe})),
                 -23.0f);
    expect_close("U4 low/high nibble order",
                 run_one(DAYAN_TYPE_U4, 2,
                         pack_nibbles({0x3, 0x4}),
                         pack_nibbles({0x5, 0x2})),
                 23.0f);
}

void test_paired_4bit_formats() {
    expect_close("paired E2M1 x U4",
                 run_one(paired_type(DAYAN_MAT_TYPE_FP4_E2M1,
                                     DAYAN_MAT_TYPE_U4),
                         2,
                         pack_nibbles({0x3, 0xc}),
                         pack_nibbles({0x2, 0x3})),
                 -3.0f);
    expect_close("paired S4 x E2M1",
                 run_one(paired_type(DAYAN_MAT_TYPE_S4,
                                     DAYAN_MAT_TYPE_FP4_E2M1),
                         2,
                         pack_nibbles({0xd, 0x4}),
                         pack_nibbles({0x4, 0x5})),
                 6.0f);
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

    DAYAN::mat_tcsr_t control{};
    control.k = k - 1;
    control.m = 1;
    control.n = 1;
    control.aStride = 32;
    control.bStride = 32;
    control.cStride = 32;
    control.dataType = DAYAN_TYPE_MX_FP4_E2M1;
    control.aScaleStride = 32;
    control.bScaleStride = 32;

    assert(DAYAN::execute_matmul(memory.data(), control,
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

    DAYAN::mat_tcsr_t control{};
    control.k = 1;
    control.m = 1;
    control.n = 1;
    control.aStride = 32;
    control.bStride = 32;
    control.cStride = 32;
    control.dataType = DAYAN_TYPE_BF16;

    assert(DAYAN::execute_matmul(memory.data(), control,
                                kABase, kBBase, kCBase, 0, 0, false));
    const float expected[2][2] = {{17.0f, 23.0f}, {39.0f, 53.0f}};
    for (size_t m = 0; m < 2; ++m) {
        for (size_t n = 0; n < 2; ++n) {
            expect_close("matrix layout",
                         load<float>(memory, kCBase + m * 32 + n * 4),
                         expected[m][n]);
        }
    }

    assert(DAYAN::execute_matmul(memory.data(), control,
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
                 run_one(DAYAN_TYPE_S8, 129, ones, ones), 129.0f);
}

void test_swizzled_addresses() {
    constexpr uint64_t swizzle = (uint64_t{8} << 16) | (uint64_t{2} << 8) | 5;
    std::vector<uint8_t> memory(kUramSize, 0);
    copy_to_swizzled(memory, kABase, swizzle, pack_u16({0x3f80, 0x4000}));
    copy_to_swizzled(memory, kABase + 32, swizzle, pack_u16({0x4040, 0x4080}));
    copy_to_swizzled(memory, kBBase, swizzle, pack_u16({0x40a0, 0x40c0}));
    copy_to_swizzled(memory, kBBase + 32, swizzle, pack_u16({0x40e0, 0x4100}));

    DAYAN::mat_tcsr_t control{};
    control.k = 1;
    control.m = 1;
    control.n = 1;
    control.aStride = 32;
    control.bStride = 32;
    control.cStride = 32;
    control.aSwizzleMBS = swizzle;
    control.bSwizzleMBS = swizzle;
    control.cSwizzleMBS = swizzle;
    control.dataType = DAYAN_TYPE_BF16;

    assert(DAYAN::execute_matmul(memory.data(), control,
                                kABase, kBBase, kCBase, 0, 0, false));
    const float expected[2][2] = {{17.0f, 23.0f}, {39.0f, 53.0f}};
    for (size_t m = 0; m < 2; ++m) {
        for (size_t n = 0; n < 2; ++n) {
            const uint64_t logical = kCBase + m * 32 + n * sizeof(float);
            const float actual = load<float>(memory, DAYAN::swizzle_addr(logical, swizzle));
            expect_close("swizzled matrix address", actual, expected[m][n]);
        }
    }
}

void test_unsupported_type() {
    std::vector<uint8_t> memory(kUramSize, 0);
    DAYAN::mat_tcsr_t control{};
    control.aStride = 32;
    control.bStride = 32;
    control.cStride = 32;
    const auto rejected = [&](uint64_t dataType) {
        control.dataType = dataType;
        store(memory, kCBase, 123.0f);
        assert(!DAYAN::execute_matmul(memory.data(), control,
                                     kABase, kBBase, kCBase,
                                     kAScaleBase, kBScaleBase, false));
        expect_close("invalid datatype leaves C unchanged",
                     load<float>(memory, kCBase), 123.0f);
    };

    rejected(0xff);
    rejected(DAYAN_TYPE_FP32);
    rejected(paired_type(DAYAN_MAT_TYPE_TF32, DAYAN_MAT_TYPE_S8));
    rejected(paired_type(DAYAN_MAT_TYPE_TF32, DAYAN_MAT_TYPE_BF16, true, false));
    rejected(DAYAN_TYPE_RTL_S8);
    rejected(uint64_t{0x100} | DAYAN_TYPE_BF16);
    rejected(paired_type(DAYAN_MAT_TYPE_BF16, DAYAN_MAT_TYPE_E4M3));
    rejected(paired_type(DAYAN_MAT_TYPE_BF16, DAYAN_MAT_TYPE_FP16) | 0x1000);
    rejected(paired_type(DAYAN_MAT_TYPE_BF16, DAYAN_MAT_TYPE_FP16,
                         true, false));
    rejected(paired_type(0x07u, DAYAN_MAT_TYPE_E4M3)); // E7M0 is reserved
    rejected(paired_type(DAYAN_MAT_TYPE_TF16, DAYAN_MAT_TYPE_S8));
}

} // namespace

int main() {
    test_instruction_decoder();
    test_scalar_decoders();
    test_bf16_and_fp16();
    test_paired_16bit_formats();
    test_tf32();
    test_bf16_conv2d();
    test_bf16_conv2d_layout();
    test_paired_scaled_conv2d();
    test_fp8();
    test_paired_8bit_formats();
    test_mxfp8();
    test_independent_mx_operands();
    test_int8_encodings();
    test_fp4_and_int4();
    test_paired_4bit_formats();
    test_mxfp4_scale_layout();
    test_layout_mma_and_k_blocking();
    test_swizzled_addresses();
    test_unsupported_type();
    std::cout << "MAT datatype tests passed\n";
    return 0;
}
