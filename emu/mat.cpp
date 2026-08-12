#include "utils.hpp"
#include "common.hpp"
#include "mat.hpp"
#include "fputils.hpp"
#include "swizzle.hpp"
#include "rvt_decoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

namespace RVTNG {
namespace {

constexpr uint64_t kMatmulBlockK = 128;
constexpr uint64_t kMXScaleGroupK = 32;
constexpr uint64_t kUramAddressMask = (uint64_t{1} << 21) - 1;

enum class operand_kind_t {
    tf32,
    bf16,
    fp16,
    fp8_e4m3,
    fp8_e5m2,
    fp4_e2m1,
    sint8,
    uint8,
    sint4,
};

struct mat_format_t {
    operand_kind_t kind;
    unsigned bitsPerElement;
    bool usesScale;
    const char* name;
};

static std::optional<mat_format_t> decode_mat_format(uint64_t dataType) {
    switch (dataType) {
    case RVT_TYPE_FP32:
        return mat_format_t{operand_kind_t::tf32, 32, false, "TF32"};
    case RVT_TYPE_BF16:
        return mat_format_t{operand_kind_t::bf16, 16, false, "BF16"};
    case RVT_TYPE_FP16:
        return mat_format_t{operand_kind_t::fp16, 16, false, "FP16"};
    case RVT_TYPE_FP8_E4M3:
        return mat_format_t{operand_kind_t::fp8_e4m3, 8, false, "FP8-E4M3"};
    case RVT_TYPE_FP8_E5M2:
        return mat_format_t{operand_kind_t::fp8_e5m2, 8, false, "FP8-E5M2"};
    case RVT_TYPE_MX_FP8_E4M3:
        return mat_format_t{operand_kind_t::fp8_e4m3, 8, true, "MXFP8-E4M3"};
    case RVT_TYPE_MX_FP8_E5M2:
        return mat_format_t{operand_kind_t::fp8_e5m2, 8, true, "MXFP8-E5M2"};
    case RVT_TYPE_FP4_E2M1:
        return mat_format_t{operand_kind_t::fp4_e2m1, 4, false, "FP4-E2M1"};
    case RVT_TYPE_MX_FP4_E2M1:
        return mat_format_t{operand_kind_t::fp4_e2m1, 4, true, "MXFP4-E2M1"};
    case RVT_TYPE_RTL_S8:
    case RVT_TYPE_S8:
    case RVT_TYPE_COMPAT_S8:
        return mat_format_t{operand_kind_t::sint8, 8, false, "INT8"};
    case RVT_TYPE_RTL_U8:
    case RVT_TYPE_U8:
    case RVT_TYPE_COMPAT_U8:
        return mat_format_t{operand_kind_t::uint8, 8, false, "UINT8"};
    case RVT_TYPE_S4:
        return mat_format_t{operand_kind_t::sint4, 4, false, "INT4"};
    default:
        return std::nullopt;
    }
}

template<typename T>
static T load_uram(const uint8_t* uram, uint64_t logicalAddress, uint64_t swizzleMBS) {
    T value;
    const uint64_t physicalAddress = swizzle_addr(logicalAddress, swizzleMBS);
    std::memcpy(&value, uram + physicalAddress, sizeof(value));
    return value;
}

template<typename T>
static void store_uram(uint8_t* uram, uint64_t logicalAddress, uint64_t swizzleMBS,
                       const T& value) {
    const uint64_t physicalAddress = swizzle_addr(logicalAddress, swizzleMBS);
    std::memcpy(uram + physicalAddress, &value, sizeof(value));
}

static uint64_t element_byte_offset(uint64_t index, unsigned bitsPerElement) {
    return bitsPerElement == 4 ? index / 2 : index * (bitsPerElement / 8);
}

static float decode_operand(const uint8_t* uram, uint64_t rowBase, uint64_t index,
                            uint64_t swizzleMBS, const mat_format_t& format) {
    const uint64_t address = rowBase + element_byte_offset(index, format.bitsPerElement);
    switch (format.kind) {
    case operand_kind_t::tf32:
        return TF32Clip(load_uram<float>(uram, address, swizzleMBS));
    case operand_kind_t::bf16:
        return BF16ToFP32(load_uram<uint16_t>(uram, address, swizzleMBS));
    case operand_kind_t::fp16:
        return FP16ToFP32(load_uram<uint16_t>(uram, address, swizzleMBS));
    case operand_kind_t::fp8_e4m3:
        return FP8ToFP32<4>(load_uram<uint8_t>(uram, address, swizzleMBS));
    case operand_kind_t::fp8_e5m2:
        return FP8ToFP32<5>(load_uram<uint8_t>(uram, address, swizzleMBS));
    case operand_kind_t::fp4_e2m1: {
        const uint8_t packed = load_uram<uint8_t>(uram, address, swizzleMBS);
        const uint8_t value = (index & 1) == 0 ? packed & 0x0fu : packed >> 4;
        return FP4E2M1ToFP32(value);
    }
    case operand_kind_t::sint8: {
        const int raw = load_uram<uint8_t>(uram, address, swizzleMBS);
        return static_cast<float>(raw < 0x80 ? raw : raw - 0x100);
    }
    case operand_kind_t::uint8:
        return static_cast<float>(load_uram<uint8_t>(uram, address, swizzleMBS));
    case operand_kind_t::sint4: {
        const uint8_t packed = load_uram<uint8_t>(uram, address, swizzleMBS);
        const int raw = (index & 1) == 0 ? packed & 0x0f : packed >> 4;
        return static_cast<float>(raw < 8 ? raw : raw - 16);
    }
    }
    return 0.0f;
}

static void fetch_row(
    const uint8_t* uram,
    uint64_t rowBase,
    uint64_t swizzleMBS,
    uint64_t firstK,
    uint64_t count,
    const mat_format_t& format,
    uint64_t scaleBase,
    uint64_t scaleStride,
    uint64_t majorIndex,
    std::array<float, kMatmulBlockK>& values,
    std::array<int, kMatmulBlockK>& scaleExponents) {
    uint64_t cachedScaleGroup = std::numeric_limits<uint64_t>::max();
    int cachedScaleExponent = 0;
    const uint64_t scaleRowBase = scaleBase + majorIndex * scaleStride;

    for (uint64_t subK = 0; subK < kMatmulBlockK; ++subK) {
        if (subK >= count) {
            values[subK] = 0.0f;
            scaleExponents[subK] = 0;
            continue;
        }

        const uint64_t logicalK = firstK + subK;
        values[subK] = decode_operand(uram, rowBase, logicalK, swizzleMBS, format);
        if (!format.usesScale) {
            scaleExponents[subK] = 0;
            continue;
        }

        const uint64_t scaleGroup = logicalK / kMXScaleGroupK;
        if (scaleGroup != cachedScaleGroup) {
            const uint8_t rawScale = load_uram<uint8_t>(uram, scaleRowBase + scaleGroup, 0);
            cachedScaleExponent = E8M0ToExponent(rawScale);
            cachedScaleGroup = scaleGroup;
        }
        scaleExponents[subK] = cachedScaleExponent;
    }
}

static float dot_product(
    float sum,
    const std::array<float, kMatmulBlockK>& a,
    const std::array<float, kMatmulBlockK>& b,
    const std::array<int, kMatmulBlockK>& aScale,
    const std::array<int, kMatmulBlockK>& bScale,
    uint64_t count,
    bool scaled) {
    if (!scaled) {
        #pragma GCC ivdep
        for (uint64_t i = 0; i < count; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    for (uint64_t i = 0; i < count; ++i) {
        sum += std::ldexp(a[i] * b[i], aScale[i] + bScale[i]);
    }
    return sum;
}

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("O3", "fast-math")))
#endif
static float dot_product_contiguous(
    float sum,
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint64_t count) {
    #pragma GCC ivdep
    for (uint64_t i = 0; i < count; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

static std::vector<float> decode_conv_input(
    const uint8_t* uram,
    const conv_tcsr_t& tcsr,
    uint64_t input,
    const mat_format_t& format) {
    std::vector<float> values(tcsr.hi * tcsr.wi * tcsr.c);
    for (uint64_t hi = 0; hi < tcsr.hi; ++hi) {
        for (uint64_t wi = 0; wi < tcsr.wi; ++wi) {
            const uint64_t source = input + hi * tcsr.hiStride + wi * tcsr.wiStride;
            float* destination = values.data() + (hi * tcsr.wi + wi) * tcsr.c;
            for (uint64_t ci = 0; ci < tcsr.c; ++ci) {
                destination[ci] = decode_operand(uram, source, ci, tcsr.iSwizzleMBS, format);
            }
        }
    }
    return values;
}

static std::vector<float> decode_conv_kernel(
    const uint8_t* uram,
    const conv_tcsr_t& tcsr,
    uint64_t kernel,
    const mat_format_t& format) {
    std::vector<float> values(tcsr.hk * tcsr.wk * tcsr.co * tcsr.c);
    for (uint64_t hk = 0; hk < tcsr.hk; ++hk) {
        for (uint64_t wk = 0; wk < tcsr.wk; ++wk) {
            for (uint64_t co = 0; co < tcsr.co; ++co) {
                const uint64_t source = kernel + hk * tcsr.hkStride + wk * tcsr.wkStride + co * tcsr.coStride;
                float* destination = values.data() + ((hk * tcsr.wk + wk) * tcsr.co + co) * tcsr.c;
                for (uint64_t ci = 0; ci < tcsr.c; ++ci) {
                    destination[ci] = decode_operand(uram, source, ci, tcsr.kSwizzleMBS, format);
                }
            }
        }
    }
    return values;
}

} // namespace

bool execute_matmul(
    uint8_t* uram,
    const mat_tcsr_t& tcsr,
    uint64_t a,
    uint64_t b,
    uint64_t c,
    uint64_t aScale,
    uint64_t bScale,
    bool acc) {
    const auto format = decode_mat_format(tcsr.dataType);
    if (!format.has_value()) {
        return false;
    }

    const uint64_t M = tcsr.m + 1;
    const uint64_t N = tcsr.n + 1;
    const uint64_t K = tcsr.k + 1;
    debugf("MAT: Calculating %s M%lu N%lu K%lu\n", format->name, M, N, K);

    std::array<float, kMatmulBlockK> aValues;
    std::array<float, kMatmulBlockK> bValues;
    std::array<int, kMatmulBlockK> aScaleExponents;
    std::array<int, kMatmulBlockK> bScaleExponents;

    for (uint64_t k = 0; k < K; k += kMatmulBlockK) {
        const uint64_t count = std::min(kMatmulBlockK, K - k);
        for (uint64_t m = 0; m < M; ++m) {
            fetch_row(uram, a + m * tcsr.aStride, tcsr.aSwizzleMBS, k, count,
                      *format, aScale, tcsr.aScaleStride, m,
                      aValues, aScaleExponents);
            for (uint64_t n = 0; n < N; ++n) {
                fetch_row(uram, b + n * tcsr.bStride, tcsr.bSwizzleMBS, k, count,
                          *format, bScale, tcsr.bScaleStride, n,
                          bValues, bScaleExponents);

                const uint64_t cAddress = c + m * tcsr.cStride + n * sizeof(float);
                float sum = (acc || k != 0)
                    ? load_uram<float>(uram, cAddress, tcsr.cSwizzleMBS)
                    : 0.0f;
                if (std::isnan(sum)) {
                    debugf("[ERROR] MAT: Encountered NaN in C input.\n");
                }
                sum = dot_product(sum, aValues, bValues, aScaleExponents,
                                  bScaleExponents, count, format->usesScale);
                if (std::isnan(sum) || std::fabs(sum) >= 1e20f) {
                    debugf("[ERROR] MAT: Encountered NaN or a large result.\n");
                }
                store_uram<float>(uram, cAddress, tcsr.cSwizzleMBS, sum);
            }
        }
    }
    debugf("MAT: 0x%08lx = 0x%08lx @ 0x%08lx\n", c, a, b);
    return true;
}

bool execute_conv2d(
    uint8_t* uram,
    const conv_tcsr_t& tcsr,
    uint64_t input,
    uint64_t kernel,
    uint64_t output,
    bool acc) {
    const auto format = decode_mat_format(tcsr.dataType);
    if (!format.has_value() || format->usesScale) {
        return false;
    }

    debugf("MAT: Calculating %s Conv Hi%lu Wi%lu Ci%lu Hk%lu Wk%lu Ho%lu Wo%lu Co%lu\n",
           format->name, tcsr.hi, tcsr.wi, tcsr.c, tcsr.hk, tcsr.wk, tcsr.ho, tcsr.wo, tcsr.co);
    const std::vector<float> inputValues = decode_conv_input(uram, tcsr, input, *format);
    const std::vector<float> kernelValues = decode_conv_kernel(uram, tcsr, kernel, *format);
    for (uint64_t ho = 0; ho < tcsr.ho; ++ho) {
        for (uint64_t wo = 0; wo < tcsr.wo; ++wo) {
            for (uint64_t co = 0; co < tcsr.co; ++co) {
                const uint64_t outputAddress = output + ho * tcsr.hoStride + wo * tcsr.woStride + co * sizeof(float);
                float sum = acc ? load_uram<float>(uram, outputAddress, tcsr.oSwizzleMBS) : 0.0f;
                for (uint64_t hk = 0; hk < tcsr.hk; ++hk) {
                    const int64_t hi = static_cast<int64_t>(ho * tcsr.strideH + hk * tcsr.dilationH) -
                        static_cast<int64_t>(tcsr.padU);
                    if (hi < 0 || static_cast<uint64_t>(hi) >= tcsr.hi) {
                        continue;
                    }
                    for (uint64_t wk = 0; wk < tcsr.wk; ++wk) {
                        const int64_t wi = static_cast<int64_t>(wo * tcsr.strideW + wk * tcsr.dilationW) -
                            static_cast<int64_t>(tcsr.padL);
                        if (wi < 0 || static_cast<uint64_t>(wi) >= tcsr.wi) {
                            continue;
                        }
                        const float* inputRow = inputValues.data() +
                            (static_cast<uint64_t>(hi) * tcsr.wi + static_cast<uint64_t>(wi)) * tcsr.c;
                        const float* kernelRow = kernelValues.data() +
                            ((hk * tcsr.wk + wk) * tcsr.co + co) * tcsr.c;
                        sum = dot_product_contiguous(sum, inputRow, kernelRow, tcsr.c);
                    }
                }
                store_uram<float>(uram, outputAddress, tcsr.oSwizzleMBS, sum);
            }
        }
    }
    debugf("MAT: Conv 0x%08lx = 0x%08lx @ 0x%08lx\n", output, input, kernel);
    return true;
}

mat_t::mat_t(exec_unit_t* parrent, uint8_t* uram):uram(uram) {
    parrent->register_issue_callback(this, &mat_t::issue);
    this->register_commit_callback(parrent, &exec_unit_t::commit);
    this->register_scoreboard_callback(parrent, &exec_unit_t::scoreboard);
    std::memset(this->tcsr, 0, sizeof(this->tcsr));
}

ulong mat_t::issue(const rvt_pack_t* inst) {
    const uint32_t unit = (inst->inst >> 23) & 0x1f;
    const uint32_t tcsrId = (inst->inst >> 13) & 0x3ff;
    const bool isMatTcsr = rvt::is_base(inst->inst) &&
        (unit == RVT_UNIT_MAT || (unit == RVT_UNIT_BASE && tcsrId == RVT_UNIT_MAT));
    const bool isMat = rvt::is_mat(inst->inst) || isMatTcsr;
    if (!isMat) {
        return 0;
    }
    this->inst_queue.enqueue(*inst);
    return 1;
}

void mat_t::execute(void) {
    while (1) {
        const auto inst = inst_queue.dequeue();
        MFENCE();

        const uint32_t unit = (inst.inst >> 23) & 0x1f;
        const uint32_t tcsrId = (inst.inst >> 13) & 0x3ff;
        const bool isMatTcsrWrite = rvt_is_base(inst.inst) && unit == RVT_UNIT_MAT &&
            ((inst.inst >> 12) & 0x1) == 0;
        const bool isMatTcsrRead = rvt_is_base(inst.inst) && unit == RVT_UNIT_MAT &&
            ((inst.inst >> 12) & 0x1) == 1;
        const bool isMatSbProduce = rvt_is_base(inst.inst) && unit == RVT_UNIT_BASE &&
            ((inst.inst >> 12) & 0x1) == 0;
        const bool isMatSbBar = rvt_is_base(inst.inst) && unit == RVT_UNIT_BASE &&
            ((inst.inst >> 12) & 0x1) == 1;

        if (isMatTcsrWrite || isMatTcsrRead) {
            debugf("MAT: tcsr read/write %u:\n", tcsrId);
            if (tcsrId >= 8 * 32) {
                debugf("* Illegal id.\n");
                continue;
            }
            uint64_t* rawTcsr = reinterpret_cast<uint64_t*>(&this->tcsr);
            if (isMatTcsrWrite) {
                if ((tcsrId % 32) != 31) {
                    const unsigned width = this->tcsr_width[tcsrId & 0x1f];
                    rawTcsr[tcsrId] = inst.rd & ((1ull << width) - 1);
                }
                debugf("* write %lx -> %lx.\n", inst.rd, rawTcsr[tcsrId]);
            } else {
                rvt_commit_t cmtPack;
                cmtPack.timestamp = inst.timestamp;
                cmtPack.rd = (inst.inst >> 7) & 0x1f;
                cmtPack.data = rawTcsr[tcsrId];
                commit(&cmtPack);
                debugf("* read %lx -> %lx.\n", rawTcsr[tcsrId], cmtPack.rd);
            }
            continue;
        }

        if (isMatSbProduce) {
            debugf("MAT: scoreboard produce %lx:\n", inst.rd);
            scoreboard(inst.rd);
            continue;
        }
        if (isMatSbBar) {
            rvt_commit_t cmtPack;
            cmtPack.timestamp = inst.timestamp;
            cmtPack.rd = (inst.inst >> 7) & 0x1f;
            cmtPack.data = inst.rd;
            debugf("MAT: scoreboard bar %lx:\n", cmtPack.rd);
            commit(&cmtPack);
            continue;
        }

        const uint64_t group = RVT_MAT_GROUP(inst.inst);
        if (IS_RVT_MAT_SETPTR(inst.inst)) {
            aScaleBase[group] = inst.rs1 & kUramAddressMask;
            bScaleBase[group] = inst.rs2 & kUramAddressMask;
            debugf("MAT: setptr group %lu A-scale=0x%lx B-scale=0x%lx\n",
                   group, aScaleBase[group], bScaleBase[group]);
            continue;
        }

        const uint64_t a = inst.rs1 & kUramAddressMask;
        const uint64_t b = inst.rs2 & kUramAddressMask;
        const uint64_t c = inst.rd & kUramAddressMask;
        debugf("MAT: DST(0x%08lx) <- A(0x%08lx) x B(0x%08lx)\n", c, a, b);
        if (IS_RVT_MAT_CONV2D(inst.inst) || IS_RVT_MAT_CONV2DA(inst.inst)) {
            conv_tcsr_t convTcsr;
            std::memcpy(&convTcsr, &tcsr[group], sizeof(convTcsr));
            if (!execute_conv2d(uram, convTcsr, a, b, c, IS_RVT_MAT_CONV2DA(inst.inst))) {
                debugf("MAT: unsupported Conv DataType 0x%02lx.\n", convTcsr.dataType);
            }
            continue;
        }
        if (!IS_RVT_MAT_MM_FAMILY(inst.inst)) {
            debugf("MAT: unsupported compute mode 0x%lx.\n", RVT_MAT_MODE(inst.inst));
            continue;
        }
        if (!execute_matmul(uram, tcsr[group], a, b, c,
                            aScaleBase[group], bScaleBase[group],
                            RVT_MAT_ACC(inst.inst) != 0)) {
            debugf("MAT: unsupported DataType 0x%02lx.\n", tcsr[group].dataType);
        }
    }
}

} // namespace RVTNG
