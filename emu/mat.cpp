#include "utils.hpp"
#include "common.hpp"
#include "mat.hpp"
#include "fputils.hpp"
#include "swizzle.hpp"
#include "rvt_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <iterator>
#include <vector>

namespace RVTNG {
namespace {

constexpr uint64_t kMatmulBlockK = 128;
constexpr uint64_t kMXScaleGroupK = 32;
constexpr uint64_t kUramAddressMask = (uint64_t{1} << 21) - 1;

static uint64_t packed_row_bytes(uint64_t elements, unsigned bits_per_element) {
    const __uint128_t bits = __uint128_t{elements} * bits_per_element;
    const __uint128_t bytes = (bits + 7) / 8;
    return bytes > std::numeric_limits<uint64_t>::max()
        ? std::numeric_limits<uint64_t>::max()
        : static_cast<uint64_t>(bytes);
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

enum class mat_operation_class_t : uint8_t {
    bits4,
    bits8,
    bits16,
};

using mat_convert_fn_t = float (*)(
    const uint8_t* uram, uint64_t rowBase, uint64_t index,
    uint64_t swizzleMBS);

struct mat_operand_format_t {
    mat_convert_fn_t convert;
    mat_operation_class_t operationClass;
    unsigned bitsPerElement;
    bool usesScale;
    const char* name;
};

struct mat_format_pair_t {
    mat_operand_format_t a;
    mat_operand_format_t b;
};

static uint64_t element_byte_offset(uint64_t index, unsigned bitsPerElement) {
    return bitsPerElement == 4 ? index / 2 : index * (bitsPerElement / 8);
}

static uint8_t load_nibble(const uint8_t* uram, uint64_t rowBase,
                           uint64_t index, uint64_t swizzleMBS) {
    const uint64_t address = rowBase + element_byte_offset(index, 4);
    const uint8_t packed = load_uram<uint8_t>(uram, address, swizzleMBS);
    return (index & 1) == 0 ? packed & 0x0fu : packed >> 4;
}

static float convert_bf16(const uint8_t* uram, uint64_t rowBase,
                          uint64_t index, uint64_t swizzleMBS) {
    return BF16ToFP32(load_uram<uint16_t>(
        uram, rowBase + index * sizeof(uint16_t), swizzleMBS));
}

static float convert_fp16(const uint8_t* uram, uint64_t rowBase,
                          uint64_t index, uint64_t swizzleMBS) {
    const uint16_t raw = load_uram<uint16_t>(
        uram, rowBase + index * sizeof(uint16_t), swizzleMBS);
    // The 16-bit MAT class enters the TF32 path after dropping the lowest
    // three FP16 fraction bits.
    return FP16ToFP32(raw & 0xfff8u);
}

template<int E>
static float convert_fp8(const uint8_t* uram, uint64_t rowBase,
                         uint64_t index, uint64_t swizzleMBS) {
    return FP8ToFP32<E>(load_uram<uint8_t>(
        uram, rowBase + index, swizzleMBS));
}

static float convert_sint8(const uint8_t* uram, uint64_t rowBase,
                           uint64_t index, uint64_t swizzleMBS) {
    const int raw = load_uram<uint8_t>(uram, rowBase + index, swizzleMBS);
    return static_cast<float>(raw < 0x80 ? raw : raw - 0x100);
}

static float convert_uint8(const uint8_t* uram, uint64_t rowBase,
                           uint64_t index, uint64_t swizzleMBS) {
    return static_cast<float>(load_uram<uint8_t>(
        uram, rowBase + index, swizzleMBS));
}

static float convert_fp4_e2m1(const uint8_t* uram, uint64_t rowBase,
                              uint64_t index, uint64_t swizzleMBS) {
    return FP4E2M1ToFP32(load_nibble(
        uram, rowBase, index, swizzleMBS));
}

static float convert_sint4(const uint8_t* uram, uint64_t rowBase,
                           uint64_t index, uint64_t swizzleMBS) {
    const int raw = load_nibble(uram, rowBase, index, swizzleMBS);
    return static_cast<float>(raw < 8 ? raw : raw - 16);
}

static float convert_uint4(const uint8_t* uram, uint64_t rowBase,
                           uint64_t index, uint64_t swizzleMBS) {
    return static_cast<float>(load_nibble(
        uram, rowBase, index, swizzleMBS));
}

static std::optional<mat_operand_format_t> decode_operand_format(
    uint8_t operandType) {
    const uint8_t base = operandType & RVT_MAT_TYPE_BASE_MASK;
    const bool usesScale = (operandType & RVT_MAT_TYPE_MX_BIT) != 0;
    const mat_operation_class_t operationClass = (base & 0x10u) == 0
        ? mat_operation_class_t::bits8
        : ((base & 0x18u) == 0x10u
            ? mat_operation_class_t::bits16
            : mat_operation_class_t::bits4);
    if(operationClass == mat_operation_class_t::bits16 && usesScale) {
        return std::nullopt;
    }

    switch(base) {
    case RVT_MAT_TYPE_E0M7:
        return mat_operand_format_t{
            convert_fp8<0>, operationClass, 8, usesScale,
            usesScale ? "MXFP8-E0M7" : "FP8-E0M7"};
    case RVT_MAT_TYPE_E1M6:
        return mat_operand_format_t{
            convert_fp8<1>, operationClass, 8, usesScale,
            usesScale ? "MXFP8-E1M6" : "FP8-E1M6"};
    case RVT_MAT_TYPE_E2M5:
        return mat_operand_format_t{
            convert_fp8<2>, operationClass, 8, usesScale,
            usesScale ? "MXFP8-E2M5" : "FP8-E2M5"};
    case RVT_MAT_TYPE_E3M4:
        return mat_operand_format_t{
            convert_fp8<3>, operationClass, 8, usesScale,
            usesScale ? "MXFP8-E3M4" : "FP8-E3M4"};
    case RVT_MAT_TYPE_E4M3:
        return mat_operand_format_t{
            convert_fp8<4>, operationClass, 8, usesScale,
            usesScale ? "MXFP8-E4M3" : "FP8-E4M3"};
    case RVT_MAT_TYPE_E5M2:
        return mat_operand_format_t{
            convert_fp8<5>, operationClass, 8, usesScale,
            usesScale ? "MXFP8-E5M2" : "FP8-E5M2"};
    case RVT_MAT_TYPE_E6M1:
        return mat_operand_format_t{
            convert_fp8<6>, operationClass, 8, usesScale,
            usesScale ? "MXFP8-E6M1" : "FP8-E6M1"};
    case RVT_MAT_TYPE_S8:
        return mat_operand_format_t{
            convert_sint8, operationClass, 8, usesScale,
            usesScale ? "MXS8" : "S8"};
    case RVT_MAT_TYPE_U8:
        return mat_operand_format_t{
            convert_uint8, operationClass, 8, usesScale,
            usesScale ? "MXU8" : "U8"};
    case RVT_MAT_TYPE_FP16:
        return mat_operand_format_t{
            convert_fp16, operationClass, 16, false, "FP16"};
    case RVT_MAT_TYPE_BF16:
        return mat_operand_format_t{
            convert_bf16, operationClass, 16, false, "BF16"};
    case RVT_MAT_TYPE_FP4_E2M1:
        return mat_operand_format_t{
            convert_fp4_e2m1, operationClass, 4, usesScale,
            usesScale ? "MXFP4-E2M1" : "FP4-E2M1"};
    case RVT_MAT_TYPE_S4:
        return mat_operand_format_t{
            convert_sint4, operationClass, 4, usesScale,
            usesScale ? "MXS4" : "S4"};
    case RVT_MAT_TYPE_U4:
        return mat_operand_format_t{
            convert_uint4, operationClass, 4, usesScale,
            usesScale ? "MXU4" : "U4"};
    default:
        // TF16 and TF32 are architecturally allocated, but their external
        // storage/container conversion paths are not implemented here yet.
        return std::nullopt;
    }
}

static std::optional<uint8_t> decode_legacy_operand(uint8_t dataType) {
    switch(dataType) {
    case RVT_TYPE_BF16:
        return RVT_MAT_OPERAND(RVT_MAT_TYPE_BF16, false);
    case RVT_TYPE_FP16:
        return RVT_MAT_OPERAND(RVT_MAT_TYPE_FP16, false);
    case RVT_TYPE_FP8_E4M3:
        return RVT_MAT_OPERAND(RVT_MAT_TYPE_E4M3, false);
    case RVT_TYPE_FP8_E5M2:
        return RVT_MAT_OPERAND(RVT_MAT_TYPE_E5M2, false);
    case RVT_TYPE_MX_FP8_E4M3:
        return RVT_MAT_OPERAND(RVT_MAT_TYPE_E4M3, true);
    case RVT_TYPE_MX_FP8_E5M2:
        return RVT_MAT_OPERAND(RVT_MAT_TYPE_E5M2, true);
    case RVT_TYPE_FP4_E2M1:
        return RVT_MAT_OPERAND(RVT_MAT_TYPE_FP4_E2M1, false);
    case RVT_TYPE_MX_FP4_E2M1:
        return RVT_MAT_OPERAND(RVT_MAT_TYPE_FP4_E2M1, true);
    case RVT_TYPE_S8:
        return RVT_MAT_OPERAND(RVT_MAT_TYPE_S8, false);
    case RVT_TYPE_U8:
        return RVT_MAT_OPERAND(RVT_MAT_TYPE_U8, false);
    case RVT_TYPE_S4:
        return RVT_MAT_OPERAND(RVT_MAT_TYPE_S4, false);
    case RVT_TYPE_U4:
        return RVT_MAT_OPERAND(RVT_MAT_TYPE_U4, false);
    default:
        return std::nullopt;
    }
}

static std::optional<mat_format_pair_t> decode_mat_formats(
    uint64_t dataType) {
    if((dataType & ~uint64_t{0xffff}) != 0) {
        return std::nullopt;
    }
    const uint16_t encoded = static_cast<uint16_t>(dataType);
    uint8_t aType = 0;
    uint8_t bType = 0;
    if((encoded & 0x8000u) == 0) {
        if((encoded & 0x7f00u) != 0) {
            return std::nullopt;
        }
        const auto operand = decode_legacy_operand(encoded & 0xffu);
        if(!operand) {
            return std::nullopt;
        }
        aType = *operand;
        bType = *operand;
    } else {
        if(((encoded >> 12) & 0x7u) != 0) {
            return std::nullopt;
        }
        aType = encoded & 0x3fu;
        bType = (encoded >> 6) & 0x3fu;
    }

    const auto a = decode_operand_format(aType);
    const auto b = decode_operand_format(bType);
    if(!a || !b || a->operationClass != b->operationClass) {
        return std::nullopt;
    }
    return mat_format_pair_t{*a, *b};
}

static void decode_row(
    const uint8_t* uram,
    uint64_t rowBase,
    uint64_t swizzleMBS,
    uint64_t firstK,
    uint64_t count,
    const mat_operand_format_t& format,
    uint64_t scaleRowBase,
    float* values) {
    const mat_convert_fn_t convert = format.convert;
    if(!format.usesScale) {
        for(uint64_t subK = 0; subK < count; ++subK) {
            values[subK] = convert(
                uram, rowBase, firstK + subK, swizzleMBS);
        }
        return;
    }

    uint64_t cachedScaleGroup = std::numeric_limits<uint64_t>::max();
    int cachedScaleExponent = 0;
    for(uint64_t subK = 0; subK < count; ++subK) {
        const uint64_t logicalK = firstK + subK;
        float value = convert(
            uram, rowBase, logicalK, swizzleMBS);
        const uint64_t scaleGroup = logicalK / kMXScaleGroupK;
        if(scaleGroup != cachedScaleGroup) {
            const uint8_t rawScale = load_uram<uint8_t>(
                uram, scaleRowBase + scaleGroup, 0);
            cachedScaleExponent = E8M0ToExponent(rawScale);
            cachedScaleGroup = scaleGroup;
        }
        values[subK] = std::ldexp(value, cachedScaleExponent);
    }
}

static void decode_matrix_block(
    const uint8_t* uram,
    uint64_t base,
    uint64_t rows,
    uint64_t rowStride,
    uint64_t swizzleMBS,
    uint64_t firstK,
    uint64_t count,
    const mat_operand_format_t& format,
    uint64_t scaleBase,
    uint64_t scaleStride,
    std::vector<float>& values) {
    values.resize(rows * kMatmulBlockK);
    for(uint64_t row = 0; row < rows; ++row) {
        decode_row(
            uram, base + row * rowStride, swizzleMBS, firstK, count,
            format, scaleBase + row * scaleStride,
            values.data() + row * kMatmulBlockK);
    }
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
    const mat_operand_format_t& format,
    uint64_t scaleBase) {
    std::vector<float> values(tcsr.hi * tcsr.wi * tcsr.c);
    for (uint64_t hi = 0; hi < tcsr.hi; ++hi) {
        for (uint64_t wi = 0; wi < tcsr.wi; ++wi) {
            const uint64_t source = input + hi * tcsr.hiStride + wi * tcsr.wiStride;
            float* destination = values.data() + (hi * tcsr.wi + wi) * tcsr.c;
            const uint64_t scaleRow = scaleBase +
                hi * tcsr.scaleHiStride + wi * tcsr.scaleWiStride;
            decode_row(uram, source, tcsr.iSwizzleMBS, 0, tcsr.c,
                       format, scaleRow, destination);
        }
    }
    return values;
}

static std::vector<float> decode_conv_kernel(
    const uint8_t* uram,
    const conv_tcsr_t& tcsr,
    uint64_t kernel,
    const mat_operand_format_t& format,
    uint64_t scaleBase) {
    std::vector<float> values(tcsr.hk * tcsr.wk * tcsr.co * tcsr.c);
    for (uint64_t hk = 0; hk < tcsr.hk; ++hk) {
        for (uint64_t wk = 0; wk < tcsr.wk; ++wk) {
            for (uint64_t co = 0; co < tcsr.co; ++co) {
                const uint64_t source = kernel + hk * tcsr.hkStride + wk * tcsr.wkStride + co * tcsr.coStride;
                float* destination = values.data() + ((hk * tcsr.wk + wk) * tcsr.co + co) * tcsr.c;
                const uint64_t scaleRow = scaleBase +
                    hk * tcsr.scaleHkStride + wk * tcsr.scaleWkStride +
                    co * tcsr.scaleCoStride;
                decode_row(uram, source, tcsr.kSwizzleMBS, 0, tcsr.c,
                           format, scaleRow, destination);
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
    const auto formats = decode_mat_formats(tcsr.dataType);
    if(!formats) {
        return false;
    }

    const uint64_t M = tcsr.m + 1;
    const uint64_t N = tcsr.n + 1;
    const uint64_t K = tcsr.k + 1;
    debugf("MAT: Calculating A=%s B=%s M%lu N%lu K%lu\n",
           formats->a.name, formats->b.name, M, N, K);

    // Resolve A/B converter function pointers once, convert each K block once,
    // and reuse one contiguous FP32 dot-product kernel for every format pair.
    std::vector<float> aValues;
    std::vector<float> bValues;
    std::vector<float> cValues(M * N, 0.0f);
    if(acc) {
        for(uint64_t m = 0; m < M; ++m) {
            for(uint64_t n = 0; n < N; ++n) {
                const uint64_t cAddress =
                    c + m * tcsr.cStride + n * sizeof(float);
                cValues[m * N + n] = load_uram<float>(
                    uram, cAddress, tcsr.cSwizzleMBS);
            }
        }
    }

    for (uint64_t k = 0; k < K; k += kMatmulBlockK) {
        const uint64_t count = std::min(kMatmulBlockK, K - k);
        decode_matrix_block(
            uram, a, M, tcsr.aStride, tcsr.aSwizzleMBS, k, count,
            formats->a, aScale, tcsr.aScaleStride, aValues);
        decode_matrix_block(
            uram, b, N, tcsr.bStride, tcsr.bSwizzleMBS, k, count,
            formats->b, bScale, tcsr.bScaleStride, bValues);

        for (uint64_t m = 0; m < M; ++m) {
            const float* aRow =
                aValues.data() + m * kMatmulBlockK;
            for (uint64_t n = 0; n < N; ++n) {
                float& sum = cValues[m * N + n];
                if (std::isnan(sum)) {
                    debugf("[ERROR] MAT: Encountered NaN in C input.\n");
                }
                const float* bRow =
                    bValues.data() + n * kMatmulBlockK;
                sum = dot_product_contiguous(sum, aRow, bRow, count);
                if (std::isnan(sum) || std::fabs(sum) >= 1e20f) {
                    debugf("[ERROR] MAT: Encountered NaN or a large result.\n");
                }
            }
        }
    }
    for(uint64_t m = 0; m < M; ++m) {
        for(uint64_t n = 0; n < N; ++n) {
            const uint64_t cAddress =
                c + m * tcsr.cStride + n * sizeof(float);
            store_uram<float>(uram, cAddress, tcsr.cSwizzleMBS,
                              cValues[m * N + n]);
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
    uint64_t inputScale,
    uint64_t kernelScale,
    bool acc) {
    const auto formats = decode_mat_formats(tcsr.dataType);
    if(!formats) {
        return false;
    }

    debugf("MAT: Calculating A=%s B=%s Conv Hi%lu Wi%lu Ci%lu Hk%lu Wk%lu Ho%lu Wo%lu Co%lu\n",
           formats->a.name, formats->b.name, tcsr.hi, tcsr.wi, tcsr.c,
           tcsr.hk, tcsr.wk, tcsr.ho, tcsr.wo, tcsr.co);
    const std::vector<float> inputValues = decode_conv_input(
        uram, tcsr, input, formats->a, inputScale);
    const std::vector<float> kernelValues = decode_conv_kernel(
        uram, tcsr, kernel, formats->b, kernelScale);
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

mat_t::mat_t(exec_unit_t* parrent, uint8_t* uram,
             tensor_context_t* context,
             interrupt_controller_t* interrupts)
    : uram(uram), context(context), interrupts(interrupts),
      active_context(tensor_context_snapshot_t::defaults()) {
    parrent->register_issue_callback(this, &mat_t::issue);
    this->register_commit_callback(parrent, &exec_unit_t::commit);
    this->register_scoreboard_callback(parrent, &exec_unit_t::scoreboard);
    std::memset(this->tcsr, 0, sizeof(this->tcsr));
}

void mat_t::enqueue_context(const rvt_pack_t& inst,
                            const tensor_context_snapshot_t& context_value) {
    inst_queue.enqueue(queued_tensor_inst_t{inst, context_value, true});
}

bool mat_t::check_uram_span(uint64_t address, uint64_t size,
                            uint64_t swizzle,
                            interrupt_source_t range_source,
                            interrupt_source_t align_source) const {
    if(context == nullptr && interrupts == nullptr) {
        return true;
    }
    constexpr uint64_t portBytes = 32;
    const uint64_t physicalAddress = swizzle_addr_raw(address, swizzle);
    if((physicalAddress & (portBytes - 1)) != 0) {
        if(interrupts) {
            interrupts->raise(align_source,
                              static_cast<uint32_t>(physicalAddress));
        }
        return false;
    }
    if(size == 0) {
        size = 1;
    }

    uint64_t consumed = 0;
    while(consumed < size) {
        if(address > std::numeric_limits<uint64_t>::max() - consumed) {
            if(interrupts) {
                interrupts->raise(range_source,
                                  std::numeric_limits<uint32_t>::max());
            }
            return false;
        }
        const uint64_t logicalAddress = address + consumed;
        const uint64_t portOffset = logicalAddress & (portBytes - 1);
        const uint64_t logicalPort = logicalAddress - portOffset;
        const uint64_t physicalPort =
            swizzle_addr_raw(logicalPort, swizzle) & ~(portBytes - 1);
        if(!active_context.uram_range.contains(physicalPort, portBytes)) {
            if(interrupts) {
                interrupts->raise(range_source,
                                  static_cast<uint32_t>(physicalPort));
            }
            return false;
        }
        consumed += std::min<uint64_t>(portBytes - portOffset,
                                       size - consumed);
    }
    return true;
}

bool mat_t::validate_matmul_accesses(const mat_tcsr_t& control,
                                     uint64_t a, uint64_t b, uint64_t c,
                                     uint64_t a_scale, uint64_t b_scale,
                                     bool acc) const {
    const auto formats = decode_mat_formats(control.dataType);
    if(!formats) {
        return true;
    }
    const uint64_t m = control.m + 1;
    const uint64_t n = control.n + 1;
    const uint64_t k = control.k + 1;
    const uint64_t aRowBytes = packed_row_bytes(
        k, formats->a.bitsPerElement);
    const uint64_t bRowBytes = packed_row_bytes(
        k, formats->b.bitsPerElement);
    const auto checkRows = [this](
        uint64_t base, uint64_t rows, uint64_t stride, uint64_t rowBytes,
        uint64_t swizzle, interrupt_source_t rangeSource,
        interrupt_source_t alignSource) {
        for(uint64_t row = 0; row < rows; ++row) {
            const __uint128_t wideAddress =
                __uint128_t{base} + __uint128_t{row} * stride;
            const uint64_t rowAddress =
                wideAddress > std::numeric_limits<uint64_t>::max()
                ? std::numeric_limits<uint64_t>::max()
                : static_cast<uint64_t>(wideAddress);
            if(!check_uram_span(rowAddress, rowBytes, swizzle,
                                rangeSource, alignSource)) {
                return false;
            }
        }
        return true;
    };

    if(!checkRows(a, m, control.aStride, aRowBytes, control.aSwizzleMBS,
                  interrupt_source_t::neomat_uram_a_read_range,
                  interrupt_source_t::neomat_uram_a_read_align) ||
       !checkRows(b, n, control.bStride, bRowBytes, control.bSwizzleMBS,
                  interrupt_source_t::neomat_uram_b_read_range,
                  interrupt_source_t::neomat_uram_b_read_align)) {
        return false;
    }
    const uint64_t scaleRow =
        (k + kMXScaleGroupK - 1) / kMXScaleGroupK;
    if(formats->a.usesScale &&
       !checkRows(a_scale, m, control.aScaleStride, scaleRow, 0,
                  interrupt_source_t::neomat_uram_scale_read_range,
                  interrupt_source_t::neomat_uram_scale_read_align)) {
        return false;
    }
    if(formats->b.usesScale &&
       !checkRows(b_scale, n, control.bScaleStride, scaleRow, 0,
                  interrupt_source_t::neomat_uram_scale_read_range,
                  interrupt_source_t::neomat_uram_scale_read_align)) {
        return false;
    }
    if(acc && !checkRows(c, m, control.cStride, n * sizeof(float),
                         control.cSwizzleMBS,
                         interrupt_source_t::neomat_uram_c_read_range,
                         interrupt_source_t::neomat_uram_c_read_align)) {
        return false;
    }
    return checkRows(c, m, control.cStride, n * sizeof(float),
                     control.cSwizzleMBS,
                     interrupt_source_t::neomat_uram_c_write_range,
                     interrupt_source_t::neomat_uram_c_write_align);
}

bool mat_t::validate_conv_accesses(const conv_tcsr_t& control,
                                   uint64_t input, uint64_t kernel,
                                   uint64_t output,
                                   uint64_t input_scale,
                                   uint64_t kernel_scale,
                                   bool acc) const {
    const auto formats = decode_mat_formats(control.dataType);
    if(!formats) {
        return true;
    }
    const uint64_t inputTail = packed_row_bytes(
        control.c, formats->a.bitsPerElement);
    const uint64_t kernelTail = packed_row_bytes(
        control.c, formats->b.bitsPerElement);
    const auto checkRegions = [this](
        uint64_t base,
        uint64_t count0, uint64_t stride0,
        uint64_t count1, uint64_t stride1,
        uint64_t count2, uint64_t stride2,
        uint64_t tailBytes, uint64_t swizzle,
        interrupt_source_t rangeSource,
        interrupt_source_t alignSource) {
        for(uint64_t index0 = 0; index0 < count0; ++index0) {
            for(uint64_t index1 = 0; index1 < count1; ++index1) {
                for(uint64_t index2 = 0; index2 < count2; ++index2) {
                const __uint128_t wideAddress = __uint128_t{base} +
                    __uint128_t{index0} * stride0 +
                    __uint128_t{index1} * stride1 +
                    __uint128_t{index2} * stride2;
                const uint64_t address =
                    wideAddress > std::numeric_limits<uint64_t>::max()
                    ? std::numeric_limits<uint64_t>::max()
                    : static_cast<uint64_t>(wideAddress);
                if(!check_uram_span(address, tailBytes, swizzle,
                                    rangeSource, alignSource)) {
                    return false;
                }
                }
            }
        }
        return true;
    };

    if(!checkRegions(input,
                     control.hi, control.hiStride,
                     control.wi, control.wiStride,
                     1, 0,
                     inputTail, control.iSwizzleMBS,
                     interrupt_source_t::neomat_uram_a_read_range,
                     interrupt_source_t::neomat_uram_a_read_align) ||
       !checkRegions(kernel,
                     control.hk, control.hkStride,
                     control.wk, control.wkStride,
                     control.co, control.coStride,
                     kernelTail, control.kSwizzleMBS,
                     interrupt_source_t::neomat_uram_b_read_range,
                     interrupt_source_t::neomat_uram_b_read_align)) {
        return false;
    }
    const uint64_t scaleTail =
        (control.c + kMXScaleGroupK - 1) / kMXScaleGroupK;
    if(formats->a.usesScale &&
       !checkRegions(input_scale,
                     control.hi, control.scaleHiStride,
                     control.wi, control.scaleWiStride,
                     1, 0, scaleTail, 0,
                     interrupt_source_t::neomat_uram_scale_read_range,
                     interrupt_source_t::neomat_uram_scale_read_align)) {
        return false;
    }
    if(formats->b.usesScale &&
       !checkRegions(kernel_scale,
                     control.hk, control.scaleHkStride,
                     control.wk, control.scaleWkStride,
                     control.co, control.scaleCoStride,
                     scaleTail, 0,
                     interrupt_source_t::neomat_uram_scale_read_range,
                     interrupt_source_t::neomat_uram_scale_read_align)) {
        return false;
    }
    if(acc && !checkRegions(output,
                            control.ho, control.hoStride,
                            control.wo, control.woStride,
                            1, 0,
                            control.co * sizeof(float), control.oSwizzleMBS,
                            interrupt_source_t::neomat_uram_c_read_range,
                            interrupt_source_t::neomat_uram_c_read_align)) {
        return false;
    }
    return checkRegions(output,
                        control.ho, control.hoStride,
                        control.wo, control.woStride,
                        1, 0,
                        control.co * sizeof(float), control.oSwizzleMBS,
                        interrupt_source_t::neomat_uram_c_write_range,
                        interrupt_source_t::neomat_uram_c_write_align);
}

ulong mat_t::issue(const rvt_pack_t* inst) {
    const uint32_t unit = (inst->inst >> 23) & 0x1f;
    const uint32_t tcsrId = (inst->inst >> 13) & 0x3ff;
    const bool isMatTcsr = rvt::is_base(inst->inst) &&
        (unit == RVT_UNIT_MAT ||
         (unit == RVT_UNIT_BASE && (tcsrId & 0x1f) == RVT_UNIT_MAT));
    const bool isMat = IS_RVT_TOP_MAT(inst->inst) || isMatTcsr;
    if (!isMat) {
        return 0;
    }
    const auto snapshot = context ? context->snapshot()
                                  : tensor_context_snapshot_t::defaults();
    this->inst_queue.enqueue(queued_tensor_inst_t{*inst, snapshot, false});
    return 1;
}

void mat_t::execute(void) {
    while (1) {
        const auto queued = inst_queue.dequeue();
        const auto& inst = queued.inst;
        active_context = queued.context;
        MFENCE();
        if(queued.context_write) {
            continue;
        }

        const uint32_t unit = (inst.inst >> 23) & 0x1f;
        const uint32_t tcsrId = (inst.inst >> 13) & 0x3ff;
        const bool isMatTcsrWrite = rvt_is_base(inst.inst) && unit == RVT_UNIT_MAT &&
            ((inst.inst >> 12) & 0x1) == 0;
        const bool isMatTcsrRead = rvt_is_base(inst.inst) && unit == RVT_UNIT_MAT &&
            ((inst.inst >> 12) & 0x1) == 1;
        const bool isMatSbProduce = rvt_is_base(inst.inst) && unit == RVT_UNIT_BASE &&
            tcsrId == RVT_UNIT_MAT &&
            ((inst.inst >> 12) & 0x1) == 0;
        const bool isMatSbBar = rvt_is_base(inst.inst) && unit == RVT_UNIT_BASE &&
            tcsrId == RVT_UNIT_MAT &&
            ((inst.inst >> 12) & 0x1) == 1;

        if (isMatTcsrWrite || isMatTcsrRead) {
            debugf("MAT: tcsr read/write %u:\n", tcsrId);
            const uint8_t logicalGroup = tcsrId / 32;
            const auto physicalGroup = active_context.physical_group(
                tensor_unit_t::mat, logicalGroup, std::size(tcsr));
            if (!physicalGroup) {
                if(interrupts) {
                    interrupts->raise(interrupt_source_t::neomat_tcsr_access_error,
                                      static_cast<uint32_t>(inst.inst));
                }
                if(isMatTcsrRead) {
                    rvt_commit_t cmtPack{};
                    cmtPack.timestamp = inst.timestamp;
                    cmtPack.rd = (inst.inst >> 7) & 0x1f;
                    commit(&cmtPack);
                }
                continue;
            }
            uint64_t* rawTcsr = reinterpret_cast<uint64_t*>(
                &this->tcsr[*physicalGroup]);
            const uint32_t subAddress = tcsrId & 0x1f;
            if (isMatTcsrWrite) {
                if (subAddress != 31) {
                    const unsigned width = this->tcsr_width[subAddress];
                    rawTcsr[subAddress] = inst.rd & ((1ull << width) - 1);
                }
                debugf("* write %lx -> %lx.\n", inst.rd, rawTcsr[subAddress]);
            } else {
                rvt_commit_t cmtPack;
                cmtPack.timestamp = inst.timestamp;
                cmtPack.rd = (inst.inst >> 7) & 0x1f;
                cmtPack.data = rawTcsr[subAddress];
                commit(&cmtPack);
                debugf("* read %lx -> %lx.\n", rawTcsr[subAddress], cmtPack.rd);
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

        if(!IS_RVT_MAT_COMPUTE_FAMILY(inst.inst) &&
           !IS_RVT_MAT_SETPTR(inst.inst)) {
            if(interrupts) {
                interrupts->raise(interrupt_source_t::neomat_decode_error,
                                  static_cast<uint32_t>(inst.inst));
            }
            continue;
        }

        const uint64_t group = RVT_MAT_GROUP(inst.inst);
        const auto physicalGroup = active_context.physical_group(
            tensor_unit_t::mat, group, std::size(tcsr));
        mat_tcsr_t emptyTcsr{};
        if(!physicalGroup && interrupts) {
            interrupts->raise(interrupt_source_t::neomat_tcsr_access_error,
                              static_cast<uint32_t>(inst.inst));
        }
        if (IS_RVT_MAT_SETPTR(inst.inst)) {
            if(physicalGroup) {
                aScaleBase[*physicalGroup] = inst.rs1 & kUramAddressMask;
                bScaleBase[*physicalGroup] = inst.rs2 & kUramAddressMask;
            }
            debugf("MAT: setptr group %lu A-scale=0x%lx B-scale=0x%lx\n",
                   group,
                   physicalGroup ? aScaleBase[*physicalGroup] : 0,
                   physicalGroup ? bScaleBase[*physicalGroup] : 0);
            continue;
        }

        const uint64_t a = inst.rs1 & kUramAddressMask;
        const uint64_t b = inst.rs2 & kUramAddressMask;
        const uint64_t c = inst.rd & kUramAddressMask;
        debugf("MAT: DST(0x%08lx) <- A(0x%08lx) x B(0x%08lx)\n", c, a, b);
        const mat_tcsr_t& selected = physicalGroup
            ? tcsr[*physicalGroup] : emptyTcsr;
        const uint64_t selectedAScale = physicalGroup
            ? aScaleBase[*physicalGroup] : 0;
        const uint64_t selectedBScale = physicalGroup
            ? bScaleBase[*physicalGroup] : 0;
        if(!decode_mat_formats(selected.dataType)) {
            if(interrupts) {
                interrupts->raise(
                    interrupt_source_t::neomat_datatype_error,
                    static_cast<uint32_t>(selected.dataType));
            }
            debugf("MAT: invalid DataType 0x%04lx; completing as no-op.\n",
                   selected.dataType);
            continue;
        }

        if (IS_RVT_MAT_CONV2D(inst.inst) || IS_RVT_MAT_CONV2DA(inst.inst)) {
            conv_tcsr_t convTcsr;
            std::memcpy(&convTcsr, &selected, sizeof(convTcsr));
            if(!validate_conv_accesses(convTcsr, a, b, c,
                                       selectedAScale, selectedBScale,
                                       IS_RVT_MAT_CONV2DA(inst.inst))) {
                continue;
            }
            if (!execute_conv2d(uram, convTcsr, a, b, c,
                                selectedAScale, selectedBScale,
                                IS_RVT_MAT_CONV2DA(inst.inst))) {
                debugf("MAT: unsupported Conv DataType 0x%02lx.\n", convTcsr.dataType);
            }
            continue;
        }
        if (!IS_RVT_MAT_MM_FAMILY(inst.inst)) {
            debugf("MAT: unsupported compute mode 0x%llx.\n",
                   static_cast<unsigned long long>(RVT_MAT_MODE(inst.inst)));
            continue;
        }
        if(!validate_matmul_accesses(selected, a, b, c,
                                    selectedAScale, selectedBScale,
                                    RVT_MAT_ACC(inst.inst) != 0)) {
            continue;
        }
        if (!execute_matmul(uram, selected, a, b, c,
                            selectedAScale, selectedBScale,
                            RVT_MAT_ACC(inst.inst) != 0)) {
            debugf("MAT: unsupported DataType 0x%02lx.\n", selected.dataType);
        }
    }
}

} // namespace RVTNG
