#pragma once
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <stdint.h>

static inline float GenRandomFloat(float scale) {
    return (double)scale * (rand() - RAND_MAX/2) / (double)RAND_MAX;
}

static inline std::vector<float> GenRandomFloats(int count, float scale) {
    std::vector<float> r(count);
    for(int i = 0; i < count; i ++) {
        r[i] = GenRandomFloat(scale);
    }
    return r;
}

static inline std::vector<int> GenRandomInt(int count, int scale) {
    std::vector<int> r(count);
    for(int i = 0; i < count ; i ++) {
        r[i] = (long long)scale * (rand() - RAND_MAX / 2) / (long long) RAND_MAX;
    }
    return r;
}

static inline float ScaleToFP32(int scale) {
    return exp2f((float)scale);
}

static inline uint8_t getUpper(uint16_t a) {
    return (a >> 8) & 0xff;
}

static inline uint8_t getLower(uint16_t a) {
    return a & 0xff;
}

union fp32_int32 {
    float f;
    uint32_t bits;
};

static inline uint16_t FP32ToBF16(float a) {
    // FIXME: Use bf16 by default
    uint32_t x = 0;
    std::memcpy(&x, &a, sizeof(x));
    // uint32_t sign = x & 0x80000000;
    int32_t exponent = (x >> 23) & 0xFF;
    // uint32_t mantissa = x & 0x007FFFFF;

    if (exponent == 0xFF) {
        return (x >> 16) | 0x007F;
    }

    if (exponent == 0) {
        return (x >> 16) & 0x8000;
    }

    uint32_t truncated = x >> 16;
    uint32_t round_bit = (x >> 15) & 1;
    uint32_t sticky_bits = x & 0x7FFF;

    if (round_bit && (sticky_bits || (truncated & 1))) {
        truncated += 1;
    }

    if ((truncated & 0x7F80) == 0x7F80) {
        truncated = (truncated & 0x8000) | 0x7F80;
    }

    return truncated;
}

static inline float BF16ToFP32(uint16_t a) {
    fp32_int32 u;
    u.bits = a << 16;
    return u.f;
}

static inline float BF16Clip(float a) {
    return BF16ToFP32(FP32ToBF16(a));
}

// TF32 operands are stored in FP32 containers and rounded to a 10-bit
// fraction before multiplication. Round-to-nearest-even matches the usual
// TF32 conversion rule and leaves infinities intact. A NaN whose payload
// lives only in discarded bits remains a NaN after truncation.
static inline float TF32Clip(float value) {
    fp32_int32 in;
    in.f = value;
    const uint32_t exponent = in.bits & 0x7f800000u;
    const uint32_t fraction = in.bits & 0x007fffffu;

    if (exponent == 0x7f800000u) {
        if (fraction != 0) {
            in.bits = (in.bits & 0xff800000u) | 0x00400000u;
        }
        return in.f;
    }

    const uint32_t rounding_bias = 0x00000fffu + ((in.bits >> 13) & 1u);
    in.bits = (in.bits + rounding_bias) & 0xffffe000u;
    return in.f;
}

static inline float FP16ToFP32(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t fraction = value & 0x03ffu;
    fp32_int32 out = {0};

    if (exponent == 0x1fu) {
        out.bits = sign | 0x7f800000u | (fraction << 13);
        if (fraction != 0) {
            out.bits |= 0x00400000u;
        }
        return out.f;
    }

    if (exponent == 0) {
        if (fraction == 0) {
            out.bits = sign;
            return out.f;
        }
        const float magnitude = std::ldexp(static_cast<float>(fraction), -24);
        return sign == 0 ? magnitude : -magnitude;
    }

    const float magnitude = std::ldexp(static_cast<float>(0x400u | fraction),
                                       static_cast<int>(exponent) - 25);
    return sign == 0 ? magnitude : -magnitude;
}

static inline uint32_t FP32ToSExp(float a) {
    fp32_int32 f;
    f.f = a;
    int raw_exp = (f.bits & (0xff << 23)) >> 23;
    int newval = raw_exp + 1 - 127; // [-126, 129]
    return newval & 0x1ff;
}

static inline uint32_t FP32ToSFrac(float a, int width) {
    fp32_int32 f;
    f.f = a;
    uint32_t sgn = (f.bits & (0x1 << 31)) >> 31;
    uint32_t raw_exp = (f.bits & (0xff << 23)) >> 23;
    int64_t  raw_frac = f.bits & (0x7fffff);
    if(raw_exp) raw_frac |= 0x800000;
    if(sgn) raw_frac = -raw_frac; // Width == 25, s.24
    if(width < 25) raw_frac = raw_frac >> (25 - width);
    else raw_frac = raw_frac << (width - 25);
    return raw_frac & ((1ull << (unsigned long long)width) - 1);
}

template<int N>
constexpr int exp_i2() {
    return 1 << N;
}

template<int N>
constexpr float exp_f2() {
    if constexpr (N>0) return exp_f2<N-1>()*2;
    if constexpr (N<0) return exp_f2<N+1>()/2;
    if constexpr (N==0) return 1.;
    return 0.; // This should never happen, but just in case.
}

template<int _E> //, int M=7-E>  1.7 bits!
struct FP8 {
    uint8_t bits;
    using type = FP8<_E>;
    static constexpr int E      = _E;
    static constexpr int M      = (7-_E);
    static constexpr int E_BIAS = exp_i2<E-1>()-1;
    static constexpr float MAX  = (2-exp_f2<-M+1>())*exp_f2<exp_i2<E-1>()>();
    static constexpr float MIN  = exp_f2<-M>()*exp_f2<2-exp_i2<E-1>()>();
};
// template<int E>
// inline float to_float(const FP8<E>& in) {
//     fp32_int32 out = {0};
//     out.bits = in.bits & 0x80;
//     out.bits <<= 24;
//     uint32_t _bits = in.bits & 0x7F;
//     _bits <<= (23-FP8<E>::M);
//     out.bits |= _bits;
//     out.f *= exp_f2<127-FP8<E>::E_BIAS>();
//     return out.f;
// }
template<int E>
inline float FP8ToFP32(uint8_t r) {
    static_assert(E == 4 || E == 5, "Only E4M3 and E5M2 FP8 are supported.");
    constexpr int M = 7 - E;
    constexpr int bias = (1 << (E - 1)) - 1;
    constexpr uint8_t fraction_mask = (1u << M) - 1u;
    constexpr uint8_t exponent_mask = (1u << E) - 1u;

    const int exponent = (r >> M) & exponent_mask;
    const int fraction = r & fraction_mask;
    float magnitude = 0.0f;
    if (exponent == 0) {
        // Subnormal: (fraction / 2^M) * 2^(1-bias).
        magnitude = std::ldexp(static_cast<float>(fraction), 1 - bias - M);
    } else {
        // NeoMat currently treats every non-zero exponent as a finite value;
        // this mirrors its internal FP14 operand unpacking for all bit patterns.
        magnitude = std::ldexp(static_cast<float>((1 << M) | fraction),
                               exponent - bias - M);
    }
    return (r & 0x80u) != 0 ? -magnitude : magnitude;
}

static inline float FP4E2M1ToFP32(uint8_t value) {
    static constexpr float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
    };
    const float decoded = magnitude[value & 0x7u];
    return (value & 0x8u) != 0 ? -decoded : decoded;
}

static inline int E8M0ToExponent(uint8_t value) {
    return static_cast<int>(value) - 127;
}
// template<int E>
// inline float fp8Mul(const FP8<E>& a, const FP8<E>& b) {
//     fp32_int32 out = {0};
//     out.bits = (a.bits ^ b.bits) & 0x80;
//     out.bits <<= 24;
//     // Calculate mantissa
//     long mant_a = (a.bits & ((1 << FP8<E>::M) - 1));
//     long mant_b = (b.bits & ((1 << FP8<E>::M) - 1));
//     // Calculate exponent
//     long exp_a = (a.bits >> FP8<E>::M) & ((1 << E) - 1);
//     long exp_b = (a.bits >> FP8<E>::M) & ((1 << E) - 1);
//     // Handle subnormal numbers
//     if(exp_a) mant_a |= (1 << FP8<E>::M); // mant_a is Q1.M
//     if(exp_b) mant_b |= (1 << FP8<E>::M); // mant_b is Q1.M
//     long exp = exp_a + exp_b - FP8<E>::E_BIAS;
//     long mant = mant_a * mant_b; // mant is Q2.2M
//     mant = (mant >> FP8<E>::M) + (mant & ((1 << FP8<E>::M) - 1)) ? 1 : 0; // mant is Q2.M
//     long mant_clz = __builtin_clz(mant);
//     const machine_width = sizeof(long) * 8;
//     long width = machine_width - mant_clz;
//     if(width == 0) return 0.0f;
//     return out.f;
// }
template<int E>
inline uint8_t FP32ToFP8(float value) {
    FP8<E> out;
    fp32_int32 in = {value};
    out.bits = (in.bits >> 24) & 0x80;
    in.bits &= 0x7fffffff;
    if (in.f >= FP8<E>::MAX) {
        out.bits |= 0x7E;
    } else if (in.f < FP8<E>::MIN) { // => 0.
    } else if(E == 4 && in.bits < 0x3C800000) { // 2^-6
        if(in.bits >= 0x3c500000)      out.bits |= 0x7;
        else if(in.bits >= 0x3c300000) out.bits |= 0x6;
        else if(in.bits >= 0x3c100000) out.bits |= 0x5;
        else if(in.bits >= 0x3be00000) out.bits |= 0x4;
        else if(in.bits >= 0x3ba00000) out.bits |= 0x3;
        else if(in.bits >= 0x3b400000) out.bits |= 0x2;
        else if(in.bits >  0x3a000000) out.bits |= 0x1;
        return out.bits;
    } else {
        in.f *= exp_f2<FP8<E>::E_BIAS-127>();
        uint32_t eps = (0x3fffff>>FP8<E>::M) + ((in.bits >> (23-FP8<E>::M)) & 0x1);
        in.bits += eps;
        out.bits |= (in.bits >> (23-FP8<E>::M)) & 0x7F;
    }
    return out.bits;
}

typedef int (*resultHandlerT)(int m, int n, float);

// Standard FP8 Matrix Multiplication Impl.
static inline void CalcFP8MatMul(uint8_t* a, uint8_t* b, resultHandlerT c, int tM, int tN, int tK) {
    float A[tK];
    for(int m = 0 ; m < tM; m ++) {
        // A Array
        for(int k = 0; k < tK; k ++) {
            A[k] = FP8ToFP32<4>(a[m * tK + k]);
        }
        for(int n = 0 ; n < tN ; n ++) {
            float acc = 0.f;
            for(int k = 0; k < tK; k ++) {
                acc += A[k] * FP8ToFP32<4>(b[n * tK + k]);
            }
            // c[m * tN] = acc;
            if(c(m, n, acc)) return;
        }
    }
}

// Standard BF16 Matrix Multiplication Impl.
static inline void CalcBF16MatMul(uint16_t* a, uint16_t* b, resultHandlerT c, int tM, int tN, int tK) {
    float A[tK];
    for(int m = 0 ; m < tM; m ++) {
        // A Array
        for(int k = 0; k < tK; k ++) {
            A[k] = BF16ToFP32(a[m * tK + k]);
        }
        for(int n = 0 ; n < tN ; n ++) {
            float acc = 0.f;
            for(int k = 0; k < tK; k ++) {
                acc += A[k] * BF16ToFP32(b[n * tK + k]);
            }
            // c[m * tN] = acc;
            if(c(m, n, acc)) return;
        }
    }
}

// FP32 快速点积
template <std::size_t LEN>
inline float fp32_dot_product(float sum, const float* a, const float* b) {
    static_assert(LEN > 0, "Vector length LEN must be greater than 0.");

    #pragma GCC ivdep
    for (std::size_t i = 0; i < LEN; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}


static inline uint16_t FP32ToFP16(float f) {
    uint32_t x = 0;
    memcpy(&x, &f, sizeof(float));
    uint16_t h = ((x >> 16) & 0x8000) |
             ((((x & 0x7f800000) - 0x38000000) >> 13) & 0x7c00) |
             ((x >> 13) & 0x03ff);

    if ((x & 0x7f800000) == 0x7f800000) {
      h = (h & 0x8000) | 0x7c00 | ((x >> 13) & 0x03ff);
    } else if ((x & 0x7f800000) > 0x477fe000) {
      h = (h & 0x8000) | 0x7c00;
    } else if ((x & 0x7f800000) < 0x38000000) {
      uint32_t sign = x & 0x80000000;
      uint32_t mantissa = (x & 0x007fffff) | 0x00800000;
      int32_t exponent = ((x >> 23) & 0xff) - 127 - 14;

      if (exponent < -24) {
        h = (uint16_t)(sign >> 16);
      } else {
        mantissa >>= (-exponent);
        h = (uint16_t)((sign >> 16) | (mantissa >> 13));
      }
    }
    return h;
}
