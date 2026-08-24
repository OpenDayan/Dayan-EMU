#pragma once
#include <cstdint>

//=============================================================================
// RVT Decoder Macros
// RISC-V Tensor Extension Instruction Decoding Utilities
//=============================================================================

// Instruction opcodes
#define RVT_OPCODE_32 0b0001011  // 0x0B - 32-bit format
#define RVT_OPCODE_64 0b0111111  // 0x3F - 64-bit format

//=============================================================================
// Helper Macros
//=============================================================================

// Extract bit range from instruction (inclusive)
#define RVT_BITS(inst, hi, lo) (((inst) >> (lo)) & ((1ULL << ((hi) - (lo) + 1)) - 1))

// Check if instruction matches mask and pattern
#define RVT_MATCH(inst, mask, pattern) (((inst) & (mask)) == (pattern))

// Sign-extend a value
#define RVT_SIGN_EXTEND(val, bits) \
    (((int64_t)(val) ^ ((1LL << ((bits) - 1)) - 1)) - (1LL << ((bits) - 1)))

//=============================================================================
// Instruction Format Detection
//=============================================================================

// Check if instruction is a 32-bit RVT instruction
#define IS_RVT_32(inst) (((inst) & 0x7F) == RVT_OPCODE_32)

// Check if instruction is a 64-bit RVT instruction
#define IS_RVT_64(inst) (((inst) & 0x7F) == RVT_OPCODE_64)

// Check if instruction is any RVT instruction
#define IS_RVT(inst) (IS_RVT_32(inst) || IS_RVT_64(inst))

// Get instruction length (0 = 32-bit, 1 = 64-bit)
#define RVT_INSN_LENGTH(inst) (IS_RVT_64(inst) ? 1 : 0)

//=============================================================================
// Common Field Extractors (32-bit format)
//=============================================================================

#define RVT32_OPCODE(inst)    RVT_BITS(inst, 6, 0)
#define RVT32_RD(inst)        RVT_BITS(inst, 11, 7)
#define RVT32_RS1(inst)       RVT_BITS(inst, 19, 15)
#define RVT32_RS2(inst)       RVT_BITS(inst, 24, 20)
#define RVT32_FUNCT3(inst)    RVT_BITS(inst, 31, 29)
#define RVT32_FUNCT5(inst)    RVT_BITS(inst, 31, 27)
#define RVT32_FUNCT7(inst)    RVT_BITS(inst, 31, 25)

//=============================================================================
// Common Field Extractors (64-bit format)
//=============================================================================

#define RVT64_OPCODE(inst)    RVT_BITS(inst, 6, 0)
#define RVT64_RD(inst)        RVT_BITS(inst, 11, 7)
#define RVT64_GROUP(inst)     RVT_BITS(inst, 14, 12)
#define RVT64_RS1(inst)       RVT_BITS(inst, 19, 15)
#define RVT64_RS2(inst)       RVT_BITS(inst, 24, 20)
#define RVT64_EXTENSION(inst) RVT_BITS(inst, 31, 25)

// For macro.elemwise
#define RVT64_ELEM_DTYPE(inst)    RVT_BITS(inst, 39, 32)
#define RVT64_ELEM_OP(inst)       RVT_BITS(inst, 45, 40)
#define RVT64_ELEM_BD(inst)       RVT_BITS(inst, 46, 46)

// For macro.convert
#define RVT64_CONVERT_SRCTYPE(inst) RVT_BITS(inst, 39, 32)
#define RVT64_CONVERT_DSTTYPE(inst) RVT_BITS(inst, 47, 40)

// For macro.sfu
#define RVT64_SFU_OP(inst)          RVT_BITS(inst, 39, 32)

// For macro.reduce
#define RVT64_REDUCE_DTYPE(inst)    RVT_BITS(inst, 39, 32)
#define RVT64_REDUCE_OP(inst)       RVT_BITS(inst, 45, 40)

//=============================================================================
// BASE Instructions (Unit = 0000)
//=============================================================================

// tx.base.tcsr.<unit>.read / tx.base.tcsr.<unit>.write
// Fields:
//   [6:0]   opcode = 0001011
//   [11:7]  rd (dest register for read, src register for write)
//   [12:12] re (1=read, 0=write)
//   [22:13] tcsr address
//   [26:23] unit id
//   [31:27] funct5 = 00000

#define RVT_BASE_TCSR_MASK       0xF800007FULL
#define RVT_BASE_TCSR_PATTERN    0x0000000BULL

#define IS_RVT_BASE_TCSR(inst) \
    (IS_RVT_32(inst) && RVT_MATCH(inst, RVT_BASE_TCSR_MASK, RVT_BASE_TCSR_PATTERN))

#define RVT_BASE_TCSR_RE(inst)       RVT_BITS(inst, 12, 12)
#define RVT_BASE_TCSR_ADDR(inst)     RVT_BITS(inst, 22, 13)
#define RVT_BASE_TCSR_UNIT(inst)     RVT_BITS(inst, 26, 23)
#define RVT_BASE_TCSR_FUNCT5(inst)   RVT_BITS(inst, 31, 27)

#define IS_RVT_BASE_TCSR_READ(inst)  (IS_RVT_BASE_TCSR(inst) && RVT_BASE_TCSR_RE(inst) == 1)
#define IS_RVT_BASE_TCSR_WRITE(inst) (IS_RVT_BASE_TCSR(inst) && RVT_BASE_TCSR_RE(inst) == 0)

// Unit IDs
#define RVT_UNIT_BASE     0b0000
#define RVT_UNIT_LSU      0b0001
#define RVT_UNIT_MAT      0b0010
#define RVT_UNIT_VEC      0b0011
#define RVT_UNIT_MACRO    0b0100

//=============================================================================
// LSU Instructions (Unit = 0001)
//=============================================================================

// tx.lsu.dma.[u|d]2[u|d][.g|.s|.sg]
// Fields:
//   [6:0]   opcode = 0001011
//   [11:7]  rd (dest ptr)
//   [12:12] dst (0=URAM, 1=DRAM)
//   [13:13] src (0=URAM, 1=DRAM)
//   [14:14] scatter table enable
//   [19:15] rs1 (src ptr)
//   [22:20] group (3 bits)
//   [23:23] gather table enable
//   [24:24] 0
//   [31:25] funct7 = 0000100

#define RVT_LSU_DMA_MASK        0xFF00007FULL
#define RVT_LSU_DMA_PATTERN     0x0800000BULL

#define IS_RVT_LSU_DMA(inst) \
    (IS_RVT_32(inst) && RVT_MATCH(inst, RVT_LSU_DMA_MASK, RVT_LSU_DMA_PATTERN))

#define RVT_LSU_DMA_DST(inst)   RVT_BITS(inst, 12, 12)
#define RVT_LSU_DMA_SRC(inst)   RVT_BITS(inst, 13, 13)
#define RVT_LSU_DMA_SCATTER(inst) RVT_BITS(inst, 14, 14)
#define RVT_LSU_DMA_GROUP(inst) RVT_BITS(inst, 22, 20)
#define RVT_LSU_DMA_GATHER(inst) RVT_BITS(inst, 23, 23)
#define RVT_LSU_DMA_FUNCT7(inst) RVT_BITS(inst, 31, 25)

// Direction variants
#define RVT_LSU_DMA_DIR_U2U 0x0  // src=0, dst=0
#define RVT_LSU_DMA_DIR_U2D 0x1  // src=0, dst=1
#define RVT_LSU_DMA_DIR_D2U 0x2  // src=1, dst=0
#define RVT_LSU_DMA_DIR_D2D 0x3  // src=1, dst=1

#define RVT_LSU_DMA_DIR(inst) ((RVT_LSU_DMA_SRC(inst) << 1) | RVT_LSU_DMA_DST(inst))

#define IS_RVT_LSU_DMA_U2U(inst) (IS_RVT_LSU_DMA(inst) && RVT_LSU_DMA_DIR(inst) == RVT_LSU_DMA_DIR_U2U)
#define IS_RVT_LSU_DMA_U2D(inst) (IS_RVT_LSU_DMA(inst) && RVT_LSU_DMA_DIR(inst) == RVT_LSU_DMA_DIR_U2D)
#define IS_RVT_LSU_DMA_D2U(inst) (IS_RVT_LSU_DMA(inst) && RVT_LSU_DMA_DIR(inst) == RVT_LSU_DMA_DIR_D2U)
#define IS_RVT_LSU_DMA_D2D(inst) (IS_RVT_LSU_DMA(inst) && RVT_LSU_DMA_DIR(inst) == RVT_LSU_DMA_DIR_D2D)

//=============================================================================
// MAT Instructions (Unit = 0010)
//=============================================================================

// tx.mat.mm / tx.mat.mma
// Fields:
//   [6:0]   opcode = 0001011
//   [11:7]  rd (C ptr)
//   [12:12] acc (0=mm, 1=mma)
//   [14:13] 0
//   [19:15] rs1 (A ptr)
//   [24:20] rs2 (B ptr)
//   [27:25] group (3 bits)
//   [28:28] 0
//   [31:29] funct3 = 001

#define RVT_MAT_MM_MASK         0xE000607FULL
#define RVT_MAT_MM_PATTERN      0x2000000BULL

#define IS_RVT_MAT_MM_FAMILY(inst) \
    (IS_RVT_32(inst) && \
     RVT_MATCH(inst, RVT_MAT_MM_MASK, RVT_MAT_MM_PATTERN))

#define RVT_MAT_ACC(inst)       RVT_BITS(inst, 12, 12)
#define RVT_MAT_MODE(inst)      RVT_BITS(inst, 14, 12)
#define RVT_MAT_GROUP(inst)     RVT_BITS(inst, 27, 25)
#define RVT_MAT_FUNCT3(inst)    RVT_BITS(inst, 31, 29)

#define IS_RVT_MAT_MM(inst)     (IS_RVT_MAT_MM_FAMILY(inst) && RVT_MAT_ACC(inst) == 0)
#define IS_RVT_MAT_MMA(inst)    (IS_RVT_MAT_MM_FAMILY(inst) && RVT_MAT_ACC(inst) == 1)

#define RVT_MAT_COMPUTE_MASK    0xE000007FULL
#define RVT_MAT_COMPUTE_PATTERN 0x2000000BULL

#define IS_RVT_MAT_COMPUTE_FAMILY(inst) \
    (IS_RVT_32(inst) && \
     RVT_MATCH(inst, RVT_MAT_COMPUTE_MASK, RVT_MAT_COMPUTE_PATTERN))

#define IS_RVT_MAT_CONV2D(inst)  (IS_RVT_MAT_COMPUTE_FAMILY(inst) && RVT_MAT_MODE(inst) == 2)
#define IS_RVT_MAT_CONV2DA(inst) (IS_RVT_MAT_COMPUTE_FAMILY(inst) && RVT_MAT_MODE(inst) == 3)

// tx.mat.setptr
// Fields:
//   [6:0]   opcode = 0001011
//   [11:7]  rd = 0
//   [14:12] mode = 100
//   [19:15] rs1 (A Ext ptr)
//   [24:20] rs2 (B Ext ptr)
//   [27:25] group (3 bits)
//   [28:28] 0
//   [31:29] funct3 = 001

#define RVT_MAT_SETPTR_MASK     0xE000707FULL
#define RVT_MAT_SETPTR_PATTERN  0x2000400BULL

#define IS_RVT_MAT_SETPTR(inst) \
    (IS_RVT_32(inst) && \
     RVT_MATCH(inst, RVT_MAT_SETPTR_MASK, RVT_MAT_SETPTR_PATTERN))

//=============================================================================
// MACRO Instructions (64-bit format)
//=============================================================================

// Extension field values
#define RVT_MACRO_EXT_ELEMWISE  0b0000000  // 0x00
#define RVT_MACRO_EXT_CONVERT   0b0000001  // 0x01
#define RVT_MACRO_EXT_SFU       0b0000010  // 0x02
#define RVT_MACRO_EXT_REDUCE    0b0000011  // 0x03

//-----------------------------------------------------------------------------
// tx.macro.elemwise.<op>.<type>
//-----------------------------------------------------------------------------

#define IS_RVT_MACRO_ELEMWISE(inst) \
    (IS_RVT_64(inst) && RVT64_EXTENSION(inst) == RVT_MACRO_EXT_ELEMWISE)

// Elemwise operation codes
#define RVT_ELEM_OP_ADD  0b000000
#define RVT_ELEM_OP_SUB  0b000001
#define RVT_ELEM_OP_MUL  0b000010

#define IS_RVT_MACRO_ELEMWISE_ADD(inst)  (IS_RVT_MACRO_ELEMWISE(inst) && RVT64_ELEM_OP(inst) == RVT_ELEM_OP_ADD)
#define IS_RVT_MACRO_ELEMWISE_SUB(inst)  (IS_RVT_MACRO_ELEMWISE(inst) && RVT64_ELEM_OP(inst) == RVT_ELEM_OP_SUB)
#define IS_RVT_MACRO_ELEMWISE_MUL(inst)  (IS_RVT_MACRO_ELEMWISE(inst) && RVT64_ELEM_OP(inst) == RVT_ELEM_OP_MUL)

//-----------------------------------------------------------------------------
// tx.macro.convert.<typeDst>.<typeSrc>
//-----------------------------------------------------------------------------

#define IS_RVT_MACRO_CONVERT(inst) \
    (IS_RVT_64(inst) && RVT64_EXTENSION(inst) == RVT_MACRO_EXT_CONVERT)

//-----------------------------------------------------------------------------
// tx.macro.sfu.<op>
//-----------------------------------------------------------------------------

#define IS_RVT_MACRO_SFU(inst) \
    (IS_RVT_64(inst) && RVT64_EXTENSION(inst) == RVT_MACRO_EXT_SFU)

//-----------------------------------------------------------------------------
// tx.macro.reduce.<op>.<type>
//-----------------------------------------------------------------------------

#define IS_RVT_MACRO_REDUCE(inst) \
    (IS_RVT_64(inst) && RVT64_EXTENSION(inst) == RVT_MACRO_EXT_REDUCE)

// Reduce operation codes
#define RVT_REDUCE_OP_ADD  0b000000
#define RVT_REDUCE_OP_MAX  0b000100
#define RVT_REDUCE_OP_MIN  0b000101

#define IS_RVT_MACRO_REDUCE_ADD(inst)  (IS_RVT_MACRO_REDUCE(inst) && RVT64_REDUCE_OP(inst) == RVT_REDUCE_OP_ADD)
#define IS_RVT_MACRO_REDUCE_MAX(inst)  (IS_RVT_MACRO_REDUCE(inst) && RVT64_REDUCE_OP(inst) == RVT_REDUCE_OP_MAX)
#define IS_RVT_MACRO_REDUCE_MIN(inst)  (IS_RVT_MACRO_REDUCE(inst) && RVT64_REDUCE_OP(inst) == RVT_REDUCE_OP_MIN)

//=============================================================================
// Data Type Encodings
//=============================================================================

#define RVT_TYPE_FP32        0b000000
#define RVT_TYPE_TF32        RVT_TYPE_FP32
#define RVT_TYPE_BF16        0b000010
#define RVT_TYPE_FP16        0b000011
#define RVT_TYPE_FP8_E4M3    0b000100
#define RVT_TYPE_FP8_E5M2    0b000101
#define RVT_TYPE_MX_FP8_E4M3 0b000110
#define RVT_TYPE_MX_FP8_E5M2 0b000111
#define RVT_TYPE_FP4_E2M1    0b001000
#define RVT_TYPE_MX_FP4_E2M1 0b001001
// Temporary encodings used by the current NeoMat RTL until the ISA table is
// extended with 4-bit formats and the RTL moves to the architectural S8/U8
// encodings below.
#define RVT_TYPE_RTL_S8      0b001010
#define RVT_TYPE_RTL_U8      0b001011
#define RVT_TYPE_S4          0b001100
#define RVT_TYPE_S32         0b010000
#define RVT_TYPE_U32         0b010001
#define RVT_TYPE_S16         0b010010
#define RVT_TYPE_U16         0b010011
#define RVT_TYPE_S8          0b010100
#define RVT_TYPE_U8          0b010101

// Deployed kernel headers also use the legacy 0x2x integer namespace. Matrix
// execution accepts those values as compatibility aliases; new software should
// use the ISA encodings above (or the temporary RTL values while they remain).
#define RVT_TYPE_COMPAT_S8   0b100100
#define RVT_TYPE_COMPAT_U8   0b100101

//=============================================================================
// Instruction Classification
//=============================================================================

// Check if instruction is from a specific category
#define IS_RVT_BASE(inst)  (IS_RVT_BASE_TCSR(inst))
#define IS_RVT_LSU(inst)   (IS_RVT_LSU_DMA(inst))
#define IS_RVT_MAT(inst)   (IS_RVT_MAT_COMPUTE_FAMILY(inst) || IS_RVT_MAT_SETPTR(inst))
#define IS_RVT_VEC(inst)   (IS_RVT_32(inst) && !IS_RVT_BASE(inst) && !IS_RVT_LSU(inst) && !IS_RVT_MAT(inst))
#define IS_RVT_MACRO(inst) (IS_RVT_64(inst))

// Top-level routing is intentionally broader than each unit's local decoder.
// A routed-but-illegal instruction must be reported by that unit instead of
// falling through to a different execution unit.
#define IS_RVT_TOP_MAT(inst) \
    (IS_RVT_32(inst) && RVT_BITS(inst, 31, 29) == 1 && \
     (RVT_BITS(inst, 14, 13) == 0 || RVT_BITS(inst, 14, 13) == 1 || \
      RVT_BITS(inst, 14, 12) == 4))
#define IS_RVT_TOP_VEC(inst) \
    (IS_RVT_32(inst) && !IS_RVT_BASE(inst) && !IS_RVT_LSU(inst) && \
     !IS_RVT_TOP_MAT(inst))

// 元素运算操作编码
#define TX_ELEM_ADD 0x00  // 0b000000
#define TX_ELEM_SUB 0x01  // 0b000001
#define TX_ELEM_MUL 0x02  // 0b000010
#define TX_ELEM_MAX 0x04  // 0b000100 - 浮点最大值
#define TX_ELEM_MIN 0x05  // 0b000101 - 浮点最小值

// SFU 操作编码
#define TX_SFU_LOG2 0   // 0b00000000 - ReLU
#define TX_SFU_EXP2 1   // 0b00000001 - 指数
#define TX_SFU_SIN  2   // 0b00000010 - 正弦
#define TX_SFU_COS  3   // 0b00000011 - 余弦
#define TX_SFU_RSQRT 4  // 0b00000100 - 平方根倒数
#define TX_SFU_RECIP 5  // 0b00000101 - 倒数

//=============================================================================
// Helper Functions (C++ inline functions for safer type handling)
//=============================================================================

//=============================================================================
// Helper Functions (C static inline functions for C compatibility)
//=============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// Instruction type checks
static inline int rvt_is_rvt(uint64_t inst) { return IS_RVT(inst); }
static inline int rvt_is_rvt32(uint32_t inst) { return IS_RVT_32(inst); }
static inline int rvt_is_rvt64(uint64_t inst) { return IS_RVT_64(inst); }

// Category checks
static inline int rvt_is_base(uint32_t inst) { return IS_RVT_BASE(inst); }
static inline int rvt_is_lsu(uint32_t inst) { return IS_RVT_LSU(inst); }
static inline int rvt_is_mat(uint32_t inst) { return IS_RVT_MAT(inst); }
static inline int rvt_is_vec(uint32_t inst) { return IS_RVT_VEC(inst); }
static inline int rvt_is_macro(uint64_t inst) { return IS_RVT_MACRO(inst); }

// Sync/Async checks
// static inline int rvt_is_sync(uint32_t inst) { return IS_RVT_SYNC(inst); }
// static inline int rvt_is_async(uint32_t inst) { return IS_RVT_ASYNC(inst); }

// Field extractors with type safety
static inline uint32_t rvt_bits(uint64_t inst, int hi, int lo) {
    return (uint32_t)RVT_BITS(inst, hi, lo);
}

#ifdef __cplusplus
} // extern "C"
#endif

//=============================================================================
// C++ Wrapper (optional, for type-safe C++ API)
//=============================================================================

#ifdef __cplusplus
namespace rvt {

// Instruction type checks
static inline bool is_rvt(uint64_t inst) { return rvt_is_rvt(inst) != 0; }
static inline bool is_rvt32(uint32_t inst) { return rvt_is_rvt32(inst) != 0; }
static inline bool is_rvt64(uint64_t inst) { return rvt_is_rvt64(inst) != 0; }

// Category checks
static inline bool is_base(uint32_t inst) { return rvt_is_base(inst) != 0; }
static inline bool is_lsu(uint32_t inst) { return rvt_is_lsu(inst) != 0; }
static inline bool is_mat(uint32_t inst) { return rvt_is_mat(inst) != 0; }
static inline bool is_vec(uint32_t inst) { return rvt_is_vec(inst) != 0; }
static inline bool is_macro(uint64_t inst) { return rvt_is_macro(inst) != 0; }

// Sync/Async checks
// static inline bool is_sync(uint32_t inst) { return rvt_is_sync(inst) != 0; }
// static inline bool is_async(uint32_t inst) { return rvt_is_async(inst) != 0; }

// Field extractors with type safety
static inline uint32_t bits(uint64_t inst, int hi, int lo) {
    return rvt_bits(inst, hi, lo);
}

} // namespace rvt
#endif // __cplusplus
