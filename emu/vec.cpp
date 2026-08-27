#include "common.hpp"
#include "rvt_decoder.hpp"
#include "swizzle.hpp"
#include "utils.hpp"
#include "vec.hpp"
#include "vec_convert.hpp"
#include "vec_mul24.hpp"
#include "vec_scalar.hpp"
#include "vec_shuffle.hpp"
#include "vec_unary.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>

namespace RVTNG {

namespace {

constexpr uint64_t VEC_CONTROL_DYNAMIC = 1ull << 40;
constexpr uint64_t VEC_CONTROL_VALID_MASK = (1ull << 41) - 1;
constexpr uint64_t VEC_URAM_SIZE = 1536ull * 1024ull;
constexpr uint64_t VEC_URAM_ADDRESS_MASK = (uint64_t{1} << 21) - 1;
constexpr uint64_t FLEXVEC_URAM_ALIGNMENT = 32;
constexpr uint64_t FLEXVEC_URAM_GROUP_BYTES = 128;
constexpr uint64_t VEC_URAM_CHANNEL_BYTES = 32;
constexpr size_t VEC_URAM_CHANNEL_COUNT = 8;
constexpr size_t VEC_LANES_PER_URAM_CHANNEL = 8;

struct vec_lane_t {
    std::array<uint32_t, 64> u32;
};

struct vec_thread_context_t {
    std::array<vec_lane_t, 16> vq0;
    std::array<vec_lane_t, 16> vq1;
    std::array<uint32_t, 2> head{};
    std::array<uint16_t, 2> valid{};
    std::array<uint64_t, 8> s{};
    vec_coord_t coord{};
    const vec_tcsr_t* tcsr = nullptr;
    uint8_t* uram = nullptr;
    vec_sparse_memory_t* sparse_memory = nullptr;
    uram_range_t uram_range{};
    interrupt_controller_t* interrupts = nullptr;
    size_t executed_insts = 0;
};

struct vec_fields_t {
    uint32_t raw = 0;
    uint8_t op6 = 0;
    uint8_t vb = 0;
    uint8_t vrs2 = 0;
    uint8_t vrs1 = 0;
    uint8_t vrs1_low = 0;
    uint8_t s2 = 0;
    uint8_t s1 = 0;
    uint8_t func8 = 0;
    uint8_t subfunc3 = 0;
    uint8_t mux_cmp_kind = 0;
    vec_unary_fields_t unary;
};

static inline vec_fields_t decode_vec_fields(uint32_t raw)
{
    vec_fields_t f;
    f.raw = raw;
    f.op6 = (raw >> 26) & 0b111111;
    f.vb = (raw >> 25) & 0b1;
    f.vrs2 = (raw >> 20) & 0b11111;
    f.vrs1 = (raw >> 15) & 0b11111;
    f.vrs1_low = (raw >> 7) & 0b11111;
    f.s2 = (raw >> 12) & 0b111;
    f.s1 = (raw >> 7) & 0b111;
    f.func8 = (raw >> 7) & 0xff;
    f.subfunc3 = (raw >> 12) & 0b111;
    f.mux_cmp_kind = (raw >> 10) & 0b11;
    f.unary = decode_vec_unary(raw);
    return f;
}

static inline bool is_vec_tcsr(const rvt_pack_t& inst)
{
    uint32_t unit = (inst.inst >> 23) & 0x1f;
    uint32_t tcsr_id = (inst.inst >> 13) & 0x3ff;
    return rvt::is_base(static_cast<uint32_t>(inst.inst))
        && (unit == RVT_UNIT_VEC ||
            (unit == RVT_UNIT_BASE &&
             (tcsr_id & 0x1f) == RVT_UNIT_VEC));
}

static inline bool is_vec_inst(uint64_t inst)
{
    return IS_RVT_TOP_VEC(inst);
}

static inline uint64_t mask_for_width(uint8_t width)
{
    if(width == 0) {
        return 0;
    }
    if(width >= 64) {
        return ~0ull;
    }
    return (1ull << width) - 1;
}

static inline vec_iterator_t* iterator_by_offset(vec_tcsr_t& tcsr, uint32_t id)
{
    if(id >= 9 && id <= 36) {
        uint32_t offset = id - 9;
        return &tcsr.s[offset / 7];
    }
    if(id >= 37 && id <= 64) {
        uint32_t offset = id - 37;
        return &tcsr.v[offset / 7];
    }
    return nullptr;
}

static inline vec_lane_t& write_vec(vec_thread_context_t& ctx, uint8_t bank)
{
    uint8_t b = bank & 0x1;
    ctx.head[b] = (ctx.head[b] + 1) & 0xf;
    ctx.valid[b] |= uint16_t{1} << ctx.head[b];
    return b ? ctx.vq1[ctx.head[b]] : ctx.vq0[ctx.head[b]];
}

static inline const vec_lane_t& read_vec(const vec_thread_context_t& ctx, uint8_t vrs)
{
    uint8_t bank = (vrs >> 4) & 0x1;
    uint8_t offset = vrs & 0xf;
    uint8_t idx = (ctx.head[bank] - offset) & 0xf;
    if((ctx.valid[bank] & (uint16_t{1} << idx)) == 0) {
        static const vec_lane_t zero{};
        return zero;
    }
    return bank ? ctx.vq1[idx] : ctx.vq0[idx];
}

static inline uint64_t normalize_uram_addr(uint64_t addr)
{
    return addr & VEC_URAM_ADDRESS_MASK;
}

static inline uint32_t read_u32(uint8_t* uram, uint64_t addr)
{
    uint32_t value = 0;
    addr = normalize_uram_addr(addr);
    addr = swizzle_addr(addr, 0);
    if(addr + sizeof(value) <= VEC_URAM_SIZE) {
        std::memcpy(&value, uram + addr, sizeof(value));
    }
    return value;
}

static inline void write_u32(uint8_t* uram, uint64_t addr, uint32_t value)
{
    addr = normalize_uram_addr(addr);
    addr = swizzle_addr(addr, 0);
    if(addr + sizeof(value) <= VEC_URAM_SIZE) {
        std::memcpy(uram + addr, &value, sizeof(value));
    }
}

static inline bool read_dense_vec(uint8_t* uram, uint64_t addr, vec_lane_t& value)
{
    addr = normalize_uram_addr(addr);
    if((addr & (sizeof(uint32_t) - 1)) != 0 || addr > VEC_URAM_SIZE - sizeof(value.u32)) {
        return false;
    }
    std::memcpy(value.u32.data(), uram + addr, sizeof(value.u32));
    return true;
}

static inline bool write_dense_vec(uint8_t* uram, uint64_t addr, const vec_lane_t& value)
{
    addr = normalize_uram_addr(addr);
    if((addr & (sizeof(uint32_t) - 1)) != 0 || addr > VEC_URAM_SIZE - sizeof(value.u32)) {
        return false;
    }
    std::memcpy(uram + addr, value.u32.data(), sizeof(value.u32));
    return true;
}

static inline uint64_t sign_extend(uint64_t value, uint8_t bits)
{
    uint64_t sign = 1ull << (bits - 1);
    return (value ^ sign) - sign;
}

static inline bool is_branch_inst(uint64_t inst)
{
    return ((inst >> 31) & 0x1) && (((inst >> 26) & 0x1f) == 0x1f);
}

static inline bool mask_lane(const vec_thread_context_t& ctx, uint8_t s2, size_t lane)
{
    return ((ctx.s[s2 & 0x7] >> lane) & 0x1) != 0;
}

static inline uint32_t read_linear(const vec_thread_context_t& ctx, bool sparse, uint64_t addr)
{
    return sparse ? ctx.sparse_memory->read_u32(addr) : read_u32(ctx.uram, addr);
}

static inline void write_linear(const vec_thread_context_t& ctx, bool sparse, uint64_t addr, uint32_t value)
{
    if(sparse) {
        ctx.sparse_memory->write_u32(addr, value);
    } else {
        write_u32(ctx.uram, addr, value);
    }
}

static bool check_flexvec_group(const vec_thread_context_t& ctx,
                                uint64_t address, unsigned group,
                                uint64_t access_bytes = FLEXVEC_URAM_GROUP_BYTES)
{
    if(ctx.interrupts == nullptr) {
        return true;
    }
    const interrupt_source_t range_source = group == 0
        ? interrupt_source_t::flexvec_uram_group_0_range
        : interrupt_source_t::flexvec_uram_group_1_range;
    const interrupt_source_t align_source = group == 0
        ? interrupt_source_t::flexvec_uram_group_0_align
        : interrupt_source_t::flexvec_uram_group_1_align;
    if((address & (FLEXVEC_URAM_ALIGNMENT - 1)) != 0) {
        ctx.interrupts->raise(align_source, static_cast<uint32_t>(address));
        return false;
    }
    const uint64_t physical_address = normalize_uram_addr(address);
    if(!ctx.uram_range.contains(physical_address,
                                access_bytes)) {
        ctx.interrupts->raise(range_source, static_cast<uint32_t>(address));
        return false;
    }
    return true;
}

static bool check_flexvec_strided_access(const vec_thread_context_t& ctx,
                                         uint64_t address, uint64_t stride)
{
    for(size_t channel = 0; channel < VEC_URAM_CHANNEL_COUNT; ++channel) {
        if(!check_flexvec_group(ctx, address + channel * stride,
                                channel / 4, VEC_URAM_CHANNEL_BYTES)) {
            return false;
        }
    }
    return true;
}

static bool check_flexvec_access(const vec_thread_context_t& ctx,
                                 uint64_t address, unsigned groups)
{
    for(unsigned group = 0; group < groups; ++group) {
        if(!check_flexvec_group(
               ctx, address + group * FLEXVEC_URAM_GROUP_BYTES, group)) {
            return false;
        }
    }
    return true;
}

static void execute_vec_regular_load(vec_thread_context_t& ctx, const vec_inst_t& inst, uint32_t raw)
{
    uint8_t vb = (raw >> 25) & 0x1;
    uint8_t sd = (raw >> 23) & 0x7;
    uint8_t subfunc = (raw >> 12) & 0x7;
    uint8_t sel = (raw >> 15) & 0x3;
    uint8_t s1 = (raw >> 7) & 0x7;
    uint64_t addr = inst.rd + vec_iter_value(vec_select_iter(*ctx.tcsr, true, sel), ctx.coord);
    bool global_to_scalar = ((raw >> 18) & 0x1f) == 0 && ((raw >> 17) & 0x1) != 0 && ((raw >> 12) & 0x1f) == 0;
    bool to_scalar = subfunc == 0x6 || subfunc == 0x7 || global_to_scalar;
    bool sparse = subfunc == 0x1 || subfunc == 0x3 || subfunc == 0x7;
    bool from_scalar = subfunc == 0x4;
    bool from_gpr = subfunc == 0x5;
    bool broadcast = subfunc == 0x2 || subfunc == 0x3 || from_scalar || from_gpr;

    if(to_scalar) {
        if(global_to_scalar) {
            ctx.s[sd] = inst.rd;
        } else {
            if(!sparse && !check_flexvec_access(ctx, addr, 1)) {
                ctx.s[sd] = 0;
                return;
            }
            ctx.s[sd] = static_cast<uint64_t>(read_linear(ctx, sparse, addr))
                | (static_cast<uint64_t>(read_linear(ctx, sparse, addr + sizeof(uint32_t))) << 32);
        }
        return;
    }

    vec_lane_t& dst = write_vec(ctx, vb);
    if(from_scalar) {
        dst.u32.fill(static_cast<uint32_t>(ctx.s[s1]));
        return;
    }
    if(from_gpr) {
        dst.u32.fill(static_cast<uint32_t>(inst.rd));
        return;
    }
    if(broadcast) {
        if(!sparse && !check_flexvec_access(ctx, addr, 1)) {
            dst.u32.fill(0);
            return;
        }
        dst.u32.fill(read_linear(ctx, sparse, addr));
        return;
    }
    if(!sparse) {
        const uint64_t stride = ctx.tcsr->uramChannelStride[sel];
        if(!check_flexvec_strided_access(ctx, addr, stride)) {
            dst.u32.fill(0);
            return;
        }
        if(stride == VEC_URAM_CHANNEL_BYTES &&
           read_dense_vec(ctx.uram, addr, dst)) {
            return;
        }
        for(size_t channel = 0; channel < VEC_URAM_CHANNEL_COUNT; ++channel) {
            for(size_t lane = 0; lane < VEC_LANES_PER_URAM_CHANNEL; ++lane) {
                const size_t dst_lane =
                    channel * VEC_LANES_PER_URAM_CHANNEL + lane;
                dst.u32[dst_lane] = read_u32(
                    ctx.uram, addr + channel * stride + lane * sizeof(uint32_t));
            }
        }
        return;
    }
    for(size_t lane = 0; lane < dst.u32.size(); lane++) {
        dst.u32[lane] = read_linear(ctx, true, addr + lane * sizeof(uint32_t));
    }
}

static void execute_vec_regular_masked_load(vec_thread_context_t& ctx, const vec_inst_t& inst, uint32_t raw)
{
    uint8_t vb = (raw >> 25) & 0x1;
    uint8_t subfunc = (raw >> 20) & 0x1f;
    uint8_t vrs1 = (raw >> 15) & 0x1f;
    uint8_t s2 = (raw >> 12) & 0x7;
    uint64_t addr = inst.rd;
    bool sparse = subfunc == 0x03 || subfunc == 0x07;
    bool from_scalar = subfunc == 0x0a;
    bool from_gpr = subfunc == 0x09;
    bool broadcast = subfunc == 0x05 || subfunc == 0x07 || from_scalar || from_gpr;
    const vec_lane_t& fallback = read_vec(ctx, vrs1);
    vec_lane_t& dst = write_vec(ctx, vb);

    if(!sparse && !from_scalar && !from_gpr &&
       !check_flexvec_access(ctx, addr, broadcast ? 1 : 2)) {
        dst = fallback;
        return;
    }

    for(size_t lane = 0; lane < dst.u32.size(); lane++) {
        if(!mask_lane(ctx, s2, lane)) {
            dst.u32[lane] = fallback.u32[lane];
            continue;
        }
        if(from_scalar) {
            dst.u32[lane] = static_cast<uint32_t>(ctx.s[(raw >> 7) & 0x7]);
        } else if(from_gpr) {
            dst.u32[lane] = static_cast<uint32_t>(inst.rd);
        } else {
            uint64_t lane_addr = broadcast ? addr : addr + lane * sizeof(uint32_t);
            dst.u32[lane] = read_linear(ctx, sparse, lane_addr);
        }
    }
}

static void execute_vec_regular_store(vec_thread_context_t& ctx, const vec_inst_t& inst, uint32_t raw)
{
    uint8_t sel = (raw >> 20) & 0x3;
    uint8_t vrs1 = (raw >> 15) & 0x1f;
    uint8_t subfunc = (raw >> 12) & 0x7;
    uint64_t addr = inst.rd + vec_iter_value(vec_select_iter(*ctx.tcsr, true, sel), ctx.coord);
    bool sparse = (subfunc & 0x1) != 0;
    bool scalar_source = (subfunc & 0x2) != 0;
    const vec_lane_t& src = read_vec(ctx, vrs1);

    if(scalar_source) {
        if(!sparse && !check_flexvec_access(ctx, addr, 1)) {
            return;
        }
        const uint64_t value = ctx.s[vrs1 & 0x7];
        write_linear(ctx, sparse, addr, static_cast<uint32_t>(value));
        write_linear(ctx, sparse, addr + sizeof(uint32_t), static_cast<uint32_t>(value >> 32));
        return;
    }

    if(!sparse) {
        const uint64_t stride = ctx.tcsr->uramChannelStride[sel];
        if(!check_flexvec_strided_access(ctx, addr, stride)) {
            return;
        }
        if(stride == VEC_URAM_CHANNEL_BYTES &&
           write_dense_vec(ctx.uram, addr, src)) {
            return;
        }
        for(size_t channel = 0; channel < VEC_URAM_CHANNEL_COUNT; ++channel) {
            for(size_t lane = 0; lane < VEC_LANES_PER_URAM_CHANNEL; ++lane) {
                const size_t src_lane =
                    channel * VEC_LANES_PER_URAM_CHANNEL + lane;
                write_u32(ctx.uram,
                          addr + channel * stride + lane * sizeof(uint32_t),
                          src.u32[src_lane]);
            }
        }
        return;
    }
    for(size_t lane = 0; lane < src.u32.size(); lane++) {
        write_linear(ctx, true, addr + lane * sizeof(uint32_t), src.u32[lane]);
    }
}

static void execute_vec_regular_masked_store(vec_thread_context_t& ctx, const vec_inst_t& inst, uint32_t raw)
{
    uint8_t sel = (raw >> 20) & 0x3;
    uint8_t vrs1 = (raw >> 15) & 0x1f;
    uint8_t s2 = (raw >> 12) & 0x7;
    bool sparse = ((raw >> 22) & 0xf) == 0x2;
    uint64_t addr = inst.rd + vec_iter_value(vec_select_iter(*ctx.tcsr, true, sel), ctx.coord);
    const vec_lane_t& src = read_vec(ctx, vrs1);

    if(!sparse && !check_flexvec_access(ctx, addr, 2)) {
        return;
    }

    for(size_t lane = 0; lane < src.u32.size(); lane++) {
        if(mask_lane(ctx, s2, lane)) {
            write_linear(ctx, sparse, addr + lane * sizeof(uint32_t), src.u32[lane]);
        }
    }
}

static void execute_vec_sparse_indexed(vec_thread_context_t& ctx, const vec_inst_t& inst, uint32_t raw)
{
    uint8_t op = (raw >> 26) & 0xf;
    uint8_t vb = (raw >> 25) & 0x1;
    uint8_t vrs2 = (raw >> 20) & 0x1f;
    uint8_t vrs1 = (raw >> 15) & 0x1f;
    uint8_t s2 = (raw >> 12) & 0x7;
    const vec_lane_t& index = read_vec(ctx, vrs2);
    const vec_lane_t& value = read_vec(ctx, vrs1);

    if(op == 0x2 || op >= 0x4) {
        vec_lane_t& dst = write_vec(ctx, vb);
        for(size_t lane = 0; lane < dst.u32.size(); lane++) {
            if(!mask_lane(ctx, s2, lane)) {
                dst.u32[lane] = value.u32[lane];
                continue;
            }
            uint64_t addr = inst.rd + index.u32[lane];
            if(op == 0x2) {
                dst.u32[lane] = ctx.sparse_memory->read_u32(addr);
            } else {
                dst.u32[lane] = ctx.sparse_memory->atomic_update_u32(addr, value.u32[lane], op & 0x3);
            }
        }
        return;
    }

    if(op == 0x3) {
        for(size_t lane = 0; lane < value.u32.size(); lane++) {
            if(mask_lane(ctx, s2, lane)) {
                ctx.sparse_memory->write_u32(inst.rd + index.u32[lane], value.u32[lane]);
            }
        }
        return;
    }

    debugf("VEC: unsupported sparse indexed op 0x%x\n", op);
}

static void execute_vec_sparse_dyn(vec_thread_context_t& ctx, const vec_inst_t& inst, uint32_t raw)
{
    (void)inst;
    uint8_t op = (((raw >> 26) & 0x1) << 1) | ((raw >> 10) & 0x1);
    uint8_t vb = (raw >> 25) & 0x1;
    uint8_t vrs2 = (raw >> 20) & 0x1f;
    uint8_t vrs1 = (raw >> 15) & 0x1f;
    uint8_t s2 = (raw >> 12) & 0x7;
    uint8_t s1 = (raw >> 7) & 0x7;
    const vec_lane_t& index = read_vec(ctx, vrs2);
    const vec_lane_t& value = read_vec(ctx, vrs1);
    uint64_t base = ctx.s[s1];

    if(((raw >> 26) & 0x1) == 0 && ((raw >> 10) & 0x3) == 0x1 && ((raw >> 25) & 0x1) == 0) {
        for(size_t lane = 0; lane < value.u32.size(); lane++) {
            if(mask_lane(ctx, s2, lane)) {
                ctx.sparse_memory->write_u32(base + index.u32[lane], value.u32[lane]);
            }
        }
        return;
    }

    vec_lane_t& dst = write_vec(ctx, vb);
    for(size_t lane = 0; lane < dst.u32.size(); lane++) {
        if(!mask_lane(ctx, s2, lane)) {
            dst.u32[lane] = value.u32[lane];
            continue;
        }
        uint64_t addr = base + index.u32[lane];
        if(((raw >> 26) & 0x1) == 0 && ((raw >> 10) & 0x3) == 0) {
            dst.u32[lane] = ctx.sparse_memory->read_u32(addr);
        } else {
            dst.u32[lane] = ctx.sparse_memory->atomic_update_u32(addr, value.u32[lane], op);
        }
    }
}

enum class shuffle_op_t {
    xor_lane,
    ror_lane,
    broadcast_lane,
    compress_lane
};

enum class v_shift_op_t {
    sll,
    sra,
    srl
};

enum class vv_int32_op_t {
    rol,
    sll,
    sra,
    srl,
    add,
    sub,
    maxu,
    minu,
    max,
    min,
    bitwise
};

enum class vv_fp32_op_t {
    add,
    sub,
    mul,
    max,
    min
};

static inline bool fp32_is_nan(uint32_t value)
{
    return (value & 0x7fffffffu) > 0x7f800000u;
}

static inline bool fp32_ordered_less(uint32_t lhs, uint32_t rhs)
{
    if(fp32_is_nan(lhs) || fp32_is_nan(rhs)) {
        return false;
    }

    const uint32_t lhs_abs = lhs & 0x7fffffffu;
    const uint32_t rhs_abs = rhs & 0x7fffffffu;
    if((lhs_abs | rhs_abs) == 0) {
        return false;
    }

    const bool lhs_negative = (lhs >> 31) != 0;
    const bool rhs_negative = (rhs >> 31) != 0;
    if(lhs_negative != rhs_negative) {
        return lhs_negative;
    }
    return lhs_negative ? lhs > rhs : lhs < rhs;
}

enum class cmp_op_t {
    int32,
    fp32
};

template <shuffle_op_t Op, bool Dynamic>
static void execute_vec_shuffle(vec_thread_context_t& ctx, const vec_inst_t& inst, const vec_fields_t& f)
{
    uint64_t param = Dynamic ? ctx.s[f.s1] : inst.rd;
    const vec_lane_t& src = read_vec(ctx, f.vrs1);

    if constexpr(Op == shuffle_op_t::compress_lane) {
        const auto result = execute_vec_shuffle_compress(src.u32, param);
        write_vec(ctx, f.vb).u32 = result;
    } else {
        vec_lane_t& dst = write_vec(ctx, f.vb);
        for(size_t lane = 0; lane < dst.u32.size(); lane++) {
            size_t idx;
            if constexpr(Op == shuffle_op_t::xor_lane) {
                idx = (lane ^ param) & 0x3f;
            } else if constexpr(Op == shuffle_op_t::ror_lane) {
                idx = (lane + param) & 0x3f;
            } else {
                idx = param & 0x3f;
            }
            dst.u32[lane] = src.u32[idx];
        }
    }
}

template <v_shift_op_t Op>
static void execute_vec_v_shift(vec_thread_context_t& ctx, const vec_fields_t& f)
{
    const vec_lane_t& src = read_vec(ctx, f.vrs1_low);
    vec_lane_t& dst = write_vec(ctx, f.vb);

    for(size_t lane = 0; lane < dst.u32.size(); lane++) {
        if constexpr(Op == v_shift_op_t::sll) {
            dst.u32[lane] = src.u32[lane] << f.unary.shift_amount;
        } else if constexpr(Op == v_shift_op_t::sra) {
            dst.u32[lane] = static_cast<uint32_t>(
                static_cast<int32_t>(src.u32[lane]) >> f.unary.shift_amount);
        } else {
            dst.u32[lane] = src.u32[lane] >> f.unary.shift_amount;
        }
    }
}

static void execute_vec_v_sfu(vec_thread_context_t& ctx, const vec_fields_t& f)
{
    const vec_lane_t& src = read_vec(ctx, f.vrs1_low);
    vec_lane_t& dst = write_vec(ctx, f.vb);
    for(size_t lane = 0; lane < dst.u32.size(); lane++) {
        dst.u32[lane] = execute_vec_sfu_lane(src.u32[lane], f.unary.op);
    }
}

static void execute_vec_v_convert(vec_thread_context_t& ctx, const vec_fields_t& f)
{
    const vec_lane_t& src = read_vec(ctx, f.vrs1_low);
    write_vec(ctx, f.vb).u32 = execute_vec_convert(
        src.u32, f.unary.op, f.unary.rounding_mode);
}

static bool cmp_predicate(uint8_t pred, int relation)
{
    if(relation < 0) {
        return (pred & 0x4) != 0;
    }
    if(relation == 0) {
        return (pred & 0x2) != 0;
    }
    return (pred & 0x1) != 0;
}

static void execute_vec_mux2(vec_thread_context_t& ctx, const vec_fields_t& f)
{
    const vec_lane_t& a = read_vec(ctx, f.vrs1);
    const vec_lane_t& b = read_vec(ctx, f.vrs2);
    vec_lane_t& dst = write_vec(ctx, f.vb);
    uint64_t mask = ctx.s[f.s2];

    for(size_t lane = 0; lane < dst.u32.size(); lane++) {
        dst.u32[lane] = ((mask >> lane) & 0x1) ? b.u32[lane] : a.u32[lane];
    }
}

template <cmp_op_t Op>
static void execute_vec_cmp(vec_thread_context_t& ctx, const vec_fields_t& f)
{
    const vec_lane_t& a = read_vec(ctx, f.vrs1);
    const vec_lane_t& b = read_vec(ctx, f.vrs2);
    uint64_t mask = 0;

    for(size_t lane = 0; lane < a.u32.size(); lane++) {
        int relation = 0;
        if constexpr(Op == cmp_op_t::int32) {
            int32_t lhs = static_cast<int32_t>(a.u32[lane]);
            int32_t rhs = static_cast<int32_t>(b.u32[lane]);
            relation = (lhs < rhs) ? -1 : ((lhs > rhs) ? 1 : 0);
        } else {
            float lhs;
            float rhs;
            std::memcpy(&lhs, &a.u32[lane], sizeof(lhs));
            std::memcpy(&rhs, &b.u32[lane], sizeof(rhs));
            relation = (lhs < rhs) ? -1 : ((lhs > rhs) ? 1 : 0);
        }
        if(cmp_predicate(f.s1, relation)) {
            mask |= 1ull << lane;
        }
    }
    ctx.s[f.s2] = mask;
}

template <vv_int32_op_t Op>
static void execute_vec_vv_int32(vec_thread_context_t& ctx, const vec_fields_t& f)
{
    const vec_lane_t& a = read_vec(ctx, f.vrs1);
    const vec_lane_t& b = read_vec(ctx, f.vrs2);
    vec_lane_t& dst = write_vec(ctx, f.vb);

    for(size_t lane = 0; lane < dst.u32.size(); lane++) {
        uint32_t shamt = b.u32[lane] & 31;
        if constexpr(Op == vv_int32_op_t::rol) {
            dst.u32[lane] = (a.u32[lane] << shamt) | (a.u32[lane] >> ((32 - shamt) & 31));
        } else if constexpr(Op == vv_int32_op_t::sll) {
            dst.u32[lane] = a.u32[lane] << shamt;
        } else if constexpr(Op == vv_int32_op_t::sra) {
            dst.u32[lane] = static_cast<uint32_t>(static_cast<int32_t>(a.u32[lane]) >> shamt);
        } else if constexpr(Op == vv_int32_op_t::srl) {
            dst.u32[lane] = a.u32[lane] >> shamt;
        } else if constexpr(Op == vv_int32_op_t::add) {
            dst.u32[lane] = a.u32[lane] + b.u32[lane];
        } else if constexpr(Op == vv_int32_op_t::sub) {
            dst.u32[lane] = a.u32[lane] - b.u32[lane];
        } else if constexpr(Op == vv_int32_op_t::maxu) {
            dst.u32[lane] = a.u32[lane] > b.u32[lane]
                ? a.u32[lane] : b.u32[lane];
        } else if constexpr(Op == vv_int32_op_t::minu) {
            dst.u32[lane] = a.u32[lane] < b.u32[lane]
                ? a.u32[lane] : b.u32[lane];
        } else if constexpr(Op == vv_int32_op_t::max) {
            dst.u32[lane] = static_cast<int32_t>(a.u32[lane])
                    > static_cast<int32_t>(b.u32[lane])
                ? a.u32[lane] : b.u32[lane];
        } else if constexpr(Op == vv_int32_op_t::min) {
            dst.u32[lane] = static_cast<int32_t>(a.u32[lane])
                    < static_cast<int32_t>(b.u32[lane])
                ? a.u32[lane] : b.u32[lane];
        } else {
            dst.u32[lane] = execute_vec_bitwise_lut<uint32_t>(
                a.u32[lane], b.u32[lane], vec_vv_int32_logic_lut(f.func8));
        }
    }
}

static void execute_vec_vv_mul24(vec_thread_context_t& ctx, const vec_fields_t& f)
{
    const vec_lane_t& a = read_vec(ctx, f.vrs1);
    const vec_lane_t& b = read_vec(ctx, f.vrs2);
    vec_lane_t& dst = write_vec(ctx, f.vb);
    for(size_t lane = 0; lane < dst.u32.size(); lane++) {
        dst.u32[lane] = execute_vec_mul24_lane(
            f.func8, a.u32[lane], b.u32[lane]);
    }
}

template <vv_fp32_op_t Op>
static void execute_vec_vv_fp32(vec_thread_context_t& ctx, const vec_fields_t& f)
{
    const vec_lane_t& a = read_vec(ctx, f.vrs1);
    const vec_lane_t& b = read_vec(ctx, f.vrs2);
    vec_lane_t& dst = write_vec(ctx, f.vb);

    for(size_t lane = 0; lane < dst.u32.size(); lane++) {
        if constexpr(Op == vv_fp32_op_t::max) {
            dst.u32[lane] = fp32_ordered_less(b.u32[lane], a.u32[lane])
                ? a.u32[lane] : b.u32[lane];
            continue;
        } else if constexpr(Op == vv_fp32_op_t::min) {
            dst.u32[lane] = fp32_ordered_less(a.u32[lane], b.u32[lane])
                ? a.u32[lane] : b.u32[lane];
            continue;
        }

        float lhs;
        float rhs;
        float out;
        std::memcpy(&lhs, &a.u32[lane], sizeof(lhs));
        std::memcpy(&rhs, &b.u32[lane], sizeof(rhs));
        if constexpr(Op == vv_fp32_op_t::add) {
            out = lhs + rhs;
        } else if constexpr(Op == vv_fp32_op_t::sub) {
            out = lhs - rhs;
        } else {
            out = lhs * rhs;
        }
        std::memcpy(&dst.u32[lane], &out, sizeof(out));
    }
}

static void execute_vec_scalar(vec_thread_context_t& ctx, const vec_inst_t& inst)
{
    const uint32_t raw = static_cast<uint32_t>(inst.inst);
    const vec_scalar_fields_t fields = decode_vec_scalar(raw);
    const uint64_t src1 = ctx.s[fields.s1];
    const uint64_t src2 = ctx.s[fields.s2];

    if(vec_scalar_is_int32(fields.op)) {
        ctx.s[fields.sd] = execute_vec_scalar_int32(
            fields.op, static_cast<uint32_t>(src1), static_cast<uint32_t>(src2));
        return;
    }

    switch(fields.op) {
    case vec_scalar_op_t::get_dim:
        ctx.s[fields.sd] = vec_outer_coord_value(ctx.coord, fields.selector);
        return;
    case vec_scalar_op_t::get_s:
        ctx.s[fields.sd] = vec_iter_value(
            vec_select_iter(*ctx.tcsr, false, fields.selector), ctx.coord);
        return;
    case vec_scalar_op_t::get_v:
        ctx.s[fields.sd] = vec_iter_value(
            vec_select_iter(*ctx.tcsr, true, fields.selector), ctx.coord);
        return;
    case vec_scalar_op_t::bitwise:
        ctx.s[fields.sd] = execute_vec_bitwise_lut<uint64_t>(src1, src2, fields.logic_lut);
        return;
    default:
        debugf("VEC: unsupported scalar instruction 0x%08x\n", raw);
        return;
    }
}

static bool execute_vec_branch(vec_thread_context_t& ctx, const vec_inst_t& inst, uint64_t pc, uint64_t program_size, uint64_t& next_pc)
{
    const uint32_t raw = static_cast<uint32_t>(inst.inst);
    const vec_scalar_fields_t fields = decode_vec_scalar(raw);
    const bool take = execute_vec_scalar_branch(
        fields.op, ctx.s[fields.s1], ctx.s[fields.s2]);

    if(!take) {
        next_pc = pc + 1;
        return true;
    }

    const int64_t target = vec_scalar_branch_target(pc, fields.branch_offset);
    if(target < 0 || static_cast<uint64_t>(target) >= program_size) {
        debugf("VEC: branch target out of range %ld\n", target);
        return false;
    }
    next_pc = static_cast<uint64_t>(target);
    return true;
}

static void execute_one_vec_inst(vec_thread_context_t& ctx, const vec_inst_t& inst)
{
    uint32_t raw = static_cast<uint32_t>(inst.inst);
    vec_fields_t f = decode_vec_fields(raw);

    switch(f.op6) {
    case 0b010000: {
        const uint8_t subfunc5 = (raw >> 20) & 0x1f;
        const bool masked = subfunc5 == 0x01 || subfunc5 == 0x03 ||
            subfunc5 == 0x05 || subfunc5 == 0x07 ||
            subfunc5 == 0x09 || subfunc5 == 0x0a;
        if(masked) {
            execute_vec_regular_masked_load(ctx, inst, raw);
        } else {
            execute_vec_regular_load(ctx, inst, raw);
        }
        return;
    }
    case 0b010001: {
        const uint8_t subfunc4 = (raw >> 22) & 0xf;
        if(subfunc4 == 0x1 || subfunc4 == 0x2) {
            execute_vec_regular_masked_store(ctx, inst, raw);
        } else {
            execute_vec_regular_store(ctx, inst, raw);
        }
        return;
    }
    case 0b010010:
    case 0b010011:
    case 0b010100:
    case 0b010101:
    case 0b010110:
    case 0b010111:
        execute_vec_sparse_indexed(ctx, inst, raw);
        return;
    case 0b011000:
    case 0b011001:
        execute_vec_sparse_dyn(ctx, inst, raw);
        return;
    case 0b011111:
        switch(f.subfunc3) {
        case 0b000:
            execute_vec_shuffle<shuffle_op_t::xor_lane, false>(ctx, inst, f);
            return;
        case 0b001:
            execute_vec_shuffle<shuffle_op_t::xor_lane, true>(ctx, inst, f);
            return;
        case 0b010:
            execute_vec_shuffle<shuffle_op_t::ror_lane, false>(ctx, inst, f);
            return;
        case 0b011:
            execute_vec_shuffle<shuffle_op_t::ror_lane, true>(ctx, inst, f);
            return;
        case 0b100:
            execute_vec_shuffle<shuffle_op_t::broadcast_lane, false>(ctx, inst, f);
            return;
        case 0b101:
            execute_vec_shuffle<shuffle_op_t::broadcast_lane, true>(ctx, inst, f);
            return;
        case 0b110:
            execute_vec_shuffle<shuffle_op_t::compress_lane, false>(ctx, inst, f);
            return;
        case 0b111:
            execute_vec_shuffle<shuffle_op_t::compress_lane, true>(ctx, inst, f);
            return;
        default:
            debugf("VEC: unsupported shuffle subfunc 0x%x\n", f.subfunc3);
            return;
        }
    case 0b100000:
        if(f.unary.kind == vec_unary_kind_t::shift) {
            switch(f.unary.op) {
            case 0b001:
                execute_vec_v_shift<v_shift_op_t::sll>(ctx, f);
                return;
            case 0b010:
                execute_vec_v_shift<v_shift_op_t::sra>(ctx, f);
                return;
            case 0b011:
                execute_vec_v_shift<v_shift_op_t::srl>(ctx, f);
                return;
            default:
                break;
            }
        }
        if(f.unary.kind == vec_unary_kind_t::sfu) {
            if(vec_sfu_op_supported(f.unary.op)) {
                execute_vec_v_sfu(ctx, f);
            } else {
                debugf("VEC: unsupported SFU op 0x%x\n", f.unary.op);
            }
            return;
        }
        if(f.unary.kind == vec_unary_kind_t::convert) {
            if(!vec_convert_op_supported(f.unary.op)) {
                debugf("VEC: convert op 0x%x is not implemented\n", f.unary.op);
                return;
            }
            if(!vec_rounding_mode_supported(f.unary.rounding_mode)) {
                debugf("VEC: unsupported convert rounding mode 0x%x\n",
                       f.unary.rounding_mode);
                return;
            }
            execute_vec_v_convert(ctx, f);
            return;
        }
        debugf("VEC: unsupported unary encoding 0x%08x\n", raw);
        return;
    case 0b100001:
        switch(f.func8) {
        case 0x00:
            execute_vec_vv_int32<vv_int32_op_t::rol>(ctx, f);
            return;
        case 0x01:
            execute_vec_vv_int32<vv_int32_op_t::sll>(ctx, f);
            return;
        case 0x02:
            execute_vec_vv_int32<vv_int32_op_t::sra>(ctx, f);
            return;
        case 0x03:
            execute_vec_vv_int32<vv_int32_op_t::srl>(ctx, f);
            return;
        case 0x04:
            execute_vec_vv_int32<vv_int32_op_t::add>(ctx, f);
            return;
        case 0x05:
            execute_vec_vv_int32<vv_int32_op_t::sub>(ctx, f);
            return;
        case 0x06:
        case 0x07:
            execute_vec_vv_mul24(ctx, f);
            return;
        case 0x08:
            execute_vec_vv_int32<vv_int32_op_t::maxu>(ctx, f);
            return;
        case 0x09:
            execute_vec_vv_int32<vv_int32_op_t::minu>(ctx, f);
            return;
        case 0x0a:
            execute_vec_vv_int32<vv_int32_op_t::max>(ctx, f);
            return;
        case 0x0b:
            execute_vec_vv_int32<vv_int32_op_t::min>(ctx, f);
            return;
        default:
            if(vec_vv_int32_is_bitwise(f.func8)) {
                execute_vec_vv_int32<vv_int32_op_t::bitwise>(ctx, f);
                return;
            }
            debugf("VEC: unsupported vv.int32 func8 0x%x\n", f.func8);
            return;
        }
    case 0b100010:
        switch(f.func8 >> 3) {
        case 0b000:
            execute_vec_vv_fp32<vv_fp32_op_t::add>(ctx, f);
            return;
        case 0b001:
            execute_vec_vv_fp32<vv_fp32_op_t::sub>(ctx, f);
            return;
        case 0b010:
            execute_vec_vv_fp32<vv_fp32_op_t::mul>(ctx, f);
            return;
        case 0b00100:
            execute_vec_vv_fp32<vv_fp32_op_t::max>(ctx, f);
            return;
        case 0b00101:
            execute_vec_vv_fp32<vv_fp32_op_t::min>(ctx, f);
            return;
        default:
            debugf("VEC: unsupported vv.fp32 op 0x%x\n", f.func8 >> 3);
            return;
        }
    case 0b100011:
        switch(f.mux_cmp_kind) {
        case 0b00:
            execute_vec_mux2(ctx, f);
            return;
        case 0b01:
            execute_vec_cmp<cmp_op_t::int32>(ctx, f);
            return;
        case 0b10:
            execute_vec_cmp<cmp_op_t::fp32>(ctx, f);
            return;
        default:
            debugf("VEC: unsupported mux/cmp kind 0x%x\n", f.mux_cmp_kind);
            return;
        }
    case 0b111110:
        execute_vec_scalar(ctx, inst);
        return;
    default:
        debugf("VEC: unsupported instruction 0x%08x\n", raw);
        return;
    }
}

static const vec_inst_t& task_inst(const vec_task_req_t& task,
                                   const vec_inst_t* inst_queue,
                                   uint64_t pc)
{
    return inst_queue[(task.start_slot + pc) % VEC_INST_QUEUE_CAPACITY];
}

static uint64_t execute_mode1(vec_thread_context_t& ctx,
                              const vec_task_req_t& task,
                              const vec_inst_t* inst_queue)
{
    const auto& c = task.control;
    auto execute_range = [&](uint8_t begin, uint8_t end) {
        for(uint64_t pc = begin; pc < end && pc < task.inst_count; pc++) {
            const vec_inst_t& inst = task_inst(task, inst_queue, pc);
            if(is_branch_inst(inst.inst)) {
                debugf("VEC: branch ignored in mode1 at pc %lu\n", pc);
                continue;
            }
            execute_one_vec_inst(ctx, inst);
            ctx.executed_insts++;
        }
    };

    execute_range(0, c.prologue0_end);
    for(ctx.coord.d0 = 0; ctx.coord.d0 <= task.tcsr_snapshot.dim0; ctx.coord.d0++) {
        execute_range(c.prologue0_end, c.prologue1_end);
        for(ctx.coord.d1 = 0; ctx.coord.d1 <= task.tcsr_snapshot.dim1; ctx.coord.d1++) {
            execute_range(c.prologue1_end, c.loop_end);
        }
        execute_range(c.loop_end, c.epilogue1_end);
    }
    execute_range(c.epilogue1_end, c.epilogue0_end);
    return ctx.executed_insts;
}

static uint64_t execute_mode2(vec_thread_context_t& ctx,
                              const vec_task_req_t& task,
                              const vec_inst_t* inst_queue)
{
    constexpr uint64_t WATCHDOG_PLACEHOLDER_LIMIT = 1ull << 24;
    uint64_t pc = 0;
    while(pc < task.inst_count && ctx.executed_insts < WATCHDOG_PLACEHOLDER_LIMIT) {
        uint64_t next_pc = pc + 1;
        const vec_inst_t& inst = task_inst(task, inst_queue, pc);
        if(is_branch_inst(inst.inst)) {
            if(!execute_vec_branch(ctx, inst, pc, task.inst_count, next_pc)) {
                break;
            }
        } else {
            execute_one_vec_inst(ctx, inst);
        }
        ctx.executed_insts++;
        pc = next_pc;
    }
    if(ctx.executed_insts >= WATCHDOG_PLACEHOLDER_LIMIT) {
        debugf("VEC: mode2 watchdog placeholder reached\n");
    }
    return ctx.executed_insts;
}

static uint64_t execute_vec_coord(const vec_task_req_t& task,
                                  const vec_coord_t& coord,
                                  const vec_inst_t* inst_queue,
                                  uint8_t* uram,
                                  vec_sparse_memory_t* sparse_memory,
                                  interrupt_controller_t* interrupts,
                                  size_t lane_count)
{
    (void)lane_count;
    vec_thread_context_t ctx;
    ctx.coord = coord;
    ctx.tcsr = &task.tcsr_snapshot;
    ctx.uram = uram;
    ctx.sparse_memory = sparse_memory;
    ctx.uram_range = task.uram_range;
    ctx.interrupts = interrupts;

    return task.control.dynamic ? execute_mode2(ctx, task, inst_queue)
                                : execute_mode1(ctx, task, inst_queue);
}

static vec_task_resp_t execute_vec_task(const vec_task_req_t& task,
                                        const vec_inst_t* inst_queue,
                                        uint8_t* uram,
                                        vec_sparse_memory_t* sparse_memory,
                                        interrupt_controller_t* interrupts,
                                        size_t lane_count)
{
    uint64_t executed = 0;
    for(uint64_t outer = task.outer_begin; outer < task.outer_end;) {
        executed += execute_vec_coord(task, vec_outer_coord(task.tcsr_snapshot, outer),
                                      inst_queue, uram, sparse_memory,
                                      interrupts, lane_count);
        if(task.outer_stride >= task.outer_end - outer) {
            break;
        }
        outer += task.outer_stride;
    }
    return vec_task_resp_t{task.block_id, executed};
}

}

uint32_t vec_sparse_memory_t::read_u32(uint64_t addr)
{
    std::lock_guard<std::mutex> lock(mutex);
    auto it = words.find(addr >> 2);
    return it == words.end() ? 0 : it->second;
}

void vec_sparse_memory_t::write_u32(uint64_t addr, uint32_t value)
{
    std::lock_guard<std::mutex> lock(mutex);
    words[addr >> 2] = value;
}

uint32_t vec_sparse_memory_t::atomic_update_u32(uint64_t addr, uint32_t value, uint8_t op)
{
    std::lock_guard<std::mutex> lock(mutex);
    uint32_t& word = words[addr >> 2];
    uint32_t old = word;
    switch(op & 0x3) {
    case 0:
        word = old + value;
        break;
    case 1:
        word = old & value;
        break;
    case 2:
        word = old | value;
        break;
    case 3:
        word = old ^ value;
        break;
    }
    return old;
}

vec_executer_t::vec_executer_t(MPMCQueue<vec_vfork_req_t, 16>* vfork_queue,
                               MPMCQueue<vec_resp_t, 16>* resp_queue,
                               const vec_inst_t* inst_queue,
                               uint8_t* uram,
                               vec_sparse_memory_t* sparse_memory,
                               size_t lane_count,
                               size_t thread_count,
                               size_t tracked_blocks,
                               interrupt_controller_t* interrupts)
    : vfork_queue(vfork_queue),
      resp_queue(resp_queue),
      inst_queue(inst_queue),
      uram(uram),
      sparse_memory(sparse_memory),
      interrupts(interrupts),
      lane_count(lane_count),
      thread_count(thread_count),
      tracked_blocks(tracked_blocks),
      running(false)
{
}

vec_executer_t::~vec_executer_t()
{
    stop();
}

void vec_executer_t::start()
{
    running.store(true, std::memory_order_release);
    scheduler_thread = std::thread(&vec_executer_t::run, this);
    for(size_t i = 0; i < thread_count; i++) {
        worker_threads.emplace_back(&vec_executer_t::worker_loop, this);
    }
}

void vec_executer_t::stop()
{
    if(!running.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    if(scheduler_thread.joinable()) {
        scheduler_thread.join();
    }
    for(auto& thread : worker_threads) {
        if(thread.joinable()) {
            thread.join();
        }
    }
}

void vec_executer_t::dispatch_block(active_block_t& block)
{
    const uint64_t outer_count = vec_outer_coord_count(block.req.tcsr_snapshot);
    const uint64_t job_count = std::min<uint64_t>(outer_count, thread_count);
    for(uint64_t job = 0; job < job_count; job++) {
        vec_task_req_t task;
        task.block_id = block.req.block_id;
        task.start_slot = block.req.start_slot;
        task.inst_count = block.req.inst_count;
        task.outer_begin = job;
        task.outer_end = outer_count;
        task.outer_stride = job_count;
        task.control = block.req.control;
        task.tcsr_snapshot = block.req.tcsr_snapshot;
        task.uram_range = block.req.uram_range;
        task_req_queue.enqueue(std::move(task));
    }
    block.total_tasks = job_count;
    block.dispatched = true;
    if(job_count == 0) {
        block.done = true;
    }
}

void vec_executer_t::drain_task_responses()
{
    while(auto resp = task_resp_queue.try_dequeue()) {
        for(auto& block : active_blocks) {
            if(block.req.block_id == resp->block_id) {
                block.finished_tasks++;
                if(block.finished_tasks >= block.total_tasks) {
                    block.done = true;
                }
                break;
            }
        }
    }
}

bool vec_executer_t::retire_ready_blocks()
{
    bool retired = false;
    while(!active_blocks.empty() && active_blocks.front().done) {
        const auto& block = active_blocks.front();
        resp_queue->enqueue(vec_resp_t{
            block.req.block_id,
            block.req.start_slot,
            block.req.inst_count
        });
        active_blocks.pop_front();
        retired = true;
    }
    return retired;
}

void vec_executer_t::run()
{
    while(running.load(std::memory_order_acquire)) {
        drain_task_responses();
        retire_ready_blocks();

        if(active_blocks.size() < tracked_blocks) {
            if(auto req = vfork_queue->try_dequeue_for(std::chrono::milliseconds(1))) {
                active_blocks.push_back(active_block_t{*req});
            }
        }

        for(auto& block : active_blocks) {
            if(!block.dispatched) {
                dispatch_block(block);
                break;
            }
        }
    }
}

void vec_executer_t::worker_loop()
{
    while(running.load(std::memory_order_acquire) || !task_req_queue.empty()) {
        auto task = task_req_queue.try_dequeue_for(std::chrono::milliseconds(1));
        if(!task) {
            continue;
        }
        task_resp_queue.enqueue(execute_vec_task(
            *task, inst_queue, uram, sparse_memory, interrupts, lane_count));
    }
}

vec_t::vec_t(exec_unit_t* parrent, uint8_t* uram, ulong lane_width,
             tensor_context_t* context,
             interrupt_controller_t* interrupts)
    : executer(&vfork_queue, &resp_queue, inst_queue, uram, &sparse_memory,
               lane_width, 4, 4, interrupts),
      inst_queue_head(0),
      inst_queue_fork_head(0),
      inst_queue_tail(0),
      uram(uram),
      lane_width(lane_width),
      context(context),
      interrupts(interrupts),
      active_context(tensor_context_snapshot_t::defaults())
{
    parrent->register_issue_callback(this, &vec_t::issue);
    this->register_commit_callback(parrent, &exec_unit_t::commit);
    this->register_scoreboard_callback(parrent, &exec_unit_t::scoreboard);
    std::fill(std::begin(tcsr), std::end(tcsr), vec_tcsr_t{});
    executer.start();
}

void vec_t::enqueue_context(const rvt_pack_t& inst,
                            const tensor_context_snapshot_t& context_value)
{
    issue_queue.enqueue(queued_tensor_inst_t{inst, context_value, true});
}

vec_t::~vec_t()
{
    executer.stop();
}

ulong vec_t::issue(const rvt_pack_t* inst)
{
    if(!is_vec_tcsr(*inst) && !is_vec_inst(inst->inst)) {
        return 0;
    }
    const auto snapshot = context ? context->snapshot()
                                  : tensor_context_snapshot_t::defaults();
    issue_queue.enqueue(queued_tensor_inst_t{*inst, snapshot, false});
    return 1;
}

uint64_t vec_t::inst_queue_free_count() const
{
    return INST_QUEUE_CAPACITY - (inst_queue_head - inst_queue_tail);
}

bool vec_t::inst_queue_has_space() const
{
    return (inst_queue_head - inst_queue_tail) < INST_QUEUE_CAPACITY;
}

vec_vfork_control_t vec_t::decode_control(uint64_t payload) const
{
    payload &= VEC_CONTROL_VALID_MASK;
    vec_vfork_control_t control;
    control.prologue0_end = payload & 0xff;
    control.prologue1_end = (payload >> 8) & 0xff;
    control.loop_end = (payload >> 16) & 0xff;
    control.epilogue1_end = (payload >> 24) & 0xff;
    control.epilogue0_end = (payload >> 32) & 0xff;
    control.dynamic = (payload & VEC_CONTROL_DYNAMIC) != 0;
    return control;
}

void vec_t::poll_responses(bool wait_one)
{
    bool got_one = false;
    do {
        std::optional<vec_resp_t> resp;
        if(wait_one && !got_one) {
            resp = resp_queue.dequeue();
        } else {
            resp = resp_queue.try_dequeue();
        }
        if(!resp) {
            break;
        }
        got_one = true;
        inst_queue_tail = std::max(inst_queue_tail, resp->start_slot + resp->inst_count);
        if(inflight_blocks.empty() || inflight_blocks.front().block_id != resp->block_id) {
            throw std::logic_error("unexpected vector block retirement response");
        }
        inflight_blocks.pop_front();
        block_retirement.retire(resp->block_id);
        debugf("VEC: block %lu retired, free=%lu\n", resp->block_id, inst_queue_free_count());
    } while(true);
    drain_scoreboard_fences();
}

void vec_t::wait_for_retirement(uint64_t target)
{
    block_retirement.wait_until(target, [this]() {
        poll_responses(true);
    });
    drain_scoreboard_fences();
}

void vec_t::drain_scoreboard_fences()
{
    scoreboard_fences.drain_ready(
        [this](uint64_t target) {
            return block_retirement.reached(target);
        },
        [this](const vec_scoreboard_fence_t& fence) {
            if(fence.kind == vec_scoreboard_fence_kind_t::produce) {
                debugf("VEC: scoreboard produce %lx after %lu blocks\n",
                       fence.data, fence.target);
                scoreboard(fence.data);
                return;
            }

            rvt_commit_t cmt_pack;
            cmt_pack.timestamp = fence.timestamp;
            cmt_pack.rd = fence.rd;
            cmt_pack.data = 0;
            debugf("VEC: scoreboard bar after %lu blocks\n", fence.target);
            commit(&cmt_pack);
        });
}

void vec_t::handle_tcsr_read(const rvt_pack_t& inst, uint32_t tcsr_id,
                             const tensor_context_snapshot_t& context_value)
{
    rvt_commit_t cmt_pack;
    cmt_pack.timestamp = inst.timestamp;
    cmt_pack.rd = (inst.inst >> 7) & 0x1f;

    if(tcsr_id == 0) {
        cmt_pack.data = inst_queue_free_count();
    } else if(tcsr_id == 1) {
        cmt_pack.data = inst_queue_head - inst_queue_fork_head;
    } else if(tcsr_id < VEC_TCSR_COUNT) {
        const size_t bank = context_value.vec_tmap.base % std::size(tcsr);
        uint64_t* raw = reinterpret_cast<uint64_t*>(&tcsr[bank]);
        cmt_pack.data = raw[tcsr_id];
    } else {
        debugf("VEC: illegal tcsr read id %u\n", tcsr_id);
        cmt_pack.data = 0;
    }
    commit(&cmt_pack);
}

void vec_t::handle_tcsr_write(const rvt_pack_t& inst, uint32_t tcsr_id,
                              const tensor_context_snapshot_t& context_value)
{
    if(tcsr_id >= VEC_TCSR_COUNT) {
        debugf("VEC: illegal tcsr write id %u\n", tcsr_id);
        return;
    }

    uint64_t value = inst.rd & mask_for_width(tcsr_width[tcsr_id]);
    if(tcsr_id == 0) {
        handle_vfork(value, context_value);
        return;
    }
    if(tcsr_id == 1) {
        handle_vec_sync();
        return;
    }

    const size_t bank = context_value.vec_tmap.base % std::size(tcsr);
    uint64_t* raw = reinterpret_cast<uint64_t*>(&tcsr[bank]);
    raw[tcsr_id] = value;
    if(auto it = iterator_by_offset(tcsr[bank], tcsr_id)) {
        (void)it;
    }
}

void vec_t::handle_vfork(uint64_t payload,
                         const tensor_context_snapshot_t& context_value)
{
    vec_vfork_control_t control = decode_control(payload);
    uint64_t pending_unforked = inst_queue_head - inst_queue_fork_head;
    uint64_t inst_count = control.dynamic ? pending_unforked : control.epilogue0_end;

    if(inst_count == 0) {
        debugf("VEC: ignore empty vfork\n");
        return;
    }
    if(inst_count > pending_unforked) {
        debugf("VEC: vfork length %lu exceeds queued instructions %lu\n",
               inst_count, pending_unforked);
        return;
    }

    vec_vfork_req_t req;
    req.block_id = block_retirement.issue();
    req.start_slot = inst_queue_fork_head;
    req.inst_count = inst_count;
    req.control = control;
    const size_t bank = context_value.vec_tmap.base % std::size(tcsr);
    req.tcsr_snapshot = tcsr[bank];
    req.tcsr_snapshot.control = payload;
    req.uram_range = context_value.uram_range;
    inst_queue_fork_head += inst_count;
    inflight_blocks.push_back(req);
    vfork_queue.enqueue(req);
    debugf("VEC: vfork block=%lu start=%lu count=%lu dynamic=%d\n",
           req.block_id, req.start_slot, req.inst_count, req.control.dynamic);
}

void vec_t::handle_vec_sync()
{
    wait_for_retirement(block_retirement.snapshot());
}

void vec_t::execute(void)
{
    while(1) {
        poll_responses(false);
        std::optional<queued_tensor_inst_t> pending_inst;
        if(scoreboard_fences.empty()) {
            pending_inst = issue_queue.dequeue();
        } else {
            pending_inst = issue_queue.try_dequeue_for(std::chrono::milliseconds(1));
            if(!pending_inst) {
                continue;
            }
        }
        const queued_tensor_inst_t queued = std::move(*pending_inst);
        const rvt_pack_t& inst = queued.inst;
        active_context = queued.context;
        MFENCE();
        poll_responses(false);
        if(queued.context_write) {
            continue;
        }

        uint32_t unit = (inst.inst >> 23) & 0x1f;
        uint32_t tcsr_id = (inst.inst >> 13) & 0x3ff;
        bool is_tcsr_write = rvt::is_base(static_cast<uint32_t>(inst.inst))
            && unit == RVT_UNIT_VEC && ((inst.inst >> 12) & 0x1) == 0;
        bool is_tcsr_read = rvt::is_base(static_cast<uint32_t>(inst.inst))
            && unit == RVT_UNIT_VEC && ((inst.inst >> 12) & 0x1) == 1;
        bool is_vec_sb_produce = rvt::is_base(static_cast<uint32_t>(inst.inst))
            && unit == RVT_UNIT_BASE && tcsr_id == RVT_UNIT_VEC && ((inst.inst >> 12) & 0x1) == 0;
        bool is_vec_sb_bar = rvt::is_base(static_cast<uint32_t>(inst.inst))
            && unit == RVT_UNIT_BASE && tcsr_id == RVT_UNIT_VEC && ((inst.inst >> 12) & 0x1) == 1;

        if(is_tcsr_write) {
            handle_tcsr_write(inst, tcsr_id, active_context);
            continue;
        }
        if(is_tcsr_read) {
            handle_tcsr_read(inst, tcsr_id, active_context);
            continue;
        }
        if(is_vec_sb_produce) {
            scoreboard_fences.enqueue(vec_scoreboard_fence_t{
                block_retirement.snapshot(),
                vec_scoreboard_fence_kind_t::produce,
                inst.rd,
                inst.timestamp,
                static_cast<uint8_t>((inst.inst >> 7) & 0x1f)
            });
            drain_scoreboard_fences();
            continue;
        }
        if(is_vec_sb_bar) {
            scoreboard_fences.enqueue(vec_scoreboard_fence_t{
                block_retirement.snapshot(),
                vec_scoreboard_fence_kind_t::bar,
                0,
                inst.timestamp,
                static_cast<uint8_t>((inst.inst >> 7) & 0x1f)
            });
            drain_scoreboard_fences();
            continue;
        }
        if(rvt::is_base(static_cast<uint32_t>(inst.inst))) {
            if(interrupts) {
                interrupts->raise(interrupt_source_t::flexvec_decode_error,
                                  static_cast<uint32_t>(inst.inst));
            }
            continue;
        }

        while(!inst_queue_has_space()) {
            poll_responses(true);
        }

        uint64_t slot = inst_queue_head % INST_QUEUE_CAPACITY;
        inst_queue[slot] = vec_inst_t{
            inst.inst,
            inst.rd,
            inst.rs1,
            inst.rs2,
            inst.timestamp
        };
        inst_queue_head++;
        const size_t bank = active_context.vec_tmap.base % std::size(tcsr);
        tcsr[bank].control = inst_queue_free_count();
    }
}

}
