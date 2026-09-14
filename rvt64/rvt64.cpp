#include "utils.hpp"
#include "rvt64.hpp"
#include "resource_scheduler.hpp"
#include "rvt64_helper.hpp"
#include "swizzle.hpp"
#include "fputils.hpp"
#include "vec_convert.hpp"
#include <atomic>
#include <thread>
#include <cmath>
#include <iterator>
#include <limits>
#include "rvt_decoder.hpp"

namespace RVTNG {
    namespace {
        unsigned macro_type_bytes(uint8_t type) {
            switch(type) {
            case RVT_TYPE_FP32:
            case RVT_TYPE_S32:
            case RVT_TYPE_U32:
                return 4;
            case RVT_TYPE_BF16:
            case RVT_TYPE_FP16:
            case RVT_TYPE_S16:
            case RVT_TYPE_U16:
                return 2;
            case RVT_TYPE_FP8_E4M3:
            case RVT_TYPE_FP8_E5M2:
            case RVT_TYPE_MX_FP8_E4M3:
            case RVT_TYPE_MX_FP8_E5M2:
            case RVT_TYPE_S8:
            case RVT_TYPE_U8:
                return 1;
            default:
                return 0;
            }
        }
    }

    ulong rvt64_t::issue(const rvt_pack_t* inst) {
        uint32_t unit = (inst->inst >> 23) & 0x1f;
        bool is_rvt64_tcsr = rvt::is_base(inst->inst) &&
            (unit == RVT_UNIT_MACRO || is_unit_scoreboard(inst->inst, RVT_UNIT_MACRO));
        bool is_rvt64 = ((inst->inst & 0x7f) == 0x3f) | is_rvt64_tcsr;
        if(!is_rvt64) {
            return 0;
        }
        const auto snapshot = context ? context->snapshot()
                                      : tensor_context_snapshot_t::defaults();
        this->inst_queue.enqueue(queued_tensor_inst_t{*inst, snapshot, false});
        // debugf("RVT64: Enqueue RVT64 Instruction %lx!\n",inst->inst);
        return 1;
    }

    rvt64_t::rvt64_t(exec_unit_t* parrent, uint8_t* uram,
                     tensor_context_t* context,
                     interrupt_controller_t* interrupts)
        : uram(uram), context(context), interrupts(interrupts),
          active_context(tensor_context_snapshot_t::defaults()) {
        // 实现 RVT64 构造函数
        parrent->register_issue_callback(this, &rvt64_t::issue);
        this->register_commit_callback(parrent, &exec_unit_t::commit);
        this->register_scoreboard_callback(parrent, &exec_unit_t::scoreboard);
        this->register_resource_produce_callback(
            parrent, &exec_unit_t::resource_produce);
        memset(this->tcsr, 0, sizeof(this->tcsr));
    }

    void rvt64_t::enqueue_context(
        const rvt_pack_t& inst,
        const tensor_context_snapshot_t& context_value) {
        inst_queue.enqueue(queued_tensor_inst_t{inst, context_value, true});
    }

    bool rvt64_t::validate_tensor(
        uint64_t base, const rvt64_tcsr_t& control,
        unsigned operand, unsigned element_bytes, bool collapse_dim0, bool write,
        interrupt_source_t range_source,
        interrupt_source_t align_source) const {
        if((context == nullptr && interrupts == nullptr) || element_bytes == 0) {
            return true;
        }
        base &= 0x1fffff;
        uint64_t stride1 = 0;
        uint64_t stride2 = 0;
        uint64_t stride3 = 0;
        if(operand == 0) {
            stride1 = control.Src0Dim1Stride;
            stride2 = control.Src0Dim2Stride;
            stride3 = control.Src0Dim3Stride;
        } else if(operand == 1) {
            stride1 = control.Src1Dim1Stride;
            stride2 = control.Src1Dim2Stride;
            stride3 = control.Src1Dim3Stride;
        } else {
            stride1 = control.DstDim1Stride;
            stride2 = control.DstDim2Stride;
            stride3 = control.DstDim3Stride;
        }
        // MACRO stores may start within a URAM beat, but must preserve element
        // alignment. Every nonempty outer dimension must also preserve it.
        const uint64_t alignment = write ? element_bytes : 32;
        const uint64_t bases[] = {base, base + stride1, base + stride2, base + stride3};
        const bool active[] = {true, control.dim1 != 0, control.dim2 != 0, control.dim3 != 0};
        for(unsigned i = 0; i < 4; ++i) {
            if(active[i] && (bases[i] % alignment) != 0) {
                if(interrupts) {
                    interrupts->raise(align_source, static_cast<uint32_t>(bases[i]));
                }
                return false;
            }
        }
        const __uint128_t span =
            __uint128_t{collapse_dim0 ? 0 : control.dim0} * element_bytes +
            __uint128_t{control.dim1} * stride1 +
            __uint128_t{control.dim2} * stride2 +
            __uint128_t{control.dim3} * stride3 + element_bytes;
        const bool overflow = span > std::numeric_limits<uint64_t>::max();
        const uint64_t size = overflow ? std::numeric_limits<uint64_t>::max()
                                       : static_cast<uint64_t>(span);
        if(overflow || size > std::numeric_limits<size_t>::max() ||
           !active_context.uram_range.contains(base,
                                                static_cast<size_t>(size))) {
            if(interrupts) {
                const __uint128_t last = __uint128_t{base} + size - 1;
                const uint64_t failing = last > std::numeric_limits<uint64_t>::max()
                    ? std::numeric_limits<uint64_t>::max()
                    : static_cast<uint64_t>(last);
                interrupts->raise(range_source, static_cast<uint32_t>(failing));
            }
            return false;
        }
        return true;
    }

    bool rvt64_t::validate_instruction(const rvt_pack_t& inst,
                                       const rvt64_tcsr_t& control) const {
        const uint8_t extension = (inst.inst >> 25) & 0x7f;
        const uint8_t data_type = (inst.inst >> 32) & 0xff;
        const uint8_t dst_type = (inst.inst >> 40) & 0xff;
        const uint8_t broadcast = (inst.inst >> 46) & 0x3;
        const auto check_a = [&](unsigned bytes, bool collapse = false) {
            return validate_tensor(
                inst.rs1, control, 0, bytes, collapse, false,
                interrupt_source_t::neovec_uram_lda_range,
                interrupt_source_t::neovec_uram_lda_align);
        };
        const auto check_b = [&](uint64_t base, unsigned bytes,
                                 bool collapse = false) {
            return validate_tensor(
                base, control, 1, bytes, collapse, false,
                interrupt_source_t::neovec_uram_ldb_range,
                interrupt_source_t::neovec_uram_ldb_align);
        };
        const auto check_store = [&](uint64_t base, unsigned bytes,
                                     bool collapse = false) {
            return validate_tensor(
                base, control, 2, bytes, collapse, true,
                interrupt_source_t::neovec_uram_store_range,
                interrupt_source_t::neovec_uram_store_align);
        };

        switch(extension) {
        case RVT_MACRO_EXT_ELEMWISE: {
            const unsigned bytes = macro_type_bytes(data_type);
            return (broadcast == 2 || check_a(bytes, broadcast == 1)) &&
                check_b(inst.rs2, bytes) && check_store(inst.rd, bytes);
        }
        case RVT_MACRO_EXT_CONVERT:
            return check_a(macro_type_bytes(data_type)) &&
                check_store(inst.rd, macro_type_bytes(dst_type));
        case RVT_MACRO_EXT_SFU:
            return check_a(sizeof(float)) && check_store(inst.rd, sizeof(float));
        case RVT_MACRO_EXT_REDUCE: {
            const unsigned bytes = macro_type_bytes(data_type);
            if(!check_a(bytes) || !check_store(inst.rd, bytes, true)) {
                return false;
            }
            return broadcast == 0 || check_store(inst.rs2, sizeof(uint32_t),
                                                  true);
        }
        default:
            return true;
        }
    }

    void rvt64_t::execute(void) {
        while(1) {
            // Fetch
            auto queued = inst_queue.dequeue();
            const auto& inst = queued.inst;
            active_context = queued.context;
            // Fence
            MFENCE();
            if(queued.context_write) {
                continue;
            }
            // Decode
            uint8_t group = (inst.inst >> 12) & 0x7;
            uint8_t extension = (inst.inst >> 25) & 0x7f;
            uint8_t data_type = (inst.inst >> 32) & 0xff;
            uint8_t operation = extension == RVT_MACRO_EXT_SFU
                ? RVT64_SFU_OP(inst.inst) : RVT64_ELEM_OP(inst.inst);
            uint8_t Bd = (inst.inst >> 46) & 0x3;

            uint8_t src_type = (inst.inst >> 32) & 0xff;
            uint8_t dst_type = (inst.inst >> 40) & 0xff; 

            uint32_t unit = (inst.inst >> 23) & 0x1f;
            uint32_t tcsr_id = (inst.inst >> 13) & 0x3ff;

            bool is_macro_tcsr_write = rvt_is_base(inst.inst) && unit == RVT_UNIT_MACRO && ((inst.inst >> 12) & 0x1) == 0;
            bool is_macro_tcsr_read  = rvt_is_base(inst.inst) && unit == RVT_UNIT_MACRO && ((inst.inst >> 12) & 0x1) == 1;
            bool is_macro_sb_produce = rvt_is_base(inst.inst) &&
                unit == RVT_UNIT_BASE && tcsr_id == RVT_UNIT_MACRO &&
                ((inst.inst >> 12) & 0x1) == 0;
            bool is_macro_sb_bar = rvt_is_base(inst.inst) &&
                unit == RVT_UNIT_BASE && tcsr_id == RVT_UNIT_MACRO &&
                ((inst.inst >> 12) & 0x1) == 1;
            const bool is_resource_produce = rvt_is_base(inst.inst) &&
                unit == RVT_UNIT_BASE &&
                tcsr_id == RESOURCE_PRODUCE_BASE + RVT_UNIT_MACRO &&
                ((inst.inst >> 12) & 0x1) == 0;
            if(is_macro_tcsr_write | is_macro_tcsr_read) {
                debugf("MACRO: tcsr read write %d:\n", tcsr_id);
                const uint8_t logical_group = tcsr_id / 32;
                const auto physical_group = active_context.physical_group(
                    tensor_unit_t::macro_unit, logical_group, std::size(tcsr));
                if(!physical_group) {
                    if(interrupts) {
                        interrupts->raise(
                            interrupt_source_t::neovec_tcsr_access_error,
                            static_cast<uint32_t>(inst.inst));
                    }
                    if(is_macro_tcsr_read) {
                        rvt_commit_t cmt_pack{};
                        cmt_pack.timestamp = inst.timestamp;
                        cmt_pack.rd = (inst.inst >> 7) & 0x1f;
                        commit(&cmt_pack);
                    }
                    continue;
                }
                uint64_t* _tcsr = reinterpret_cast<uint64_t*>(
                    &this->tcsr[*physical_group]);
                const uint32_t sub_address = tcsr_id & 0x1f;
                if(is_macro_tcsr_write) {
                    if(sub_address != 31)
                        _tcsr[sub_address] = inst.rd & ((1ull << this->tcsr_width[sub_address]) - 1);
                    this->tcsr[*physical_group].checksum += 1;
                    debugf("* write %lx -> %lx.\n", inst.rd, _tcsr[sub_address]);
                } else {
                    rvt_commit_t cmt_pack;
                    cmt_pack.timestamp = inst.timestamp;
                    cmt_pack.rd = (inst.inst >> 7) & 0x1f;
                    cmt_pack.data = _tcsr[sub_address];
                    commit(&cmt_pack);
                    debugf("* read %lx -> %lx.\n", _tcsr[sub_address], cmt_pack.rd);
                }
                continue;
            }
            if(is_macro_sb_produce) {
                debugf("MACRO: scoreboard produce %lx:\n", inst.rd);
                scoreboard(inst.rd);
                continue;
            }
            if(is_macro_sb_bar) {
                rvt_commit_t cmt_pack;
                cmt_pack.timestamp = inst.timestamp;
                cmt_pack.rd = (inst.inst >> 7) & 0x1f;
                cmt_pack.data = inst.rd;
                debugf("MACRO: scoreboard bar %lx:\n", cmt_pack.rd);
                commit(&cmt_pack);
                continue;
            }
            if(is_resource_produce) {
                resource_produce(RVT_UNIT_MACRO);
                continue;
            }
            if(rvt_is_base(inst.inst)) {
                if(interrupts) {
                    interrupts->raise(interrupt_source_t::neovec_decode_error,
                                      static_cast<uint32_t>(inst.inst));
                }
                continue;
            }
            {
                rvt64_tcsr_t empty_tcsr{};
                const auto physical_group = active_context.physical_group(
                    tensor_unit_t::macro_unit, group, std::size(tcsr));
                if(!physical_group && interrupts) {
                    interrupts->raise(
                        interrupt_source_t::neovec_tcsr_access_error,
                        static_cast<uint32_t>(inst.inst));
                }
                const rvt64_tcsr_t& selected_tcsr = physical_group
                    ? tcsr[*physical_group] : empty_tcsr;
                if(!validate_instruction(inst, selected_tcsr)) {
                    continue;
                }
                switch (extension) {
                    case RVT_MACRO_EXT_ELEMWISE:
                        switch (data_type) {
                            case RVT_TYPE_S32:
                                switch (operation) {
                                    case TX_ELEM_ADD:
                                        rvt64_handle_binary_operation<int32_t, int32_t>(
                                            [](auto a, auto b) -> int32_t { return a + b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, selected_tcsr
                                        );
                                        break;
                                    case TX_ELEM_SUB:
                                        rvt64_handle_binary_operation<int32_t, int32_t>(
                                            [](auto a, auto b) -> int32_t { return a - b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, selected_tcsr
                                        );
                                        break;
                                    case TX_ELEM_MUL:
                                        rvt64_handle_binary_operation<int32_t, int32_t>(
                                            [](auto a, auto b) -> int32_t { return a * b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, selected_tcsr
                                        );
                                        break;   
                                    default:
                                        debugf("RVT64: unsupported integer elemwise op %u\n",
                                               operation);
                                        break;
                                }
                                break;
                            case RVT_TYPE_FP32:
                                switch (operation) {
                                    case TX_ELEM_ADD:
                                        rvt64_handle_binary_operation<float, float>(
                                            [](auto a, auto b) -> float { return a + b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, selected_tcsr
                                        );
                                        break;
                                    case TX_ELEM_SUB:
                                        rvt64_handle_binary_operation<float, float>(
                                            [](auto a, auto b) -> float { return a - b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, selected_tcsr
                                        );
                                        break;
                                    case TX_ELEM_MUL:
                                        rvt64_handle_binary_operation<float, float>(
                                            [](auto a, auto b) -> float { return a * b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, selected_tcsr
                                        );
                                        break;                          
                                    case TX_ELEM_MAX:
                                        rvt64_handle_binary_operation<float, float>(
                                            [](auto a, auto b) -> float { return a > b ? a : b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, selected_tcsr
                                        );
                                        break;
                                    case TX_ELEM_MIN:
                                        rvt64_handle_binary_operation<float, float>(
                                            [](auto a, auto b) -> float { return a < b ? a : b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, selected_tcsr
                                        );
                                        break;
                                    default:
                                        debugf("RVT64: unsupported FP elemwise op %u\n",
                                               operation);
                                        break;
                                }
                                break;
                            default:
                                debugf("RVT64: Data Type not implemented!\n");
                                break;
                        }
                        break;
                    case RVT_MACRO_EXT_CONVERT:
                        switch (src_type) {
                            case RVT_TYPE_FP32: // FP32
                                switch (dst_type) {
                                    case RVT_TYPE_S32: // S32
                                        rvt64_handle_unary_operation<uint32_t, uint32_t>(
                                            [](uint32_t x) { return vec_fp32_to_s32(x, vec_rounding_mode_t::rne); },
                                            uram, inst.rs1, inst.rd, selected_tcsr
                                        );
                                        break;
                                    case RVT_TYPE_BF16: // BF16
                                        rvt64_handle_unary_operation<uint32_t, uint16_t>(
                                            [](uint32_t x) { return vec_fp32_to_bf16(x, vec_rounding_mode_t::rne); },
                                            uram, inst.rs1, inst.rd, selected_tcsr
                                        );
                                        break;
                                    case RVT_TYPE_FP16: // FP16
                                        rvt64_handle_unary_operation<uint32_t, uint16_t>(
                                            [](uint32_t x) { return vec_fp32_to_fp16(x, vec_rounding_mode_t::rne); },
                                            uram, inst.rs1, inst.rd, selected_tcsr
                                        );
                                        break;
                                    case RVT_TYPE_MX_FP8_E4M3: // FP8E4M3
                                        rvt64_handle_unary_operation<float, uint8_t>(
                                            FP32ToFP8<4>,
                                            uram, inst.rs1, inst.rd, selected_tcsr
                                        );
                                        break;
                                    default:
                                        debugf("RVT64: Unsupport tx.convert src_type==%d, dst_type==%d!\n", src_type, dst_type);
                                        break;
                                }
                                break;
                            default:
                                debugf("RVT64: Unsupport tx.convert src_type==%d!\n", src_type);
                                break;
                        }
                        break;
                    case RVT_MACRO_EXT_SFU:
                        switch (operation) {
                            case TX_SFU_LOG2: // tx.sfu.log2
                                rvt64_handle_unary_operation<float, float>(
                                        [](auto x) -> float { return log2f(x); },
                                        uram, inst.rs1, inst.rd, selected_tcsr
                                    );
                                break;
                            case TX_SFU_EXP2: // tx.sfu.exp2
                                rvt64_handle_unary_operation<float, float>(
                                        [](auto x) -> float { return exp2f(x); },
                                        uram, inst.rs1, inst.rd, selected_tcsr
                                    );
                                break;
                            case TX_SFU_SIN: // tx.sfu.sin
                                rvt64_handle_unary_operation<float, float>(
                                        [](auto x) -> float { return sinf(x); },
                                        uram, inst.rs1, inst.rd, selected_tcsr
                                    );
                                break;
                            case TX_SFU_COS: // tx.sfu.cos
                                rvt64_handle_unary_operation<float, float>(
                                        [](auto x) -> float { return cosf(x); },
                                        uram, inst.rs1, inst.rd, selected_tcsr
                                    );
                                break;
                            case TX_SFU_RSQRT: // tx.sfu.rsqrt
                                rvt64_handle_unary_operation<float, float>(
                                        [](auto x) -> float { return 1.0f / sqrtf(x); },
                                        uram, inst.rs1, inst.rd, selected_tcsr
                                    );
                                break;
                            case TX_SFU_RECIP: // tx.sfu.recip
                                rvt64_handle_unary_operation<float, float>(
                                        [](auto x) -> float { return  1.0f / x; },
                                        uram, inst.rs1, inst.rd, selected_tcsr
                                    );
                                break;
                            default:
                                debugf("RVT64: Unsupport tx.sfu op==%d!\n", operation);
                                break;
                        }
                        break;
                    case RVT_MACRO_EXT_REDUCE:
                        switch(data_type){
                            case RVT_TYPE_S32: // S32
                                switch (operation) {
                                    case TX_ELEM_MAX: // tx.int.reduce.max
                                        rvt64_handle_reduce_index_operation<int32_t, int32_t>(
                                            [](auto a, auto b) -> bool { return a > b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, selected_tcsr
                                        );
                                        break;
                                    case TX_ELEM_MIN: // tx.int.reduce.min
                                        rvt64_handle_reduce_index_operation<int32_t, int32_t>(
                                            [](auto a, auto b) -> bool { return a < b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, selected_tcsr
                                        );
                                        break;
                                    case TX_ELEM_ADD: // tx.int.reduce.sum
                                        rvt64_handle_reduce_operation<int32_t, int32_t>(
                                            [](auto a, auto b) -> int32_t { return a + b; },
                                            uram, inst.rs1, inst.rd, selected_tcsr
                                        );
                                        break;
                                    default:
                                        debugf("RVT64: Unsupport tx.int.reduce op==%d!\n", operation);
                                        break;
                                }
                                break;
                            case RVT_TYPE_FP32: // FP32
                                switch (operation) {
                                    case TX_ELEM_MAX: // tx.fp.reduce.max
                                        rvt64_handle_reduce_index_operation<float, float>(
                                            [](auto a, auto b) -> bool { return a > b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, selected_tcsr
                                        );
                                        break;
                                    case TX_ELEM_MIN: // tx.fp.reduce.min
                                        rvt64_handle_reduce_index_operation<float, float>(
                                            [](auto a, auto b) -> bool { return a < b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, selected_tcsr
                                        );
                                        break;
                                    case TX_ELEM_ADD: // tx.fp.reduce.sum
                                        rvt64_handle_reduce_operation<float, float>(
                                            [](auto a, auto b) -> float { return a + b; },
                                            uram, inst.rs1, inst.rd, selected_tcsr
                                        );
                                        break;
                                    default:
                                        debugf("RVT64: Unsupport tx.fp.reduce op==%d!\n", operation);
                                        break;
                                }
                                break;
                            default:
                                debugf("RVT64: Unsupport Reduce Type %x!\n", data_type);
                                break;  
                        }
                        break;
                    default:
                        debugf("RVT64: Unsupport Instruction %lx with extension %x!\n", inst.inst, extension);
                        break;
                }
                continue;
            }
        }
    }
}
