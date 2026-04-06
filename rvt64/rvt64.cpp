#include "utils.hpp"
#include "rvt64.hpp"
#include "rvt64_helper.hpp"
#include "swizzle.hpp"
#include "fputils.hpp"
#include <atomic>
#include <thread>
#include <cmath>
#include "rvt_decoder.hpp"

namespace RVTNG {
    ulong rvt64_t::issue(const rvt_pack_t* inst) {
        uint32_t unit = (inst->inst >> 23) & 0x1f;
        uint32_t tcsr_id = (inst->inst >> 13) & 0x3ff;
        bool is_rvt64_tcsr = rvt::is_base(inst->inst) && (unit == RVT_UNIT_MACRO || (unit== RVT_UNIT_BASE && tcsr_id == RVT_UNIT_MACRO));
        bool is_rvt64 = ((inst->inst & 0x7f) == 0x3f) | is_rvt64_tcsr;
        if(!is_rvt64) {
            return 0;
        }
        this->inst_queue.enqueue(*inst);
        // debugf("RVT64: Enqueue RVT64 Instruction %lx!\n",inst->inst);
        return 1;
    }

    rvt64_t::rvt64_t(exec_unit_t* parrent, uint8_t* uram): uram(uram) {
        // 实现 RVT64 构造函数
        parrent->register_issue_callback(this, &rvt64_t::issue);
        this->register_commit_callback(parrent, &exec_unit_t::commit);
        this->register_scoreboard_callback(parrent, &exec_unit_t::scoreboard);
        memset(this->tcsr, 0, sizeof(this->tcsr));
    }

    void rvt64_t::execute(void) {
        while(1) {
            // Fetch
            auto inst = inst_queue.dequeue();
            // Fence
            MFENCE();
            // Decode
            uint8_t group = (inst.inst >> 12) & 0x7;
            uint8_t extension = (inst.inst >> 25) & 0x7f;
            uint8_t data_type = (inst.inst >> 32) & 0xff;
            uint8_t operation = (inst.inst >> 40) & 0x3f;
            uint8_t Bd = (inst.inst >> 46) & 0x1;

            uint8_t src_type = (inst.inst >> 32) & 0xff;
            uint8_t dst_type = (inst.inst >> 40) & 0xff; 

            uint32_t unit = (inst.inst >> 23) & 0x1f;
            uint32_t tcsr_id = (inst.inst >> 13) & 0x3ff;

            bool is_macro_tcsr_write = rvt_is_base(inst.inst) && unit == RVT_UNIT_MACRO && ((inst.inst >> 12) & 0x1) == 0;
            bool is_macro_tcsr_read  = rvt_is_base(inst.inst) && unit == RVT_UNIT_MACRO && ((inst.inst >> 12) & 0x1) == 1;
            bool is_macro_sb_produce = rvt_is_base(inst.inst) && unit == RVT_UNIT_BASE && ((inst.inst >> 12) & 0x1) == 0;
            bool is_macro_sb_bar     = rvt_is_base(inst.inst) && unit == RVT_UNIT_BASE && ((inst.inst >> 12) & 0x1) == 1;
            if(is_macro_tcsr_write | is_macro_tcsr_read) {
                debugf("MACRO: tcsr read write %d:\n", tcsr_id);
                if(tcsr_id > 8 * 32) {
                    debugf("* Illgal id.\n");
                    continue;
                }
                uint64_t* _tcsr = (uint64_t*) &this->tcsr;
                if(is_macro_tcsr_write) {
                    if((tcsr_id % 32) != 31)
                        _tcsr[tcsr_id] = inst.rd & ((1 << this->tcsr_width[tcsr_id & 0x1f]) - 1);
                    this->tcsr[tcsr_id / 32].checksum += 1;
                    debugf("* write %lx -> %lx.\n", inst.rd, _tcsr[tcsr_id]);
                } else {
                    rvt_commit_t cmt_pack;
                    cmt_pack.timestamp = inst.timestamp;
                    cmt_pack.rd = (inst.inst >> 7) & 0x1f;
                    cmt_pack.data = _tcsr[tcsr_id];
                    commit(&cmt_pack);
                    debugf("* read %lx -> %lx.\n", _tcsr[tcsr_id], cmt_pack.rd);
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
            {
                switch (extension) {
                    case RVT_MACRO_EXT_ELEMWISE:
                        switch (data_type) {
                            case RVT_TYPE_S32:
                                switch (operation) {
                                    case TX_ELEM_ADD:
                                        rvt64_handle_binary_operation<int32_t, int32_t>(
                                            [](auto a, auto b) -> int32_t { return a + b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, tcsr[group], sizeof(uint32_t)
                                        );
                                        break;
                                    case TX_ELEM_SUB:
                                        rvt64_handle_binary_operation<int32_t, int32_t>(
                                            [](auto a, auto b) -> int32_t { return a - b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, tcsr[group], sizeof(uint32_t)
                                        );
                                        break;
                                    case TX_ELEM_MUL:
                                        rvt64_handle_binary_operation<int32_t, int32_t>(
                                            [](auto a, auto b) -> int32_t { return a * b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, tcsr[group], sizeof(uint32_t)
                                        );
                                        break;   
                                    default: 
                                        assert(0);                       
                                }
                                break;
                            case RVT_TYPE_FP32:
                                switch (operation) {
                                    case TX_ELEM_ADD:
                                        rvt64_handle_binary_operation<float, float>(
                                            [](auto a, auto b) -> float { return a + b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, tcsr[group], sizeof(float)
                                        );
                                        break;
                                    case TX_ELEM_SUB:
                                        rvt64_handle_binary_operation<float, float>(
                                            [](auto a, auto b) -> float { return a - b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, tcsr[group], sizeof(float)
                                        );
                                        break;
                                    case TX_ELEM_MUL:
                                        rvt64_handle_binary_operation<float, float>(
                                            [](auto a, auto b) -> float { return a * b; },
                                            uram, Bd, inst.rs1, inst.rs2, inst.rd, tcsr[group], sizeof(float)
                                        );
                                        break;                          
                                    default:
                                        assert(0);
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
                                        // TODO: Support Round Mode
                                        rvt64_handle_unary_operation<float, int32_t>(
                                            [](auto x) -> int32_t { return static_cast<int32_t>(x); },
                                            uram, inst.rs1, inst.rd, tcsr[group], sizeof(float), sizeof(int32_t)
                                        );
                                        break;
                                    case RVT_TYPE_BF16: // BF16
                                        rvt64_handle_unary_operation<float, uint16_t>(
                                            FP32ToBF16,
                                            uram, inst.rs1, inst.rd, tcsr[group], sizeof(float), 2
                                        );
                                        break;
                                    case RVT_TYPE_FP16: // FP16
                                        rvt64_handle_unary_operation<float, uint16_t>(
                                            FP32ToFP16,
                                            uram, inst.rs1, inst.rd, tcsr[group], sizeof(float), 2
                                        );
                                        break;
                                    case RVT_TYPE_MX_FP8_E4M3: // FP8E4M3
                                        rvt64_handle_unary_operation<float, uint8_t>(
                                            FP32ToFP8<4>,
                                            uram, inst.rs1, inst.rd, tcsr[group], sizeof(float), 1
                                        );
                                        break;
                                    default:
                                        debugf("RVT64: Unsupport tx.convert rs1type==%d, rs3type==%d!\n", rs1type, rs3type);
                                        break;
                                }
                                break;
                            default:
                                debugf("RVT64: Unsupport tx.convert rs1type==%d!\n", rs1type);
                                break;
                        }
                        break;
                    case RVT_MACRO_EXT_SFU:
                        switch (operation) {
                            case TX_SFU_LOG2: // tx.sfu.log2
                                rvt64_handle_unary_operation<float, float>(
                                        [](auto x) -> float { return log2f(x); },
                                        uram, inst.rs1, inst.rd, tcsr[group], sizeof(float), sizeof(float)
                                    );
                                break;
                            case TX_SFU_EXP2: // tx.sfu.exp2
                                rvt64_handle_unary_operation<float, float>(
                                        [](auto x) -> float { return exp2f(x); },
                                        uram, inst.rs1, inst.rd, tcsr[group], sizeof(float), sizeof(float)
                                    );
                                break;
                            case TX_SFU_SIN: // tx.sfu.sin
                                rvt64_handle_unary_operation<float, float>(
                                        [](auto x) -> float { return sinf(x); },
                                        uram, inst.rs1, inst.rd, tcsr[group], sizeof(float), sizeof(float)
                                    );
                                break;
                            case TX_SFU_COS: // tx.sfu.cos
                                rvt64_handle_unary_operation<float, float>(
                                        [](auto x) -> float { return cosf(x); },
                                        uram, inst.rs1, inst.rd, tcsr[group], sizeof(float), sizeof(float)
                                    );
                                break;
                            case TX_SFU_RSQRT: // tx.sfu.rsqrt
                                rvt64_handle_unary_operation<float, float>(
                                        [](auto x) -> float { return 1.0f / sqrtf(x); },
                                        uram, inst.rs1, inst.rd, tcsr[group], sizeof(float), sizeof(float)
                                    );
                                break;
                            case TX_SFU_RECIP: // tx.sfu.recip
                                rvt64_handle_unary_operation<float, float>(
                                        [](auto x) -> float { return  1.0f / x; },
                                        uram, inst.rs1, inst.rd, tcsr[group], sizeof(float), sizeof(float)
                                    );
                                break;
                            default:
                                debugf("RVT64: Unsupport tx.sfu subop==%d!\n", subop);
                                break;
                        }
                        break;
                    case RVT_MACRO_EXT_REDUCE:
                        switch(data_type){
                            case 0b10001100: // tx.int.reduce
                                switch (operation) {
                                    case 0: // tx.int.reduce.max
                                        rvt64_handle_reduce_operation<int32_t, int32_t>(
                                            [](auto a, auto b) -> int32_t { return std::max(a, b); }, 0xffffffff,
                                            uram, inst.rs1, inst.rd, tcsr[group], 4
                                        );
                                        break;
                                    case 1: // tx.int.reduce.min
                                        rvt64_handle_reduce_operation<int32_t, int32_t>(
                                            [](auto a, auto b) -> int32_t { return std::min(a, b); }, 0x7fffffff,
                                            uram, inst.rs1, inst.rd, tcsr[group], 4
                                        );
                                        break;
                                    case 2: // tx.int.reduce.sum
                                        rvt64_handle_reduce_operation<int32_t, int32_t>(
                                            [](auto a, auto b) -> int32_t { return a + b; }, 0,
                                            uram, inst.rs1, inst.rd, tcsr[group], 4
                                        );
                                        break;
                                    default:
                                        debugf("RVT64: Unsupport tx.int.reduce subop==%d!\n", subop);
                                        break;
                                }
                                break;
                            case 0b10001101: // tx.fp.reduce
                                switch (operation) {
                                    case 0: // tx.fp.reduce.max
                                        rvt64_handle_reduce_operation<float, float>(
                                            [](auto a, auto b) -> float { return std::max(a, b); }, std::numeric_limits<float>::lowest(),
                                            uram, inst.rs1, inst.rd, tcsr[group], 4
                                        );
                                        break;
                                    case 1: // tx.fp.reduce.min
                                        rvt64_handle_reduce_operation<float, float>(
                                            [](auto a, auto b) -> float { return std::min(a, b); }, std::numeric_limits<float>::max(),
                                            uram, inst.rs1, inst.rd, tcsr[group], 4
                                        );
                                        break;
                                    case 2: // tx.fp.reduce.sum
                                        rvt64_handle_reduce_operation<float, float>(
                                            [](auto a, auto b) -> float { return a + b; }, 0.f,
                                            uram, inst.rs1, inst.rd, tcsr[group], 4
                                        );
                                        break;
                                    default:
                                        debugf("RVT64: Unsupport tx.fp.reduce subop==%d!\n", subop);
                                        break;
                                }
                                break;
                            default:
                                debugf("RVT64: Unsupport Instruction %lx!\n",inst.inst);
                                break;  
                        }
                        break;
                    default:
                        debugf("RVT64: Unsupport Instruction %lx!\n",inst.inst);
                        break;
                }
                continue;
            }
        }
    }
}
