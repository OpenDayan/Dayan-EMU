#include "debugf.hpp"
#include "rvt64.hpp"
#include "rvt64_helper.hpp"
#include "swizzle.hpp"
#include "fputils.hpp"
#include <atomic>
#include <thread>
#include <cmath>

namespace RVTNG {
    ulong rvt64_t::issue(const rvt_pack_t* inst) {
        bool is_rvt64 = (inst->inst & 0x7f) == 0x3f;
        if(!is_rvt64) {
            return 0;
        }
        this->inst_queue.enqueue(*inst);
        debugf("RVT64: Enqueue RVT64 Instruction %lx!\n",inst->inst);
        return 1;
    }

    rvt64_t::rvt64_t(exec_unit_t* parrent, uint8_t* uram): uram(uram) {
        // 实现 RVT64 构造函数
        parrent->register_issue_callback(this, &rvt64_t::issue);
        this->register_commit_callback(parrent, &exec_unit_t::commit);
        this->register_scoreboard_callback(parrent, &exec_unit_t::scoreboard);
    }

    ulong rvt64_t::helper_tcsr_xchg(ulong tcsr_id, ulong value, ulong mask) {
        if(tcsr_id > 8 * 8) {
            debugf("* Illgal id.\n");
            return 0;
        }
        ulong* _tcsr = (uint64_t*) &this->tcsr;
        ulong old = _tcsr[tcsr_id];
        _tcsr[tcsr_id] = (value & mask) | (old & (~mask));
        return old;
    }

    void rvt64_t::execute(void) {
        while(1) {
            rvt_commit_t cmt_pack;
            // Fetch
            auto inst = inst_queue.dequeue();
            // Decode
            uint8_t func = inst.inst >> 56;
            uint16_t tcsr_id = (inst.inst >> 32) & 0x3ff;
            uint8_t rd = (inst.inst >> 7) & 0x1f;
            uint8_t broadcast = (inst.inst >> 45) & 0xf;
            uint8_t tcfgrs1 = (inst.inst >> 32) & 0x7;
            uint8_t tcfgrs2 = (inst.inst >> 35) & 0x7;
            uint8_t tcfgrs3 = (inst.inst >> 38) & 0x7;
            uint8_t subop = (inst.inst >> 50) & 0x3f;
            uint16_t rs1type = ((inst.inst >> 12) & 0x7) | (((inst.inst >> 41) & 0xf) << 3);
            uint16_t rs3type = (inst.inst >> 45) & 0x7f;
            uint8_t reduce_dim = (inst.inst >> 25) & 0x3;
            switch (func) {
            case 0b10000000: // tx.tcsr.xchg.nmask
                cmt_pack.timestamp = inst.timestamp;
                cmt_pack.rd = rd;
                cmt_pack.data = helper_tcsr_xchg(tcsr_id, inst.rs1, ~inst.rd);
                commit(&cmt_pack);
                debugf("* nmask %lx -> %lx.\n", cmt_pack.data, inst.rs1);
                break;
            case 0b10000001: // tx.tcsr.xchg.mask
                cmt_pack.timestamp = inst.timestamp;
                cmt_pack.rd = rd;
                cmt_pack.data = helper_tcsr_xchg(tcsr_id, inst.rs1, inst.rd);
                commit(&cmt_pack);
                debugf("* mask %lx -> %lx.\n", cmt_pack.data, inst.rs1);
                break;
            case 0b11111111: // tx.sb
                if((inst.inst >> 52) & 0x1) {
                    // Bar
                    debugf("RVT64: scoreboard barrier %lx:\n", inst.rd);
                    cmt_pack.timestamp = inst.timestamp;
                    cmt_pack.rd = rd;
                    cmt_pack.data = inst.rd;
                    commit(&cmt_pack);
                } else {
                    // Produce
                    debugf("RVT64: scoreboard produce %lx:\n", inst.rd);
                    scoreboard(inst.rd);
                }
                break;
            case 0b00001000: // tx.int
                switch (subop) {
                case 0: // tx.int.add
                    rvt64_handle_binary_operation<int32_t, int32_t>(
                        [](auto a, auto b) -> int32_t { return a + b; },
                        uram, broadcast, inst.rs1, tcsr[tcfgrs1], inst.rs2, tcsr[tcfgrs2], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                case 1: // tx.int.sub
                    rvt64_handle_binary_operation<int32_t, int32_t>(
                        [](auto a, auto b) -> int32_t { return a - b; },
                        uram, broadcast, inst.rs1, tcsr[tcfgrs1], inst.rs2, tcsr[tcfgrs2], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                case 2: // tx.int.mul
                    rvt64_handle_binary_operation<int32_t, int32_t>(
                        [](auto a, auto b) -> int32_t { return a * b; },
                        uram, broadcast, inst.rs1, tcsr[tcfgrs1], inst.rs2, tcsr[tcfgrs2], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                case 3: // tx.int.srl
                    rvt64_handle_binary_operation<uint32_t, uint32_t>(
                        [](auto a, auto b) -> uint32_t { return a >> b; },
                        uram, broadcast, inst.rs1, tcsr[tcfgrs1], inst.rs2, tcsr[tcfgrs2], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                case 4: // tx.int.sra
                    rvt64_handle_binary_operation<int32_t, int32_t>(
                        [](auto a, auto b) -> int32_t { return a >> b; },
                        uram, broadcast, inst.rs1, tcsr[tcfgrs1], inst.rs2, tcsr[tcfgrs2], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                case 5: // tx.int.sll
                    rvt64_handle_binary_operation<uint32_t, uint32_t>(
                        [](auto a, auto b) -> uint32_t { return a << b; },
                        uram, broadcast, inst.rs1, tcsr[tcfgrs1], inst.rs2, tcsr[tcfgrs2], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                default:
                    if(subop > 16) {
                        // LUT
                        uint8_t lut = subop & 0xf;
                        rvt64_handle_binary_operation<uint32_t, uint32_t>(
                            [lut](auto a, auto b) -> uint32_t {
                                auto v00 = ~(a | b); // 00
                                auto v01 = (~a) & b; // 01
                                auto v10 = a & (~b); // 10
                                auto v11 = a & b;    // 11
                                return (v00 * ((lut & 0x1) != 0)) | (v01 * ((lut & 0x2) != 0)) |
                                       (v10 * ((lut & 0x4) != 0)) | (v11 * ((lut & 0x8) != 0));
                            },
                            uram, broadcast, inst.rs1, tcsr[tcfgrs1], inst.rs2, tcsr[tcfgrs2], inst.rd, tcsr[tcfgrs3]
                        );
                    } else {
                        debugf("RVT64: Unsupport tx.int subop==%d!\n", subop);
                        break;
                    }
                }
                break;
            case 0b00001001: // tx.fp
                switch (subop) {
                case 0: // tx.fp.add
                    rvt64_handle_binary_operation<float, float>(
                        [](auto a, auto b) -> float { return a + b; },
                        uram, broadcast, inst.rs1, tcsr[tcfgrs1], inst.rs2, tcsr[tcfgrs2], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                case 1: // tx.fp.sub
                    rvt64_handle_binary_operation<float, float>(
                        [](auto a, auto b) -> float { return a - b; },
                        uram, broadcast, inst.rs1, tcsr[tcfgrs1], inst.rs2, tcsr[tcfgrs2], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                case 2: // tx.fp.mul
                    rvt64_handle_binary_operation<float, float>(
                        [](auto a, auto b) -> float { return a * b; },
                        uram, broadcast, inst.rs1, tcsr[tcfgrs1], inst.rs2, tcsr[tcfgrs2], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                default:
                    debugf("RVT64: Unsupport tx.fp subop==%d!\n", subop);
                    break;
                }
                break;
            case 0b10010000: // tx.sfu
                switch (subop) {
                case 0: // tx.sfu.recip
                    rvt64_handle_unary_operation<float, float>(
                            [](auto x) -> float { return 1.0f / x; },
                            uram, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                        );
                    break;
                case 1: // tx.sfu.rsqrt
                    rvt64_handle_unary_operation<float, float>(
                            [](auto x) -> float { return 1.0f / sqrtf(x); },
                            uram, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                        );
                    break;
                case 2: // tx.sfu.sin
                    rvt64_handle_unary_operation<float, float>(
                            [](auto x) -> float { return sinf(x); },
                            uram, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                        );
                    break;
                case 3: // tx.sfu.cos
                    rvt64_handle_unary_operation<float, float>(
                            [](auto x) -> float { return cosf(x); },
                            uram, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                        );
                    break;
                case 4: // tx.sfu.exp
                    rvt64_handle_unary_operation<float, float>(
                            [](auto x) -> float { return expf(x); },
                            uram, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                        );
                    break;
                case 5: // tx.sfu.relu
                    rvt64_handle_unary_operation<float, float>(
                            [](auto x) -> float { return x < 0.f ? 0.f : x; },
                            uram, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                        );
                    break;
                default:
                    debugf("RVT64: Unsupport tx.sfu subop==%d!\n", subop);
                    break;
                }
                break;
            case 0b10001110: // tx.convert
                // 目前仅支持 FP -> S32 / BF16 / FP16 / FP8E4M3 的转换
                switch (rs1type) {
                case 0b1001100: // FP32
                    switch (rs3type) {
                    case 0b1000101: // S32
                        // TODO: Support Round Mode
                        rvt64_handle_unary_operation<float, int32_t>(
                            [](auto x) -> int32_t { return static_cast<int32_t>(x); },
                            uram, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                        );
                        break;
                    case 0b1001011: // BF16
                        rvt64_handle_unary_operation<float, uint16_t>(
                            FP32ToBF16,
                            uram, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                        );
                        break;
                    case 0b1001100: // FP16
                        rvt64_handle_unary_operation<float, uint16_t>(
                            FP32ToFP16,
                            uram, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                        );
                        break;
                    case 0b0011100: // FP8E4M3
                        rvt64_handle_unary_operation<float, uint8_t>(
                            FP32ToFP8<4>,
                            uram, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
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
            case 0b10001100: // tx.int.reduce
                switch (subop) {
                case 0: // tx.int.reduce.max
                    rvt64_handle_reduce_operation<int32_t, int32_t>(
                        [](auto a, auto b) -> int32_t { return std::max(a, b); }, 0xffffffff,
                        uram, reduce_dim, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                case 1: // tx.int.reduce.min
                    rvt64_handle_reduce_operation<int32_t, int32_t>(
                        [](auto a, auto b) -> int32_t { return std::min(a, b); }, 0x7fffffff,
                        uram, reduce_dim, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                case 2: // tx.int.reduce.sum
                    rvt64_handle_reduce_operation<int32_t, int32_t>(
                        [](auto a, auto b) -> int32_t { return a + b; }, 0,
                        uram, reduce_dim, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                default:
                    debugf("RVT64: Unsupport tx.int.reduce subop==%d!\n", subop);
                    break;
                }
                break;
            case 0b10001101: // tx.fp.reduce
                switch (subop) {
                case 0: // tx.fp.reduce.max
                    rvt64_handle_reduce_operation<float, float>(
                        [](auto a, auto b) -> float { return std::max(a, b); }, std::numeric_limits<float>::lowest(),
                        uram, reduce_dim, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                case 1: // tx.fp.reduce.min
                    rvt64_handle_reduce_operation<float, float>(
                        [](auto a, auto b) -> float { return std::min(a, b); }, std::numeric_limits<float>::max(),
                        uram, reduce_dim, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
                    );
                    break;
                case 2: // tx.fp.reduce.sum
                    rvt64_handle_reduce_operation<float, float>(
                        [](auto a, auto b) -> float { return a + b; }, 0.f,
                        uram, reduce_dim, inst.rs1, tcsr[tcfgrs1], inst.rd, tcsr[tcfgrs3]
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
        }
    }
}
