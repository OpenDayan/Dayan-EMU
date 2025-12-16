#include "rvt64.hpp"
#include "rvt64_helper.hpp"
#include "swizzle.hpp"
#include <atomic>
#include <thread>

namespace RVTNG {
    ulong rvt64_t::issue(const rvt_pack_t* inst) {
        bool is_rvt64 = (inst->inst & 0x7f) == 0x3f;
        if(!is_rvt64) {
            return 0;
        }
        this->inst_queue.enqueue(*inst);
        printf("RVT64: Enqueue RVT64 Instruction %lx!\n",inst->inst);
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
            printf("* Illgal id.\n");
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
            switch (func) {
            case 0b10000000: // tx.tcsr.xchg.nmask
                cmt_pack.timestamp = inst.timestamp;
                cmt_pack.rd = rd;
                cmt_pack.data = helper_tcsr_xchg(tcsr_id, inst.rs1, ~inst.rd);
                commit(&cmt_pack);
                printf("* nmask %lx -> %lx.\n", cmt_pack.data, inst.rs1);
                break;
            case 0b10000001: // tx.tcsr.xchg.mask
                cmt_pack.timestamp = inst.timestamp;
                cmt_pack.rd = rd;
                cmt_pack.data = helper_tcsr_xchg(tcsr_id, inst.rs1, inst.rd);
                commit(&cmt_pack);
                printf("* mask %lx -> %lx.\n", cmt_pack.data, inst.rs1);
                break;
            case 0b11111111: // tx.sb
                if((inst.inst >> 52) & 0x1) {
                    // Bar
                    printf("RVT64: scoreboard barrier %lx:\n", inst.rd);
                    cmt_pack.timestamp = inst.timestamp;
                    cmt_pack.rd = rd;
                    cmt_pack.data = inst.rd;
                    commit(&cmt_pack);
                } else {
                    // Produce
                    printf("RVT64: scoreboard produce %lx:\n", inst.rd);
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
                                return (v00 * (lut & 0x1)) | (v01 * (lut & 0x2)) |
                                       (v10 * (lut & 0x4)) | (v11 * (lut & 0x8));
                            },
                            uram, broadcast, inst.rs1, tcsr[tcfgrs1], inst.rs2, tcsr[tcfgrs2], inst.rd, tcsr[tcfgrs3]
                        );
                    } else {
                        printf("RVT64: Unsupport tx.int subop==%d!\n", subop);
                        break;
                    }
                }
                break;
            default:
                printf("RVT64: Unsupport Instruction %lx!\n",inst.inst);
                break;
            }
        }
    }
}
