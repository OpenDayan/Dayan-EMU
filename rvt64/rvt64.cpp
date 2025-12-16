#include "rvt64.hpp"
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
                cmt_pack.data = helper_tcsr_xchg(tcsr_id, inst.rs1,  inst.rd);
                commit(&cmt_pack);
                printf("* mask %lx -> %lx.\n", cmt_pack.data, inst.rs1);
                break;
            default:
                printf("RVT64: Unsupport Instruction %lx!\n",inst.inst);
                break;
            }
        }
    }
}
