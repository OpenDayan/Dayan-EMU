#include "rvt64.hpp"
#include "swizzle.hpp"
#include <atomic>
#include <thread>

namespace RVTNG {
    void rvt64_t::issue(const rvt_pack_t* inst) {
        printf("RVT64: Enqueue RVT64 Instruction %lx!\n",inst->inst);
        // TODO: 实现 RVT64 指令发射
        bool is_rvt64 = (inst->inst & 0x7f) == 0x3f;
        if(!is_rvt64) {
            // ignore
            return;
        }
        this->inst_queue.enqueue(*inst);
    }

    rvt64_t::rvt64_t(exec_unit_t* parrent, uint8_t* uram) {
        // 实现 RVT64 构造函数
        parrent->register_issue_callback(this, &rvt64_t::issue);
        this->register_commit_callback(parrent, &exec_unit_t::commit);
        this->register_scoreboard_callback(parrent, &exec_unit_t::scoreboard);
    }

    void rvt64_t::execute(void) {
        // TODO: 实现 RVT64 执行
    }
}
