#ifndef __RVT64_H__
#define __RVT64_H__

#include "exec_unit.hpp"
#include "thread_safe_queue.h"
#include <vector>

namespace RVTNG {

// TODO: 适配大小端情况
struct rvt64_tcsr_t {
};

#define RESQ_SIZE 16
class rvt64_t: exec_unit_t {
public:
    void issue(const rvt_pack_t* inst) override;
    rvt64_t(exec_unit_t* parrent, uint8_t* uram);
    void execute(void);  // 独占循环

protected:
    // 私有指令队列
    MPMCQueue<rvt_pack_t, 128> inst_queue;
};

}

#endif