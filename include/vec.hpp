#ifndef __VEC_H__
#define __VEC_H__

#include "exec_unit.hpp"
#include "thread_safe_queue.h"

namespace RVTNG {

class vec_t: exec_unit_t {
public:
    void issue(const rvt_pack_t* inst) override;
    vec_t(exec_unit_t* parrent, ulong lane_width);
    void execute(void);  // 独占循环

protected:
    // 寄存器堆
    uint32_t *rf;
    // 掩码寄存器
    uint8_t *mask;
    // 指令队列0
    SPSCQueue<rvt_pack_t, 32> inst_queue;
    // 指令队列1
    std::vector<rvt_pack_t> inst_queue_block;
    // 其它状态变量及取指指针
    uint32_t pc;
    bool is_in_block;
};

}

#endif