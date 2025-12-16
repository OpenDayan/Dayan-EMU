#ifndef __RVT64_H__
#define __RVT64_H__

#include "exec_unit.hpp"
#include "thread_safe_queue.h"
#include <vector>

namespace RVTNG {

struct rvt64_tcsr_t {
    uint64_t dim0;
    uint64_t stride0;
    uint64_t dim1;
    uint64_t stride1;
    uint64_t dim2;
    uint64_t stride2;
    uint64_t dim3;
    uint64_t stride3;
};

#define RESQ_SIZE 16
class rvt64_t: exec_unit_t {
public:
    ulong issue(const rvt_pack_t* inst) override;
    rvt64_t(exec_unit_t* parrent, uint8_t* uram);
    void execute(void);  // 独占循环

protected:
    // 私有指令队列
    MPMCQueue<rvt_pack_t, 128> inst_queue;
    // URAM
    uint8_t* uram;
    // TCSR
    rvt64_tcsr_t tcsr[8];
    // TCSR 写入掩码
    const char tcsr_width[8] = {
        32, 32, 32, 32,
        32, 32, 32, 32
    };
    // TCSR Helper
    ulong helper_tcsr_xchg(ulong tcsr_id, ulong value, ulong mask);
};

}

#endif