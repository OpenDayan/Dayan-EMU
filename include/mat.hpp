#ifndef __MAT_H__
#define __MAT_H__

#include "exec_unit.hpp"
#include "thread_safe_queue.h"

namespace RVTNG {

// TODO: 适配大小端情况
struct mat_tcsr_t {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t aStride;
    uint64_t bStride;
    uint64_t cStride;
    uint64_t aSwizzleMBS;
    uint64_t bSwizzleMBS;
    uint64_t cSwizzleMBS;
    uint64_t persistentMode;
    uint64_t dataType;
    uint64_t pad[20];
    uint64_t checksum;
};

class mat_t: exec_unit_t {
public:
    ulong issue(const rvt_pack_t* inst) override;
    mat_t(exec_unit_t* parrent, uint8_t *uram);
    void execute(void);  // 独占循环

protected:
    // 执行一条 BF16 MAT 指令
    void execute_mma_bf16_once(mat_tcsr_t* tcsr, uint64_t a, uint64_t b, uint64_t c);
    // 私有指令队列
    MPMCQueue<rvt_pack_t, 32> inst_queue;
    // 私有 TCSR
    mat_tcsr_t tcsr[8];
    // TCSR 写入掩码
    const char tcsr_width[32] = {
        20, 20, 20, 20, 20, 20, 20, 20, 20, 2, 8,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        20
    };
    // uram buffer
    uint8_t* uram;
};

}

#endif