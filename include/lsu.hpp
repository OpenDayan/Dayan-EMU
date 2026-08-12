#ifndef __LSU_H__
#define __LSU_H__

#include "exec_unit.hpp"
#include "thread_safe_queue.h"
#include <cstddef>

namespace RVTNG {

// TODO: 适配大小端情况
struct lsu_tcsr_t {
    uint64_t block;
    uint64_t dim0;
    uint64_t dim1;
    uint64_t dim2;
    uint64_t inBlockStride;
    uint64_t inDim0Stride;
    uint64_t inDim1Stride;
    uint64_t inDim2Stride;
    uint64_t outBlockStride;
    uint64_t outDim0Stride;
    uint64_t outDim1Stride;
    uint64_t outDim2Stride;
    uint64_t blockTranspose;
    uint64_t blockPadding;
    union {
        uint64_t requestReadSize;
        uint64_t requestPadding; // Legacy name.
    };
    uint8_t  paddingValue[8];
    uint64_t uramWriteSwizzleMBS;
    uint64_t uramReadSwizzleMBS;
    uint64_t requestWriteSize;
    uint64_t tableSize;
    uint64_t tableStride;
    uint64_t pad[10];
    uint64_t checksum;
};
static_assert(sizeof(lsu_tcsr_t) == 32 * sizeof(uint64_t));
static_assert(offsetof(lsu_tcsr_t, requestWriteSize) == 18 * sizeof(uint64_t));
static_assert(offsetof(lsu_tcsr_t, tableSize) == 19 * sizeof(uint64_t));
static_assert(offsetof(lsu_tcsr_t, tableStride) == 20 * sizeof(uint64_t));

class lsu_t: exec_unit_t {
public:
    ulong issue(const rvt_pack_t* inst) override;
    lsu_t(exec_unit_t* parrent, uint8_t *uram, uint8_t *dram,
          uint8_t *const *cluster_urams = nullptr,
          size_t cluster_core_count = 0);
    ~lsu_t();
    void execute(void);  // 独占循环

protected:
    uint8_t* tensor_uram_address(uint64_t addr, uint64_t size) const;
    // 执行一条 DMA 指令，包括批量任务与地址表展开。
    void execute_dma(lsu_tcsr_t* tcsr, uint64_t src, uint64_t dst,
                     bool src_dram, bool dst_dram, bool gather, bool scatter);
    // 执行一个 DMA 子任务。
    bool execute_dma_once(lsu_tcsr_t* tcsr, uint64_t src, uint64_t dst, bool src_dram, bool dst_dram);
    // 私有指令队列
    MPMCQueue<rvt_pack_t, 128> inst_queue;
    // 私有 LSU buffer
    uint8_t* lsu_buffer;
    // 私有 TCSR
    lsu_tcsr_t tcsr[8];
    // TCSR 写入掩码
    const char tcsr_width[32] = {
        8, 24, 24, 24, 28, 28, 28, 28, 28, 28, 28, 28, 3, 8, 6, 64, 24, 24,
        6, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        20
    };
    // uram buffer
    uint8_t* uram;
    // dram
    uint8_t* dram;
    // URAM windows exposed on the tensor/D-side address range.
    uint8_t *const *cluster_urams;
    size_t cluster_core_count;
};

}

#endif
