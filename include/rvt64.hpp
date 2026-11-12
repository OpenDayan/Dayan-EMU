#ifndef __RVT64_H__
#define __RVT64_H__

#include "exec_unit.hpp"
#include "thread_safe_queue.h"
#include "tensor_context.hpp"
#include "interrupt.hpp"
#include <vector>

namespace DAYAN {

struct rvt64_tcsr_t {
    uint64_t dim0;
    uint64_t dim1;
    uint64_t dim2;
    uint64_t dim3;
    uint64_t Src0Dim1Stride;
    uint64_t Src0Dim2Stride;
    uint64_t Src0Dim3Stride;
    uint64_t Src1Dim1Stride;
    uint64_t Src1Dim2Stride;
    uint64_t Src1Dim3Stride;
    uint64_t DstDim1Stride;
    uint64_t DstDim2Stride;
    uint64_t DstDim3Stride;
    uint64_t Src0SwizzleMBS;
    uint64_t Src1SwizzleMBS;
    uint64_t DstSwizzleMBS;
    uint64_t ImplDefHint;
    uint64_t pad[14];
    uint64_t checksum;
};

#define RESQ_SIZE 16
class rvt64_t: exec_unit_t {
public:
    ulong issue(const dayan_pack_t* inst) override;
    rvt64_t(exec_unit_t* parrent, uint8_t* uram,
            tensor_context_t* context = nullptr,
            interrupt_controller_t* interrupts = nullptr);
    void execute(void);  // 独占循环
    void enqueue_context(const dayan_pack_t& inst,
                         const tensor_context_snapshot_t& context);

protected:
    // 私有指令队列
    MPMCQueue<queued_tensor_inst_t, 128> inst_queue;
    // URAM
    uint8_t* uram;
    // TCSR
    rvt64_tcsr_t tcsr[16];
    tensor_context_t* context;
    interrupt_controller_t* interrupts;
    tensor_context_snapshot_t active_context;
    // TCSR 写入掩码
    const char tcsr_width[32] = {
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        24, 24, 24, 24, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        20
    };
    bool validate_tensor(uint64_t base, const rvt64_tcsr_t& control,
                         unsigned operand, unsigned element_bytes,
                         bool collapse_dim0, bool write,
                         interrupt_source_t range_source,
                         interrupt_source_t align_source) const;
    bool validate_instruction(const dayan_pack_t& inst,
                              const rvt64_tcsr_t& control) const;
    // TCSR Helper
};

}

#endif
