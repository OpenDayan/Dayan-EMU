#ifndef __VEC_H__
#define __VEC_H__

#include "exec_unit.hpp"
#include "thread_safe_queue.h"
#include <vector>

namespace RVTNG {

// TODO: 适配大小端情况
struct vec_tcsr_t {
    uint64_t block;
    uint64_t dim0;
    uint64_t dim1;
    uint64_t dim0VStride0;
    uint64_t dim1VStride0;
    uint64_t dim0SStride0;
    uint64_t dim1SStride0;
    uint64_t swizzleMBS0;
    uint64_t dim0VStride1;
    uint64_t dim1VStride1;
    uint64_t dim0SStride1;
    uint64_t dim1SStride1;
    uint64_t swizzleMBS1;
    uint64_t dim0VStride2;
    uint64_t dim1VStride2;
    uint64_t dim0SStride2;
    uint64_t dim1SStride2;
    uint64_t swizzleMBS2;
    uint64_t dim0VStride3;
    uint64_t dim1VStride3;
    uint64_t dim0SStride3;
    uint64_t dim1SStride3;
    uint64_t swizzleMBS3;
    uint64_t pad[8];
    uint64_t checksum;
};

#define RESQ_SIZE 16
struct ResQ {
    std::array<std::vector<int>, 2> buffer;

    int lane_width;
    int head[2] = {};

    explicit ResQ(int lane_width) : lane_width(lane_width) {
        for (size_t vb = 0; vb < 2; ++vb)
            buffer[vb].resize(RESQ_SIZE * lane_width);
    }

    void enqueue(int vb, int result) {
        buffer[vb][head[vb]] = result;
        head[vb] = (head[vb] + 1) % RESQ_SIZE;
    }

    int get(int vb, int index){
        int realIndex = (head[vb] - index + RESQ_SIZE) % RESQ_SIZE;
        return buffer[vb][realIndex];
    }
};

class vec_t: exec_unit_t {
public:
    void issue(const rvt_pack_t* inst) override;
    vec_t(exec_unit_t* parrent, uint8_t *uram, ulong lane_width);
    void execute(void);  // 独占循环

protected:
    // 执行一条 VEC 指令
    void exec_once(rvt_pack_t inst);
    // 寄存器堆
    uint32_t *rf;
    // 掩码寄存器
    uint8_t *mask;
    // 指令队列0
    MPMCQueue<rvt_pack_t, 32> inst_queue;
    // 指令队列1
    std::vector<rvt_pack_t> inst_queue_block;
    // 私有 TCSR
    vec_tcsr_t tcsr[8];
    // TCSR 写入掩码
    const char tcsr_width[32] = {
        32, 16, 16, 28, 28, 28, 28, 24, 28, 28, 28, 28, 24, 
        8, 8, 28, 28, 24, 28, 8, 8, 8, 24, 0, 0, 0, 0, 0, 0, 0, 0, 
        20
    };
    // uram buffer
    uint8_t* uram;
    // 存储 lane 宽度
    ulong lane_width;
    // resQ[vb][idx][lane]
    ResQ resQ;

    enum {OUTER_MODE, INNER_MODE};
    bool mode;

    int curBlock, curDim0, curDim1;
    // 其它状态变量及取指指针
    uint32_t pc;
    bool is_in_block;
};

}

#endif