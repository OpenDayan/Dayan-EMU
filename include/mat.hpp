#ifndef __MAT_H__
#define __MAT_H__

#include "exec_unit.hpp"
#include "thread_safe_queue.h"

namespace RVTNG {

// TODO: 适配大小端情况
// TCSR 寄存器布局 — 字段顺序对应物理寄存器 index 0-31
struct mat_tcsr_t {
    uint64_t k;                       // index 0:  K 维大小
    uint64_t pad0;                    // index 1:  保留
    uint64_t pad1;                    // index 2:  保留
    uint64_t m;                       // index 3:  M 维大小
    uint64_t pad2;                    // index 4:  保留
    uint64_t pad3;                    // index 5:  保留
    uint64_t pad4;                    // index 6:  保留
    uint64_t n;                       // index 7:  N 维大小
    uint64_t aStride;                 // index 8:  A 矩阵行步长
    uint64_t pad5;                    // index 9:  保留
    uint64_t pad6;                    // index 10: 保留
    uint64_t pad7;                    // index 11: 保留
    uint64_t bStride;                 // index 12: B 矩阵列步长
    uint64_t cStride;                 // index 13: C 矩阵行步长
    uint64_t pad8;                    // index 14: 保留
    uint64_t aSwizzleMBS;             // index 15: A 矩阵 Swizzle 配置
    uint64_t bSwizzleMBS;             // index 16: B 矩阵 Swizzle 配置
    uint64_t cSwizzleMBS;             // index 17: C 矩阵 Swizzle 配置
    uint64_t dataType;                // index 18: 数据类型编码
    uint64_t aScaleStride;            // index 19: A Scale 矩阵行步长
    uint64_t pad9;                    // index 20: 保留
    uint64_t pad10;                   // index 21: 保留
    uint64_t pad11;                   // index 22: 保留
    uint64_t bScaleStride;            // index 23: B Scale 矩阵列步长
    uint64_t pad12;                   // index 24: 保留
    uint64_t pad13;                   // index 25: 保留
    uint64_t pad14;                   // index 26: 保留
    uint64_t pad15;                   // index 27: 保留
    uint64_t pad16;                   // index 28: 保留
    uint64_t pad17;                   // index 29: 保留
    uint64_t pad18;                   // index 30: 保留
    uint64_t pad19;                   // index 31: 保留
};

// TCSR 卷积寄存器布局 — 字段顺序对应物理寄存器 index 0-31
struct conv_tcsr_t {
    uint64_t c;                       // index 0:  输入通道数 C
    uint64_t wi;                      // index 1:  输入宽度 Wi
    uint64_t hi;                      // index 2:  输入高度 Hi
    uint64_t wo;                      // index 3:  输出宽度 Wo
    uint64_t ho;                      // index 4:  输出高度 Ho
    uint64_t wk;                      // index 5:  卷积核宽度 Wk
    uint64_t hk;                      // index 6:  卷积核高度 Hk
    uint64_t co;                      // index 7:  输出通道数 Co
    uint64_t wiStride;                // index 8:  输入 Wi 维度步长
    uint64_t hiStride;                // index 9:  输入 Hi 维度步长
    uint64_t wkStride;                // index 10: 卷积核 Wk 维度步长
    uint64_t hkStride;                // index 11: 卷积核 Hk 维度步长
    uint64_t coStride;                // index 12: 卷积核 Co 维度步长
    uint64_t woStride;                // index 13: 输出 Wo 维度步长
    uint64_t hoStride;                // index 14: 输出 Ho 维度步长
    uint64_t iSwizzleMBS;             // index 15: 输入张量 Swizzle 配置
    uint64_t kSwizzleMBS;             // index 16: 卷积核 Swizzle 配置
    uint64_t oSwizzleMBS;             // index 17: 输出张量 Swizzle 配置
    uint64_t dataType;                // index 18: 数据类型编码
    uint64_t scaleWiStride;           // index 19: 输入 Scale Wi 维度步长
    uint64_t scaleHiStride;           // index 20: 输入 Scale Hi 维度步长
    uint64_t scaleWkStride;           // index 21: 卷积核 Scale Wk 维度步长
    uint64_t scaleHkStride;           // index 22: 卷积核 Scale Hk 维度步长
    uint64_t scaleCoStride;           // index 23: 卷积核 Scale Co 维度步长
    uint64_t padL;                    // index 24: 左侧填充数量
    uint64_t padU;                    // index 25: 上方填充数量
    uint64_t dilationW;               // index 26: 水平扩张率
    uint64_t dilationH;               // index 27: 垂直扩张率
    uint64_t strideW;                 // index 28: 水平步长
    uint64_t strideH;                 // index 29: 垂直步长
    uint64_t strideDilationW;         // index 30: 扩张后卷积核水平内存步长
    uint64_t strideDilationH;         // index 31: 扩张后卷积核垂直内存步长
};

class mat_t: exec_unit_t {
public:
    ulong issue(const rvt_pack_t* inst) override;
    mat_t(exec_unit_t* parrent, uint8_t *uram);
    void execute(void);  // 独占循环

protected:
    // 执行一条 BF16 MAT 指令
    void execute_mm_bf16_once(mat_tcsr_t* tcsr, uint64_t a, uint64_t b, uint64_t c, bool acc);
    // 私有指令队列
    MPMCQueue<rvt_pack_t, 32> inst_queue;
    // 私有 TCSR
    mat_tcsr_t tcsr[8];
    // TCSR 写入掩码（按 conv_tcsr_t 规范定义，每个 offset 的位宽）
    // c,wi,hi,wo,ho,wk,hk,co, wiStride,hiStride,wkStride,hkStride,coStride,woStride,hoStride,
    // iSwizzleMBS,kSwizzleMBS,oSwizzleMBS, dataType,
    // scaleWiStride,scaleHiStride,scaleWkStride,scaleHkStride,scaleCoStride,
    // padL,padU, dilationW,dilationH, strideW,strideH, strideDilationW,strideDilationH
    const char tcsr_width[32] = {
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        24, 24, 24, 8,
        20, 20, 20, 20, 20,
        8, 8, 8, 8, 8, 8, 20, 20
    };
    // uram buffer
    uint8_t* uram;
};

}

#endif