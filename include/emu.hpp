#ifndef __EMU_H__
#define __EMU_H__

#include "thread_safe_queue.h"
#include "exec_unit.hpp"
#include "rvtng-intf.h"

namespace RVTNG {

// 仿真器后端基类实现
class rvtng_backend_t {
public:
    // 例化接口
    rvtng_backend_t(const char* backend_name);
    // 释放接口
    ~rvtng_backend_t();
protected:
    // IPC 接口指针
    int fdMsg;
    int fdInst;
    // 顶层执行单元
    exec_unit_t *top;
    // 子执行单元
    exec_unit_t *vec;
    exec_unit_t *mat;
    exec_unit_t *dma;
    // 存储资源
    void* uram;
};

}

#endif
