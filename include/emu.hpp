/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-emu is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#ifndef __EMU_H__
#define __EMU_H__

#include "thread_safe_queue.h"
#include "exec_unit.hpp"
#include "dayan-intf.h"

namespace DAYAN {

// 仿真器后端基类实现
class dayan_backend_t {
public:
    // 例化接口
    dayan_backend_t(const char* backend_name);
    // 释放接口
    ~dayan_backend_t();
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
