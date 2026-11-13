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

#pragma once

#include <cstdio>

// 调试信息打印，通过 NDEBUG 控制（Release 模式下自动剔除）
#ifdef NDEBUG
#define debugf(fmt, ...) do {} while (0)
#else
#define debugf(fmt, ...) printf("[DEBUG] " fmt, ##__VA_ARGS__)
#endif

// TODO: 目前仅支持 X86 平台
#define MFENCE() asm volatile("mfence" ::: "memory")
