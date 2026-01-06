#pragma once

// 控制调试信息打印
// #define debugf(fmt, ...) printf("[DEBUG] " fmt, ##__VA_ARGS__)
#define debugf(fmt, ...) do {} while (0)

// TODO: 目前仅支持 X86 平台
#define MFENCE() asm volatile("mfence" ::: "memory")
