#pragma once

// 调试信息打印，通过 NDEBUG 控制（Release 模式下自动剔除）
#ifdef NDEBUG
#define debugf(fmt, ...) do {} while (0)
#else
#define debugf(fmt, ...) printf("[DEBUG] " fmt, ##__VA_ARGS__)
#endif

// TODO: 目前仅支持 X86 平台
#define MFENCE() asm volatile("mfence" ::: "memory")
