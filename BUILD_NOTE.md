# 使用方式

## 固定前端模拟器启动

无参数启动 ./bin/rvt_emulator 时，默认采用模拟器前端参数：

`/workspace/npu-qemu/build/dl-riscv64 -monitor null -bios /workspace/npu-firmware/bl0/bootrom.elf -machine rvtng-v0 -nographic -m 4096 -S -s`

这种模式下，QEMU 启动并等待调试器链接，不会自动开始执行标量部分指令。


## 自定义前端模拟器启动

rvtng c model 支持选用自定义的 riscv 标量前端，只需要将 riscv 标量前端模拟器启动参数，作为参数传递给 ./bin/rvt_emulator 即可。

例如，启动无调试模式的 QEMU（仅快速仿真）：

`./bin/rvt_emulator /workspace/npu-qemu/build/dl-riscv64 -monitor null -bios /workspace/npu-firmware/bl0/bootrom.elf -machine rvtng-v0 -nographic -m 4096`
