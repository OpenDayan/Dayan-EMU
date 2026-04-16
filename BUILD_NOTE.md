# 使用方式

## 固定前端模拟器启动

无参数启动 ./bin/rvt_emulator 时，默认采用模拟器前端参数：

`/workspace/npu-qemu/build/dl-riscv64 -monitor null -bios /workspace/npu-firmware/bl0/bootrom.elf -machine rvtng-v0 -nographic -m 4096 -S -s`

这种模式下，QEMU 启动并等待调试器挂载。

## 自定义前端模拟器启动

rvtng-cmodel 支持选用自定义的 riscv 标量前端，只需要将 riscv 标量前端模拟器启动参数，作为参数传递给 ./bin/rvt_emulator 即可。

例如，启动无调试的 QEMU：

`./bin/rvt_emulator /workspace/npu-qemu/build/dl-riscv64 -monitor null -bios /workspace/npu-firmware/bl0/bootrom.elf -machine rvtng-v0 -nographic -m 4096`

## 无运行时环境启动

针对不希望使用运行时环境，希望能够直接运行裸机指令测试的场景，可以使用如下命令启动模拟器：
`./bin/rvt_emulator /workspace/npu-qemu/build/dl-riscv64 -monitor null -bios /workspace/npu-firmware/bl0-noruntime/bootrom.elf -kernel /workspace/npu-kernels/bin/kernels_pack.elf -bios /workspace/npu-firmware/bl0-noruntime/bootrom.elf -machine rvtng-v0 -nographic -m 4096 -serial none -monitor stdio -S -s`

模拟器会等待调试器挂载，并开始执行 npu-kernels 仓库中指定的默认测试算子。
