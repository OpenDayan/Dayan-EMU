#include "rvtng-intf.h"
#include "exec_unit.hpp"
#include "lsu.hpp"
#include <thread>

int main() {
    int fdMsg, fdInst;
    // 创建 URAM
    uint8_t* uram = (uint8_t *)backend_get_shm_ptr("/uram_shm_0", 1024 * 1024);
    // 创建 QEMU 子进程
    pid_t qemuPid = fork();
    if (qemuPid == -1) {
        return -1;
    }
    else if (qemuPid == 0) {
        execlp("/workspace/npu-qemu/build/dl-riscv64", "QEMU_THREAD", "-monitor", "null",
        "-bios", "/workspace/npu-firmware/bl0/bootrom.elf",
        "-machine", "rvtng-v0", "-nographic", "-m", "4096", "-S", "-s", NULL);
        exit(EXIT_FAILURE);
    }
    // 父进程创建 server socket
    backend_init((int*)&fdMsg, (int*)&fdInst);
    printf("Main: QEMU Ready!\n");
    // 创建顶层
    auto top = RVTNG::exec_unit_t();
    // 绑定顶层提交函数
    top.register_commit_callback([fdMsg](const rvt_commit_t* pkg) {
        printf("* Commit: %lx -> %lx\n", pkg->data, pkg->rd);
        backend_instruction_commit(fdMsg, 0, pkg);
    });
    // 顶层提交 Scoreboard
    auto sb = RVTNG::MPMCQueue<uint64_t, 32>();
    top.register_scoreboard_callback([&sb](const uint64_t entry) {
        sb.enqueue(entry);
    });
    // 创建 LSU
    auto lsu = RVTNG::lsu_t(&top, uram, fdMsg);
    // 创建 Scoreboard
    top.register_issue_callback([&top, &sb](const rvt_pack_t* inst) {
        bool is_sb = (inst->inst & 0xfe000000) == 0x0e000000;
        if(!is_sb) return;
        rvt_commit_t cmt_pack;
        cmt_pack.timestamp = inst->timestamp;
        cmt_pack.rd = (inst->inst >> 7) & 0x1f;
        bool v = sb.try_dequeue(cmt_pack.data);
        if(!v) cmt_pack.data = 0;
        top.commit(&cmt_pack);
        printf("Scoreboard: Execute scoreboard Instruction %x!\n",(uint32_t)inst->inst);
    });
    // 创建子执行线程
    std::thread lsu_exec(&RVTNG::lsu_t::execute, &lsu);
    // 主要循环体
    while(1) {
        // 取指令
        rvt_pack_t inst;
        int count = backend_instruction_fetch(fdInst, 0, &inst);
        if(count) top.issue(&inst);
        else usleep(100); // 10kHz
    }
}
