#include "rvtng-intf.h"
#include "exec_unit.hpp"
#include "lsu.hpp"
#include "mat.hpp"
#include "vec.hpp"
#include "thread_safe_queue.h"
#include <thread>
#include <optional>
#include <mutex>

int main(int argc, char** argv) {
    int fdMsg, fdInst;
    // 注意 fdMsg 存在竞争问题，例化一个锁避免相关问题出现。
    std::mutex fdMsgMutex;
    // 创建 URAM
    uint8_t* uram = (uint8_t *)backend_get_shm_ptr("/uram_shm_0", 1024 * 1024);
    // 创建 QEMU 子进程
    pid_t qemuPid = fork();
    if (qemuPid == -1) {
        return -1;
    }
    else if (qemuPid == 0) {
        if(argc <= 1) {// Default debug mode.
            execlp("/workspace/npu-qemu/build/dl-riscv64", "QEMU_THREAD", "-monitor", "null",
            "-bios", "/workspace/npu-firmware/bl0/bootrom.elf",
            "-machine", "rvtng-v0", "-nographic", "-m", "4096", "-S", "-s", NULL);
            exit(EXIT_FAILURE);
        } else {
            execvp(argv[1], &argv[1]);
            exit(EXIT_FAILURE);
        }
    }
    // 父进程创建 server socket
    backend_init((int*)&fdMsg, (int*)&fdInst);
    printf("Main: QEMU Ready!\n");
    // 创建顶层
    auto top = RVTNG::exec_unit_t();
    // 绑定顶层提交函数
    top.register_commit_callback([fdMsg, &fdMsgMutex](const rvt_commit_t* pkg) {
        // printf("* Commit: %lx -> %lx\n", pkg->data, pkg->rd);
        std::lock_guard<std::mutex> lock(fdMsgMutex);
        backend_instruction_commit(fdMsg, 0, pkg);
    });
    // 顶层提交 Scoreboard
    auto sb = RVTNG::MPMCQueue<uint64_t, 32>();
    top.register_scoreboard_callback([&sb](const uint64_t entry) {
        sb.enqueue(entry);
    });
    // 创建 LSU
    auto lsu = RVTNG::lsu_t(&top, uram, fdMsg, &fdMsgMutex);
    // 创建 MAT
    auto mat = RVTNG::mat_t(&top, uram);
    // 创建 VEC
    auto vec = RVTNG::vec_t(&top, uram, 16);
    // 创建 Scoreboard
    top.register_issue_callback([&top, &sb](const rvt_pack_t* inst) {
        bool is_sb = (inst->inst & 0xfe000000) == 0x0e000000;
        if(!is_sb) return;
        rvt_commit_t cmt_pack;
        cmt_pack.timestamp = inst->timestamp;
        cmt_pack.rd = (inst->inst >> 7) & 0x1f;
        auto p = sb.try_dequeue();
        if(p != std::nullopt) printf("Scoreboard: empty:%x data:%lx!\n", p == std::nullopt, p.value_or(0ul));
        cmt_pack.data = p.value_or(0ul);
        top.commit(&cmt_pack);
    });
    // 创建子执行线程
    std::thread lsu_exec(&RVTNG::lsu_t::execute, &lsu);
    std::thread mat_exec(&RVTNG::mat_t::execute, &mat);
    std::thread vec_exec(&RVTNG::vec_t::execute, &vec);
    // 主要循环体
    while(1) {
        // 取指令
        rvt_pack_t inst;
        int count = backend_instruction_fetch(fdInst, 0, &inst);
        if(count) top.issue(&inst);
        else usleep(100); // 10kHz
    }
}
