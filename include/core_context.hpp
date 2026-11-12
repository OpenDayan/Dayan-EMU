#ifndef CORE_CONTEXT_HPP
#define CORE_CONTEXT_HPP

#include "utils.hpp"
#include "dayan-intf.h"
#include "exec_unit.hpp"
#include "mmu.hpp"
#include "lsu.hpp"
#include "mat.hpp"
#include "vec.hpp"
#include "rvt64.hpp"
#include "thread_safe_queue.h"
#include "tensor_context.hpp"
#include "interrupt.hpp"
#include "resource_scheduler.hpp"

// Core context class
class core_context_t {
public:
    int core_id;
    uint8_t* uram;  // Per-core URAM pointer
    std::unique_ptr<DAYAN::exec_unit_t> top;
    std::unique_ptr<DAYAN::mmu_t> mmu;
    std::unique_ptr<DAYAN::lsu_t> lsu;
    std::unique_ptr<DAYAN::mat_t> mat;
    std::unique_ptr<DAYAN::vec_t> vec;
    std::unique_ptr<DAYAN::rvt64_t> legacy;
    std::unique_ptr<DAYAN::tensor_context_t> tensor_context;
    std::unique_ptr<DAYAN::interrupt_controller_t> interrupts;
    std::unique_ptr<DAYAN::MPMCQueue<DAYAN::scoreboard_entry_t, 32>> scoreboard;
    std::unique_ptr<DAYAN::resource_scheduler_t> resource_scheduler;
    std::thread lsu_exec_thread;
    std::thread mat_exec_thread;
    std::thread vec_exec_thread;
    std::thread legacy_exec_thread;
    std::atomic<bool> running;

    // Constructor: Initialize all core components
    core_context_t(int id, int fdMsg, std::mutex& msgMutex, uint8_t* dram,
                   uint8_t* uram, uint8_t *const *cluster_urams,
                   size_t cluster_core_count,
                   DAYAN::resource_cluster_t& resource_cluster,
                   size_t dram_size = DAYAN::MMU_DEFAULT_DRAM_SIZE,
                   uint64_t dram_base = DAYAN::MMU_DRAM_BASE);

    // Destructor: Clean up resources
    ~core_context_t();

    // Delete copy constructor and assignment operator
    core_context_t(const core_context_t&) = delete;
    core_context_t& operator=(const core_context_t&) = delete;

    // Fetch one instruction for this core (called from main loop)
    void fetch_one(int fdInstReq, int fdInstRsp, std::mutex& fdInstMutex);

private:
    ulong issue_global(const dayan_pack_t* inst);
};

#endif // CORE_CONTEXT_HPP
