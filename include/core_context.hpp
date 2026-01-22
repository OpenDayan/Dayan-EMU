#ifndef CORE_CONTEXT_HPP
#define CORE_CONTEXT_HPP

#include "utils.hpp"
#include "rvtng-intf.h"
#include "exec_unit.hpp"
#include "lsu.hpp"
#include "mat.hpp"
#include "vec.hpp"
#include "rvt64.hpp"
#include "thread_safe_queue.h"

// Core context class
class core_context_t {
public:
    int core_id;
    uint8_t* uram;  // Per-core URAM pointer
    std::unique_ptr<RVTNG::exec_unit_t> top;
    std::unique_ptr<RVTNG::lsu_t> lsu;
    std::unique_ptr<RVTNG::mat_t> mat;
    std::unique_ptr<RVTNG::vec_t> vec;
    std::unique_ptr<RVTNG::rvt64_t> legacy;
    std::unique_ptr<RVTNG::MPMCQueue<uint64_t, 32>> scoreboard;
    std::thread lsu_exec_thread;
    std::thread mat_exec_thread;
    std::thread vec_exec_thread;
    std::thread legacy_exec_thread;
    std::atomic<bool> running;

    // Constructor: Initialize all core components
    core_context_t(int id, int fdMsg, std::mutex& msgMutex, uint8_t* dram, uint8_t* uram);

    // Destructor: Clean up resources
    ~core_context_t();

    // Delete copy constructor and assignment operator
    core_context_t(const core_context_t&) = delete;
    core_context_t& operator=(const core_context_t&) = delete;

    // Fetch one instruction for this core (called from main loop)
    void fetch_one(int fdInst, std::mutex& fdInstMutex);
};

#endif // CORE_CONTEXT_HPP

