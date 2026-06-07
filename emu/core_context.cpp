#include "core_context.hpp"
#include <memory>
#include <unordered_map>

// Constructor: Initialize all core components
core_context_t::core_context_t(int id, int fdMsg, std::mutex& msgMutex, uint8_t* dram, uint8_t* uram)
    : core_id(id), uram(uram), running(true) {

    // Create top-level execution unit
    top = std::make_unique<RVTNG::exec_unit_t>();

    // Register Commit callback (using this core's core_id)
    top->register_commit_callback([fdMsg, &msgMutex, core_id = id](const rvt_commit_t* pkg) {
        if(pkg->rd == 0) return;
        std::lock_guard<std::mutex> lock(msgMutex);
        backend_instruction_commit(fdMsg, core_id, pkg);
    });

    // Create Scoreboard queue
    scoreboard = std::make_unique<RVTNG::MPMCQueue<uint64_t, 32>>();
    top->register_scoreboard_callback([&sb = *scoreboard](const uint64_t entry) {
        sb.enqueue(entry);
    });

    auto vec_barriers = std::make_shared<std::unordered_map<uint64_t, uint8_t>>();
    auto vec_barrier_mutex = std::make_shared<std::mutex>();
    top->register_vec_barrier_callback(
        [top_ptr = top.get(), vec_barriers, vec_barrier_mutex](const rvt_pack_t* inst) {
            std::lock_guard<std::mutex> lock(*vec_barrier_mutex);
            uint8_t& arrived = (*vec_barriers)[inst->timestamp];
            arrived++;
            if(arrived < 2) {
                return;
            }
            vec_barriers->erase(inst->timestamp);
            rvt_commit_t cmt_pack;
            cmt_pack.timestamp = inst->timestamp;
            cmt_pack.rd = (inst->inst >> 7) & 0x1f;
            cmt_pack.data = inst->rd;
            top_ptr->commit(&cmt_pack);
        });

    // Create execution units (each core has its own URAM, but shares DRAM)
    lsu = std::make_unique<RVTNG::lsu_t>(top.get(), uram, dram);
    mat = std::make_unique<RVTNG::mat_t>(top.get(), uram);
    vec = std::make_unique<RVTNG::vec_t>(top.get(), uram, 16);
    legacy = std::make_unique<RVTNG::rvt64_t>(top.get(), uram);

    // Register Scoreboard Issue callback
    top->register_issue_callback([top_ptr = top.get(), &sb = *scoreboard](const rvt_pack_t* inst) {
        bool is_sb = (inst->inst & 0xfe000000) == 0x0e000000;
        if(!is_sb) return 0ul;
        rvt_commit_t cmt_pack;
        cmt_pack.timestamp = inst->timestamp;
        cmt_pack.rd = (inst->inst >> 7) & 0x1f;
        auto p = sb.try_dequeue();
        if(p != std::nullopt) debugf("Scoreboard: empty:%x data:%lx!\n", p == std::nullopt, p.value_or(0ul));
        cmt_pack.data = p.value_or(0ul);
        top_ptr->commit(&cmt_pack);
        return 1ul;
    });

    // Start execution threads
    lsu_exec_thread = std::thread(&RVTNG::lsu_t::execute, lsu.get());
    mat_exec_thread = std::thread(&RVTNG::mat_t::execute, mat.get());
    vec_exec_thread = std::thread(&RVTNG::vec_t::execute, vec.get());
    legacy_exec_thread = std::thread(&RVTNG::rvt64_t::execute, legacy.get());
}

// Destructor: Clean up resources
core_context_t::~core_context_t() {
    // Stop all threads
    running = false;
    
    // Wait for all threads to finish
    if(lsu_exec_thread.joinable()) {
        lsu_exec_thread.join();
    }
    if(mat_exec_thread.joinable()) {
        mat_exec_thread.join();
    }
    if(vec_exec_thread.joinable()) {
        vec_exec_thread.join();
    }
    if(legacy_exec_thread.joinable()) {
        legacy_exec_thread.join();
    }
}

// Fetch one instruction for this core
void core_context_t::fetch_one(int fdInst, std::mutex& fdInstMutex) {
    rvt_pack_t inst;
    int count;
    {
        std::lock_guard<std::mutex> lock(fdInstMutex);
        count = backend_instruction_fetch(fdInst, core_id, &inst);
    }
    if(count) {
        ulong v = top->issue(&inst);
        bool is_vec_sb_bar = (inst.inst & 0xfe00707f) == 0x0a00500b;
        if(v != 1 && !(is_vec_sb_bar && v == 2)) {
            debugf("Core %d: Warning: Inst %lx Issued %ld times.\n", core_id, inst.inst, v);
        }
    }
}

