#include "core_context.hpp"

// Constructor: Initialize all core components
core_context_t::core_context_t(int id, int fdMsg, std::mutex& msgMutex, uint8_t* dram)
    : core_id(id), uram(nullptr), running(true) {
    
    // Create per-core URAM
    char uram_shm_name[64];
    snprintf(uram_shm_name, sizeof(uram_shm_name), "/uram_shm_%d", id);
    uram = (uint8_t *)backend_get_shm_ptr(uram_shm_name, 1024 * 1024);
    if(uram == nullptr) {
        fprintf(stderr, "Error: Failed to create URAM for core %d\n", id);
        throw std::runtime_error("Failed to create URAM");
    }
    printf("Core %d: Created URAM '%s' at %p\n", id, uram_shm_name, uram);

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
        if(v != 1) {
            debugf("Core %d: Warning: Inst %lx Issued %ld times.\n", core_id, inst.inst, v);
        }
    }
}

