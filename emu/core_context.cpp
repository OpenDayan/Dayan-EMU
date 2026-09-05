#include "core_context.hpp"
#include "rvt_decoder.hpp"

// Constructor: Initialize all core components
core_context_t::core_context_t(int id, int fdMsg, std::mutex& msgMutex,
                               uint8_t* dram, uint8_t* uram,
                               uint8_t *const *cluster_urams,
                               size_t cluster_core_count,
                               RVTNG::resource_cluster_t& resource_cluster,
                               size_t dram_size,
                               uint64_t dram_base)
    : core_id(id), uram(uram), running(true) {

    // Create top-level execution unit
    top = std::make_unique<RVTNG::exec_unit_t>();
    tensor_context = std::make_unique<RVTNG::tensor_context_t>();
    interrupts = std::make_unique<RVTNG::interrupt_controller_t>(
        [fdMsg, &msgMutex, core_id = id](bool level) {
            std::lock_guard<std::mutex> lock(msgMutex);
            backend_interrupt_update(fdMsg, core_id, level);
        });

    // Register Commit callback (using this core's core_id)
    top->register_commit_callback([fdMsg, &msgMutex, core_id = id](const rvt_commit_t* pkg) {
        std::lock_guard<std::mutex> lock(msgMutex);
        backend_instruction_commit(fdMsg, core_id, pkg);
    });

    // Create Scoreboard queue
    scoreboard = std::make_unique<
        RVTNG::MPMCQueue<RVTNG::scoreboard_entry_t, 32>>();
    top->register_scoreboard_callback([&sb = *scoreboard](const uint64_t entry) {
        sb.enqueue(RVTNG::scoreboard_entry_t{entry, 0});
    });
    resource_scheduler = std::make_unique<RVTNG::resource_scheduler_t>(
        resource_cluster, *scoreboard, interrupts.get());
    top->register_resource_produce_callback(
        resource_scheduler.get(),
        &RVTNG::resource_scheduler_t::complete_produce);

    // Create one MMU/cache context per core.  The PTW reader observes the
    // shared DRAM image; reset defaults leave the full-range identity FixMap
    // enabled, preserving the pre-MMU CModel behaviour until software
    // programs the register image exposed by core_context_t::mmu.
    mmu = std::make_unique<RVTNG::mmu_t>(dram, dram_size, dram_base);

    // Create execution units (each core has its own URAM, but shares DRAM)
    lsu = std::make_unique<RVTNG::lsu_t>(top.get(), uram, dram,
                                        cluster_urams,
                                        cluster_core_count,
                                        tensor_context.get(), interrupts.get(),
                                        mmu.get(), dram_size, dram_base,
                                        static_cast<uint8_t>(4 + (id & 0x3)));
    mat = std::make_unique<RVTNG::mat_t>(top.get(), uram,
                                        tensor_context.get(), interrupts.get());
    vec = std::make_unique<RVTNG::vec_t>(top.get(), uram, 16,
                                        tensor_context.get(), interrupts.get());
    legacy = std::make_unique<RVTNG::rvt64_t>(top.get(), uram,
                                              tensor_context.get(), interrupts.get());

    // Register Scoreboard Issue callback
    top->register_issue_callback(
        [top_ptr = top.get(), &sb = *scoreboard,
         scheduler = resource_scheduler.get()](const rvt_pack_t* inst) {
        const uint32_t unit = (inst->inst >> 23) & 0x0f;
        const uint32_t address = (inst->inst >> 13) & 0x3ff;
        const bool is_sb = rvt::is_base(static_cast<uint32_t>(inst->inst)) &&
            unit == RVT_UNIT_BASE && address == RVT_UNIT_BASE;
        if(!is_sb) return 0ul;
        if(inst->inst & 0b1000000000000) {
            rvt_commit_t cmt_pack;
            cmt_pack.timestamp = inst->timestamp;
            cmt_pack.rd = (inst->inst >> 7) & 0x1f;
            auto p = sb.try_dequeue();
            if(p != std::nullopt) {
                debugf("Scoreboard: empty:0 data:%lx!\n", p->token);
            }
            cmt_pack.data = p ? p->token : 0;
            top_ptr->commit(&cmt_pack);
            if(p) {
                scheduler->output_popped(*p);
            }
        } else {
            // sb.enqueue(entry);
            // DO NOTHING
            ;;
        }
        return 1ul;
    });

    // Start execution threads
    lsu_exec_thread = std::thread(&RVTNG::lsu_t::execute, lsu.get());
    mat_exec_thread = std::thread(&RVTNG::mat_t::execute, mat.get());
    vec_exec_thread = std::thread(&RVTNG::vec_t::execute, vec.get());
    legacy_exec_thread = std::thread(&RVTNG::rvt64_t::execute, legacy.get());
}

ulong core_context_t::issue_global(const rvt_pack_t* inst) {
    if(RVTNG::is_context_write(inst->inst)) {
        const uint16_t address = (inst->inst >> 13) & 0x3ff;
        if(!tensor_context->write(address, inst->rd)) {
            interrupts->raise(RVTNG::interrupt_source_t::resource_scheduler_error, address);
            return 1;
        }
        const auto context = tensor_context->snapshot();
        lsu->enqueue_context(*inst, context);
        mat->enqueue_context(*inst, context);
        vec->enqueue_context(*inst, context);
        legacy->enqueue_context(*inst, context);
        return 1;
    }

    if(RVTNG::is_context_read(inst->inst)) {
        const uint16_t address = (inst->inst >> 13) & 0x3ff;
        rvt_commit_t response{};
        response.rd = (inst->inst >> 7) & 0x1f;
        response.data = tensor_context->read(address);
        response.timestamp = inst->timestamp;
        top->commit(&response);
        return 1;
    }

    if(RVTNG::is_interrupt_tcsr(inst->inst)) {
        const uint16_t address = (inst->inst >> 13) & 0x3ff;
        const bool read = ((inst->inst >> 12) & 1) != 0;
        if(read) {
            rvt_commit_t response{};
            response.rd = (inst->inst >> 7) & 0x1f;
            response.data = interrupts->read(address);
            response.timestamp = inst->timestamp;
            top->commit(&response);
        } else {
            interrupts->write(address, inst->rd);
        }
        return 1;
    }

    if(rvt::is_base(static_cast<uint32_t>(inst->inst)) &&
       ((inst->inst >> 23) & 0x0f) == RVT_UNIT_BASE) {
        const uint16_t address = (inst->inst >> 13) & 0x3ff;
        if(resource_scheduler->handles(address)) {
            const bool read = ((inst->inst >> 12) & 1) != 0;
            bool accepted = true;
            if(read) {
                rvt_commit_t response{};
                response.rd = (inst->inst >> 7) & 0x1f;
                response.data = resource_scheduler->read(address);
                response.timestamp = inst->timestamp;
                top->commit(&response);
            } else {
                accepted = resource_scheduler->write(address, inst->rd,
                                                     tensor_context->snapshot());
            }
            const bool deferred_produce = accepted && !read &&
                address >= RVTNG::RESOURCE_PRODUCE_BASE &&
                address < RVTNG::RESOURCE_PRODUCE_BASE + 16;
            return deferred_produce ? 0 : 1;
        }
    }

    return 0;
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
void core_context_t::fetch_one(int fdInstReq, int fdInstRsp, std::mutex& fdInstMutex) {
    rvt_pack_t inst;
    int count;
    {
        std::lock_guard<std::mutex> lock(fdInstMutex);
        count = backend_instruction_fetch(fdInstReq, fdInstRsp, core_id, &inst);
    }
    if(count) {
        // 全局 CSR 先处理一次；只有合法 Resource Produce 写才继续进入目标单元队列。
        // 非法写不得产生完成事件，否则会错配该单元更早登记的 Produce 描述符。
        ulong v = issue_global(&inst);
        if(v == 0) {
            v = top->issue(&inst);
        }
        if(v == 0) {
            interrupts->raise(RVTNG::interrupt_source_t::instruction_decode_error,
                              static_cast<uint32_t>(inst.inst));
        }
        if(v != 1) {
            debugf("Core %d: Warning: Inst %lx Issued %ld times.\n", core_id, inst.inst, v);
        }
    }
}
