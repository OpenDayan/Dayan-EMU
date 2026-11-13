/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-emu is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include "mmu.hpp"

#include <cstring>
#include <memory>
#include <utility>

namespace DAYAN {

namespace {

// redir[] index order shared by the core's config image.
enum redirect_index_t {
    kRedirPageFault = 0,
    kRedirReadPerm,
    kRedirWritePerm,
    kRedirExecPerm,
    kRedirPtwError,
    kRedirDefault,
};

} // namespace

DayanMmuPteRead mmu_t::pte_trampoline(void* opaque, uint64_t pa)
{
    auto* bridge = static_cast<reader_bridge_t*>(opaque);
    if(!bridge || !bridge->fn) {
        // Matches an all-zero page-table image: PTE reads succeed with 0.
        DayanMmuPteRead read = {DAYAN_MMU_PTE_OKAY, 0};
        return read;
    }
    const mmu_pte_read_result_t read = bridge->fn(pa);
    DayanMmuPteRead result = {
        static_cast<DayanMmuPteResponse>(read.response), read.data};
    return result;
}

mmu_t::mmu_t(mmu_pte_reader_t reader)
{
    dayan_mmu_core_init(&core_);
    bridge_.fn = std::move(reader);
    dayan_mmu_core_set_reader(&core_, &mmu_t::pte_trampoline, &bridge_);
}

mmu_t::mmu_t(uint8_t* dram, size_t dram_size, uint64_t dram_base)
{
    dayan_mmu_core_init(&core_);
    dayan_mmu_core_bind_dram(&core_, dram, dram_base, dram_size);
}

mmu_config_snapshot_t mmu_t::snapshot() const
{
    std::shared_lock lock(mutex_);
    mmu_config_snapshot_t config;
    config.asid = core_.global_asid;
    config.ptw_base = core_.ptw_base;
    config.fixmap.n_enable = core_.n_fixmap_en;
    config.fixmap.base = core_.fixmap_base;
    config.fixmap.size_pages = core_.fixmap_size_pages;
    config.redirect.page_fault = core_.redir[kRedirPageFault];
    config.redirect.read_perm = core_.redir[kRedirReadPerm];
    config.redirect.write_perm = core_.redir[kRedirWritePerm];
    config.redirect.exec_perm = core_.redir[kRedirExecPerm];
    config.redirect.ptw_error = core_.redir[kRedirPtwError];
    config.redirect.default_base = core_.redir[kRedirDefault];
    config.redirect.offset_enable = core_.redir_offset_enable;
    return config;
}

mmu_request_t mmu_t::capture_request(uint64_t va, mmu_access_type_t access,
                                     uint8_t port) const
{
    mmu_request_t request;
    request.va = va & MMU_PA_MASK;
    request.access = access;
    request.port = port < DAYAN_MMU_NUM_PORTS ? port : 0;
    request.config = snapshot();
    return request;
}

mmu_translation_result_t mmu_t::run_translation(const DayanMmuCore* cfg,
                                                uint64_t va,
                                                mmu_access_type_t access,
                                                uint8_t port)
{
    DayanMmuFault fault = DAYAN_MMU_FAULT_NONE;
    DayanMmuPageSize page_size = DAYAN_MMU_PAGE_4K;
    DayanMmuTranslateDetail detail;
    detail.walk_depth = 0;
    detail.pte_snapshot = 0;
    detail.global_mapping = false;
    detail.redirected = false;

    const uint64_t pa = dayan_mmu_core_translate_full(
        &core_, cfg, va, static_cast<DayanMmuAccess>(access), port, &fault,
        &page_size, &detail);

    mmu_translation_result_t result;
    result.pa = pa;
    result.fault = static_cast<mmu_fault_code_t>(fault);
    result.page_size = static_cast<mmu_page_size_t>(page_size);
    result.redirected = detail.redirected;
    result.global_mapping = detail.global_mapping;
    result.pte_snapshot = detail.pte_snapshot;
    result.walk_addresses.assign(detail.walk_addresses,
                                 detail.walk_addresses + detail.walk_depth);
    return result;
}

mmu_translation_result_t mmu_t::translate(uint64_t va,
                                          mmu_access_type_t access,
                                          uint8_t port)
{
    std::unique_lock lock(mutex_);
    return run_translation(nullptr, va, access, port);
}

mmu_translation_result_t mmu_t::translate(const mmu_request_t& request)
{
    // Config-only image: the core reads TLB/PDC/transport from core_ and
    // masks the config fields into its own copy.
    DayanMmuCore cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.global_asid = request.config.asid;
    cfg.ptw_base = request.config.ptw_base;
    cfg.n_fixmap_en = request.config.fixmap.n_enable;
    cfg.fixmap_base = request.config.fixmap.base;
    cfg.fixmap_size_pages = request.config.fixmap.size_pages;
    cfg.redir[kRedirPageFault] = request.config.redirect.page_fault;
    cfg.redir[kRedirReadPerm] = request.config.redirect.read_perm;
    cfg.redir[kRedirWritePerm] = request.config.redirect.write_perm;
    cfg.redir[kRedirExecPerm] = request.config.redirect.exec_perm;
    cfg.redir[kRedirPtwError] = request.config.redirect.ptw_error;
    cfg.redir[kRedirDefault] = request.config.redirect.default_base;
    cfg.redir_offset_enable = request.config.redirect.offset_enable;

    std::unique_lock lock(mutex_);
    return run_translation(&cfg, request.va, request.access, request.port);
}

void mmu_t::set_pte_reader(mmu_pte_reader_t reader)
{
    std::unique_lock lock(mutex_);
    bridge_.fn = std::move(reader);
    // Re-registering also invalidates every cached translation, as the
    // PTW transport may now observe a different page-table image.
    dayan_mmu_core_set_reader(&core_, &mmu_t::pte_trampoline, &bridge_);
}

void mmu_t::set_ptw_base(uint64_t value)
{
    std::unique_lock lock(mutex_);
    core_.ptw_base = value & MMU_PA_MASK;
}

void mmu_t::set_global_asid(uint8_t value)
{
    std::unique_lock lock(mutex_);
    core_.global_asid = value;
}

void mmu_t::set_fixmap(bool n_enable, uint64_t base, uint64_t size_pages)
{
    std::unique_lock lock(mutex_);
    core_.n_fixmap_en = n_enable;
    core_.fixmap_base = base & MMU_PA_MASK;
    core_.fixmap_size_pages = size_pages & MMU_PA_MASK;
}

void mmu_t::set_redirects(const mmu_fault_redirect_config_t& value)
{
    std::unique_lock lock(mutex_);
    core_.redir[kRedirPageFault] = value.page_fault & MMU_PA_MASK;
    core_.redir[kRedirReadPerm] = value.read_perm & MMU_PA_MASK;
    core_.redir[kRedirWritePerm] = value.write_perm & MMU_PA_MASK;
    core_.redir[kRedirExecPerm] = value.exec_perm & MMU_PA_MASK;
    core_.redir[kRedirPtwError] = value.ptw_error & MMU_PA_MASK;
    core_.redir[kRedirDefault] = value.default_base & MMU_PA_MASK;
    core_.redir_offset_enable = value.offset_enable;
}

bool mmu_t::write_register(uint16_t address, uint32_t data)
{
    std::unique_lock lock(mutex_);
    return dayan_mmu_core_write_reg(&core_, address, data);
}

uint32_t mmu_t::read_register(uint16_t address) const
{
    std::shared_lock lock(mutex_);
    return dayan_mmu_core_read_reg(&core_, address);
}

void mmu_t::flush(bool asid_selective, uint8_t asid)
{
    std::unique_lock lock(mutex_);
    dayan_mmu_core_flush(&core_, asid_selective, asid);
}

bool mmu_t::flush_done() const
{
    std::shared_lock lock(mutex_);
    return core_.flush_done;
}

bool mmu_t::flush_busy() const
{
    std::shared_lock lock(mutex_);
    return core_.flush_busy;
}

uint8_t mmu_t::flush_error_code() const
{
    std::shared_lock lock(mutex_);
    return core_.flush_error_code;
}

void mmu_t::acknowledge_flush()
{
    std::unique_lock lock(mutex_);
    dayan_mmu_core_ack_flush(&core_);
}

mmu_pte_reader_t mmu_t::make_dram_reader(uint8_t* dram, size_t dram_size,
                                         uint64_t dram_base)
{
    // The bound-image description must outlive the callable, and a
    // std::function must be copyable, hence the shared_ptr.
    auto image = std::make_shared<DayanMmuCore>();
    dayan_mmu_core_init(image.get());
    dayan_mmu_core_bind_dram(image.get(), dram, dram_base, dram_size);
    return [image](uint64_t physical_address) {
        const DayanMmuPteRead read =
            dayan_mmu_core_dram_read(image.get(), physical_address);
        return mmu_pte_read_result_t{
            static_cast<mmu_pte_read_result_t::response_t>(read.response),
            read.data};
    };
}

} // namespace DAYAN
