#ifndef DAYAN_MMU_HPP
#define DAYAN_MMU_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <vector>

// The behavioral ClusterMMU implementation lives in the shared C core
// hw/riscv/dayan_mmu_core.c (built from deps/npu-qemu and also linked by
// the QEMU device).  This class is a C++ shell over it: per-instance
// thread safety, the std::function PTE-reader bridge and the Dayan-emu-side
// enum/result surface.  The access, fault, page-size and PTE-response
// enums mirror the core's numerically, so conversion is a static_cast.
#include "hw/riscv/dayan_mmu_core.h"

namespace DAYAN {

// ClusterMMU uses 56-bit physical addresses.  Keep all public results masked
// to this width so host pointer arithmetic never accidentally becomes part of
// the architectural address.
constexpr uint64_t MMU_PA_MASK = DAYAN_MMU_PA_MASK;
constexpr uint64_t MMU_VA_MASK = DAYAN_MMU_VA_MASK;
constexpr uint64_t MMU_PAGE_OFFSET_MASK = DAYAN_MMU_PAGE_MASK;
constexpr uint64_t MMU_DRAM_BASE = DAYAN_MMU_DRAM_BASE;
constexpr size_t MMU_DEFAULT_DRAM_SIZE = DAYAN_MMU_DEFAULT_DRAM_SIZE;

enum class mmu_access_type_t : uint8_t {
    read,
    write,
    execute,
};

enum class mmu_page_size_t : uint8_t {
    page_4k,
    page_2m,
    page_1g,
};

enum class mmu_fault_code_t : uint8_t {
    none = 0,
    pte_invalid = 1,
    read_perm = 2,
    write_perm = 3,
    exec_perm = 4,
    illegal_pte = 5,
    ptw_bus_error = 9,
    ptw_timeout = 10,
};

struct mmu_fixmap_config_t {
    bool n_enable = false;
    uint64_t base = 0;
    uint64_t size_pages = 0;
};

struct mmu_fault_redirect_config_t {
    uint64_t page_fault = 0;
    uint64_t read_perm = 0;
    uint64_t write_perm = 0;
    uint64_t exec_perm = 0;
    uint64_t ptw_error = 0;
    uint64_t default_base = 0;
    bool offset_enable = false;
};

struct mmu_config_snapshot_t {
    uint8_t asid = 0;
    uint64_t ptw_base = 0;
    mmu_fixmap_config_t fixmap;
    mmu_fault_redirect_config_t redirect;
};

struct mmu_request_t {
    uint64_t va = 0;
    mmu_access_type_t access = mmu_access_type_t::read;
    uint8_t port = 0;
    mmu_config_snapshot_t config;
};

struct mmu_translation_result_t {
    uint64_t pa = 0;
    mmu_fault_code_t fault = mmu_fault_code_t::none;
    mmu_page_size_t page_size = mmu_page_size_t::page_4k;
    bool redirected = false;
    bool global_mapping = false;
    std::vector<uint64_t> walk_addresses;
    uint64_t pte_snapshot = 0;

    bool ok() const { return fault == mmu_fault_code_t::none; }
};

struct mmu_pte_read_result_t {
    enum class response_t : uint8_t {
        okay,
        bus_error,
        timeout,
    };

    response_t response = response_t::okay;
    uint64_t data = 0;
};

using mmu_pte_reader_t = std::function<mmu_pte_read_result_t(uint64_t)>;

class mmu_t {
public:
    // A missing reader behaves like an all-zero page-table image, which
    // naturally produces PTE_INVALID after FixMap is disabled.
    explicit mmu_t(mmu_pte_reader_t reader = {});

    // Convenience constructor for the Dayan-emu's shared DRAM image.  The
    // physical address accepted by the callback is [dram_base, dram_base+size).
    mmu_t(uint8_t* dram, size_t dram_size,
          uint64_t dram_base = MMU_DRAM_BASE);

    mmu_t(const mmu_t&) = delete;
    mmu_t& operator=(const mmu_t&) = delete;

    mmu_config_snapshot_t snapshot() const;
    mmu_request_t capture_request(uint64_t va, mmu_access_type_t access,
                                  uint8_t port = 0) const;

    // translate(va, ...) captures the current configuration at call entry,
    // which is the software equivalent of the hardware slave AR/AW handshake.
    mmu_translation_result_t translate(uint64_t va, mmu_access_type_t access,
                                       uint8_t port = 0);
    // Two-phase form: translate against a config image pinned earlier by
    // capture_request(), while the TLB/PDC state stays live.
    mmu_translation_result_t translate(const mmu_request_t& request);

    void set_pte_reader(mmu_pte_reader_t reader);

    // Direct configuration API for Dayan-emu clients.
    void set_ptw_base(uint64_t value);
    void set_global_asid(uint8_t value);
    void set_fixmap(bool n_enable, uint64_t base, uint64_t size_pages);
    void set_redirects(const mmu_fault_redirect_config_t& value);

    // APB4-compatible register image.  Writes are observed at access phase;
    // return false only for an unsupported address.  Alignment-constrained
    // writes are accepted but silently ignored, matching the RTL.
    bool write_register(uint16_t address, uint32_t data);
    uint32_t read_register(uint16_t address) const;

    // The hardware flush is a drain/clear/reopen sequence.  A synchronous
    // Dayan-emu has no cycle-level in-flight ownership, so this API performs the
    // clear atomically and sets the same sticky done/error status bits.
    void flush(bool asid_selective, uint8_t asid);
    bool flush_done() const;
    bool flush_busy() const;
    uint8_t flush_error_code() const;
    void acknowledge_flush();

    static mmu_pte_reader_t make_dram_reader(
        uint8_t* dram, size_t dram_size, uint64_t dram_base = MMU_DRAM_BASE);

private:
    struct reader_bridge_t {
        mmu_pte_reader_t fn;
    };

    static DayanMmuPteRead pte_trampoline(void* opaque, uint64_t pa);

    // cfg == NULL uses the config image live at call entry; otherwise the
    // caller-pinned image from request.config.  Both run under the lock.
    mmu_translation_result_t run_translation(const DayanMmuCore* cfg,
                                             uint64_t va,
                                             mmu_access_type_t access,
                                             uint8_t port);

    mutable std::shared_mutex mutex_;
    DayanMmuCore core_;
    // The bridge must outlive every core_ PTW callback, so it lives in the
    // object and never moves; the callable itself rides inside the bridge.
    reader_bridge_t bridge_;
};

} // namespace DAYAN

#endif // DAYAN_MMU_HPP
