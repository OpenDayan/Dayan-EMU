#ifndef RVTNG_MMU_HPP
#define RVTNG_MMU_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <shared_mutex>
#include <vector>

namespace RVTNG {

// ClusterMMU uses 56-bit physical addresses.  Keep all public results masked
// to this width so host pointer arithmetic never accidentally becomes part of
// the architectural address.
constexpr uint64_t MMU_PA_MASK = 0x00ffffffffffffffull;
constexpr uint64_t MMU_VA_MASK = (uint64_t{1} << 39) - 1;
constexpr uint64_t MMU_PAGE_OFFSET_MASK = 0xfffull;
constexpr uint64_t MMU_DRAM_BASE = 0x80000000ull;
constexpr size_t MMU_DEFAULT_DRAM_SIZE =
    size_t{4} * 1024 * 1024 * 1024;

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

    // Convenience constructor for the CModel's shared DRAM image.  The
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
    mmu_translation_result_t translate(const mmu_request_t& request);

    void set_pte_reader(mmu_pte_reader_t reader);

    // Direct configuration API for CModel clients.
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
    // CModel has no cycle-level in-flight ownership, so this API performs the
    // clear atomically and sets the same sticky done/error status bits.
    void flush(bool asid_selective, uint8_t asid);
    bool flush_done() const;
    bool flush_busy() const;
    uint8_t flush_error_code() const;
    void acknowledge_flush();

    static mmu_pte_reader_t make_dram_reader(
        uint8_t* dram, size_t dram_size, uint64_t dram_base = MMU_DRAM_BASE);

private:
    struct tlb_entry_t {
        bool valid = false;
        bool negative = false;
        mmu_fault_code_t fault = mmu_fault_code_t::none;
        uint64_t vpn = 0;
        uint64_t ppn = 0;
        uint8_t asid = 0;
        mmu_page_size_t page_size = mmu_page_size_t::page_4k;
        bool global = false;
        bool r = false;
        bool w = false;
        bool x = false;
        bool d = false;
        bool a = false;
        bool u = false;
    };

    struct pdc_entry_t {
        bool valid = false;
        uint16_t vpn2 = 0;
        uint16_t vpn1 = 0;
        uint8_t asid = 0;
        bool global = false;
        uint64_t ppn = 0;
    };

    struct pdc_lookup_t {
        bool hit = false;
        bool global = false;
        uint64_t ppn = 0;
    };

    struct l2_bank_t {
        unsigned ways = 1;
        unsigned sets = 128;
        std::vector<tlb_entry_t> entries;
    };

    static uint64_t mask_pa(uint64_t value);
    static uint64_t vpn2(uint64_t vpn);
    static uint64_t vpn1(uint64_t vpn);
    static uint64_t vpn0(uint64_t vpn);
    static uint64_t vpn_index(uint64_t vpn, unsigned level);
    static uint64_t page_size_bytes(mmu_page_size_t size);
    static bool page_match(const tlb_entry_t& entry, uint64_t vpn);
    static bool permission_ok(const tlb_entry_t& entry,
                              mmu_access_type_t access,
                              mmu_fault_code_t* fault);

    bool fixmap_hit(uint64_t va, const mmu_config_snapshot_t& config) const;
    uint64_t fixmap_pa(uint64_t va, const mmu_config_snapshot_t& config) const;
    uint64_t redirect_pa(uint64_t va, mmu_fault_code_t fault,
                         const mmu_config_snapshot_t& config) const;

    tlb_entry_t* l1_lookup(unsigned port, uint64_t vpn, uint8_t asid);
    const tlb_entry_t* l1_lookup(unsigned port, uint64_t vpn,
                                  uint8_t asid) const;
    void l1_insert(unsigned port, const tlb_entry_t& entry);

    tlb_entry_t* l2_lookup(uint64_t vpn, uint8_t asid);
    const tlb_entry_t* l2_lookup(uint64_t vpn, uint8_t asid) const;
    void l2_insert(const tlb_entry_t& entry);

    pdc_lookup_t pdc_l0_lookup(uint16_t v2, uint8_t asid) const;
    pdc_lookup_t pdc_l1_lookup(uint16_t v2, uint16_t v1,
                               uint8_t asid) const;
    void pdc_l0_insert(uint16_t v2, uint8_t asid, bool global, uint64_t ppn);
    void pdc_l1_insert(uint16_t v2, uint16_t v1, uint8_t asid,
                       bool global, uint64_t ppn);

    mmu_translation_result_t walk(const mmu_request_t& request,
                                  uint64_t vpn);
    mmu_translation_result_t make_fault(const mmu_request_t& request,
                                         mmu_fault_code_t fault,
                                         const std::vector<uint64_t>& addresses,
                                         uint64_t pte_snapshot);
    mmu_translation_result_t make_success(const mmu_request_t& request,
                                          const tlb_entry_t& entry);
    void invalidate_all();
    void invalidate_asid(uint8_t asid);
    void advance_lfsr();

    mutable std::shared_mutex mutex_;
    mmu_pte_reader_t reader_;
    mmu_config_snapshot_t config_;
    std::array<std::vector<tlb_entry_t>, 8> l1_;
    l2_bank_t l2_4k_;
    l2_bank_t l2_2m_;
    l2_bank_t l2_1g_;
    std::vector<pdc_entry_t> pdc_l0_;
    std::vector<pdc_entry_t> pdc_l1_;
    uint16_t l1_lfsr_ = 1;
    uint8_t replacement_lfsr_ = 1;
    uint64_t error_slv_mask_ = 0;
    bool flush_done_ = false;
    bool flush_busy_ = false;
    uint8_t flush_error_code_ = 0;
};

} // namespace RVTNG

#endif // RVTNG_MMU_HPP
