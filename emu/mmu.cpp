#include "mmu.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

namespace RVTNG {

namespace {

constexpr uint64_t kPpnMask = (uint64_t{1} << 44) - 1;
constexpr uint64_t kVpnMask = (uint64_t{1} << 27) - 1;
constexpr unsigned kPageOffsetBits = 12;

constexpr uint16_t kPtwBaseLo = 0x000;
constexpr uint16_t kPtwBaseHi = 0x004;
constexpr uint16_t kFixMapBaseLo = 0x008;
constexpr uint16_t kFixMapBaseHi = 0x00c;
constexpr uint16_t kNFixMapEn = 0x010;
constexpr uint16_t kTlbFlush = 0x014;
constexpr uint16_t kFlushAck = 0x018;
constexpr uint16_t kFlushDone = 0x01c;
constexpr uint16_t kFlushBusy = 0x020;
constexpr uint16_t kFlushErrCode = 0x024;
constexpr uint16_t kGlobalAsid = 0x028;
constexpr uint16_t kErrorSlvMaskLo = 0x048;
constexpr uint16_t kErrorSlvMaskHi = 0x04c;
constexpr uint16_t kErrorSlvOffMode = 0x050;
constexpr uint16_t kErrorSlvCfgErr = 0x070;
constexpr uint16_t kErrorSlvCfgClr = 0x074;
constexpr uint16_t kRedirPageFaultLo = 0x080;
constexpr uint16_t kRedirPageFaultHi = 0x084;
constexpr uint16_t kRedirReadPermLo = 0x088;
constexpr uint16_t kRedirReadPermHi = 0x08c;
constexpr uint16_t kRedirWritePermLo = 0x090;
constexpr uint16_t kRedirWritePermHi = 0x094;
constexpr uint16_t kRedirExecPermLo = 0x098;
constexpr uint16_t kRedirExecPermHi = 0x09c;
constexpr uint16_t kRedirPtwErrLo = 0x0a0;
constexpr uint16_t kRedirPtwErrHi = 0x0a4;
constexpr uint16_t kRedirDefaultLo = 0x0a8;
constexpr uint16_t kRedirDefaultHi = 0x0ac;
constexpr uint16_t kFixMapSizeLo = 0x0b0;
constexpr uint16_t kFixMapSizeHi = 0x0b4;

} // namespace

mmu_t::mmu_t(mmu_pte_reader_t reader)
    : reader_(std::move(reader)),
      l2_4k_{1, 128, std::vector<tlb_entry_t>(128)},
      l2_2m_{4, 256, std::vector<tlb_entry_t>(4 * 256)},
      l2_1g_{1, 128, std::vector<tlb_entry_t>(128)},
      pdc_l0_(16), pdc_l1_(16)
{
    if(!reader_) {
        reader_ = [](uint64_t) {
            return mmu_pte_read_result_t{
                mmu_pte_read_result_t::response_t::okay, 0};
        };
    }
    for(size_t port = 0; port < l1_.size(); port++) {
        l1_[port].resize(port < 4 ? 16 : 32);
    }
}

mmu_t::mmu_t(uint8_t* dram, size_t dram_size, uint64_t dram_base)
    : mmu_t(make_dram_reader(dram, dram_size, dram_base))
{
}

uint64_t mmu_t::mask_pa(uint64_t value)
{
    return value & MMU_PA_MASK;
}

uint64_t mmu_t::vpn2(uint64_t vpn)
{
    return (vpn >> 18) & 0x1ff;
}

uint64_t mmu_t::vpn1(uint64_t vpn)
{
    return (vpn >> 9) & 0x1ff;
}

uint64_t mmu_t::vpn0(uint64_t vpn)
{
    return vpn & 0x1ff;
}

uint64_t mmu_t::vpn_index(uint64_t vpn, unsigned level)
{
    switch(level) {
    case 0:
        return vpn2(vpn);
    case 1:
        return vpn1(vpn);
    default:
        return vpn0(vpn);
    }
}

uint64_t mmu_t::page_size_bytes(mmu_page_size_t size)
{
    switch(size) {
    case mmu_page_size_t::page_1g:
        return uint64_t{1} << 30;
    case mmu_page_size_t::page_2m:
        return uint64_t{1} << 21;
    case mmu_page_size_t::page_4k:
    default:
        return uint64_t{1} << 12;
    }
}

bool mmu_t::page_match(const tlb_entry_t& entry, uint64_t vpn)
{
    switch(entry.page_size) {
    case mmu_page_size_t::page_1g:
        return vpn2(entry.vpn) == vpn2(vpn);
    case mmu_page_size_t::page_2m:
        return vpn2(entry.vpn) == vpn2(vpn) &&
               vpn1(entry.vpn) == vpn1(vpn);
    case mmu_page_size_t::page_4k:
    default:
        return (entry.vpn & kVpnMask) == (vpn & kVpnMask);
    }
}

bool mmu_t::permission_ok(const tlb_entry_t& entry,
                          mmu_access_type_t access,
                          mmu_fault_code_t* fault)
{
    if(!entry.valid) {
        if(fault) {
            *fault = mmu_fault_code_t::pte_invalid;
        }
        return false;
    }
    switch(access) {
    case mmu_access_type_t::read:
        if(!entry.r) {
            if(fault) {
                *fault = mmu_fault_code_t::read_perm;
            }
            return false;
        }
        break;
    case mmu_access_type_t::write:
        if(!entry.w) {
            if(fault) {
                *fault = mmu_fault_code_t::write_perm;
            }
            return false;
        }
        break;
    case mmu_access_type_t::execute:
        if(!entry.x) {
            if(fault) {
                *fault = mmu_fault_code_t::exec_perm;
            }
            return false;
        }
        break;
    }
    if(fault) {
        *fault = mmu_fault_code_t::none;
    }
    return true;
}

mmu_config_snapshot_t mmu_t::snapshot() const
{
    std::shared_lock lock(mutex_);
    return config_;
}

mmu_request_t mmu_t::capture_request(uint64_t va, mmu_access_type_t access,
                                     uint8_t port) const
{
    std::shared_lock lock(mutex_);
    mmu_request_t request;
    request.va = mask_pa(va);
    request.access = access;
    request.port = port < l1_.size() ? port : 0;
    request.config = config_;
    return request;
}

mmu_translation_result_t mmu_t::translate(uint64_t va,
                                          mmu_access_type_t access,
                                          uint8_t port)
{
    mmu_request_t request;
    {
        std::shared_lock lock(mutex_);
        request.va = mask_pa(va);
        request.access = access;
        request.port = port < l1_.size() ? port : 0;
        request.config = config_;
    }
    return translate(request);
}

mmu_translation_result_t mmu_t::translate(const mmu_request_t& input)
{
    std::unique_lock lock(mutex_);
    mmu_request_t request = input;
    request.va = mask_pa(request.va);
    if(request.port >= l1_.size()) {
        request.port = 0;
    }
    request.config.ptw_base = mask_pa(request.config.ptw_base);
    request.config.fixmap.base = mask_pa(request.config.fixmap.base);
    request.config.fixmap.size_pages = mask_pa(request.config.fixmap.size_pages);
    request.config.redirect.page_fault = mask_pa(request.config.redirect.page_fault);
    request.config.redirect.read_perm = mask_pa(request.config.redirect.read_perm);
    request.config.redirect.write_perm = mask_pa(request.config.redirect.write_perm);
    request.config.redirect.exec_perm = mask_pa(request.config.redirect.exec_perm);
    request.config.redirect.ptw_error = mask_pa(request.config.redirect.ptw_error);
    request.config.redirect.default_base = mask_pa(request.config.redirect.default_base);

    const uint64_t vpn = (request.va >> kPageOffsetBits) & kVpnMask;
    if(fixmap_hit(request.va, request.config)) {
        mmu_translation_result_t result;
        result.pa = fixmap_pa(request.va, request.config);
        result.page_size = mmu_page_size_t::page_4k;
        return result;
    }

    if(const tlb_entry_t* entry = l1_lookup(request.port, vpn,
                                             request.config.asid)) {
        if(entry->negative) {
            return make_fault(request, entry->fault, {}, 0);
        }
        mmu_fault_code_t fault = mmu_fault_code_t::none;
        if(!permission_ok(*entry, request.access, &fault)) {
            return make_fault(request, fault, {}, 0);
        }
        return make_success(request, *entry);
    }

    if(const tlb_entry_t* entry = l2_lookup(vpn, request.config.asid)) {
        if(entry->negative) {
            return make_fault(request, entry->fault, {}, 0);
        }
        const tlb_entry_t cached = *entry;
        l1_insert(request.port, cached);
        mmu_fault_code_t fault = mmu_fault_code_t::none;
        if(!permission_ok(cached, request.access, &fault)) {
            return make_fault(request, fault, {}, 0);
        }
        return make_success(request, cached);
    }

    mmu_translation_result_t walked = walk(request, vpn);
    if(!walked.ok()) {
        tlb_entry_t negative;
        negative.valid = true;
        negative.negative = true;
        negative.fault = walked.fault;
        negative.vpn = vpn;
        negative.asid = request.config.asid;
        negative.page_size = mmu_page_size_t::page_4k;
        l2_insert(negative);
        return walked;
    }

    tlb_entry_t entry;
    entry.valid = true;
    entry.vpn = vpn;
    entry.ppn = (walked.pa >> kPageOffsetBits) & kPpnMask;
    entry.asid = request.config.asid;
    entry.page_size = walked.page_size;
    entry.global = walked.global_mapping;
    entry.r = true;
    entry.w = false;
    entry.x = false;
    // Recover the leaf permission bits from the walk's PTE snapshot.  The
    // result of walk() is successful only for a leaf, so the PTE fields are
    // architectural and not a derived permission policy.
    entry.r = (walked.pte_snapshot & (uint64_t{1} << 1)) != 0;
    entry.w = (walked.pte_snapshot & (uint64_t{1} << 2)) != 0;
    entry.x = (walked.pte_snapshot & (uint64_t{1} << 3)) != 0;
    entry.u = (walked.pte_snapshot & (uint64_t{1} << 4)) != 0;
    entry.d = (walked.pte_snapshot & (uint64_t{1} << 7)) != 0;
    entry.a = (walked.pte_snapshot & (uint64_t{1} << 6)) != 0;
    entry.ppn = (walked.pte_snapshot >> 10) & kPpnMask;

    l2_insert(entry);
    l1_insert(request.port, entry);

    mmu_fault_code_t fault = mmu_fault_code_t::none;
    if(!permission_ok(entry, request.access, &fault)) {
        return make_fault(request, fault, walked.walk_addresses,
                          walked.pte_snapshot);
    }
    mmu_translation_result_t result = make_success(request, entry);
    result.walk_addresses = std::move(walked.walk_addresses);
    result.pte_snapshot = walked.pte_snapshot;
    return result;
}

bool mmu_t::fixmap_hit(uint64_t va,
                       const mmu_config_snapshot_t& config) const
{
    if(config.fixmap.n_enable) {
        return false;
    }
    const uint64_t effective_va = va & MMU_VA_MASK;
    const uint64_t page = effective_va >> kPageOffsetBits;
    return config.fixmap.size_pages == 0 ||
           page < config.fixmap.size_pages;
}

uint64_t mmu_t::fixmap_pa(uint64_t va,
                          const mmu_config_snapshot_t& config) const
{
    const uint64_t effective_va = va & MMU_VA_MASK;
    return mask_pa(config.fixmap.base + effective_va);
}

uint64_t mmu_t::redirect_pa(uint64_t va, mmu_fault_code_t fault,
                            const mmu_config_snapshot_t& config) const
{
    uint64_t base = config.redirect.default_base;
    switch(fault) {
    case mmu_fault_code_t::pte_invalid:
    case mmu_fault_code_t::illegal_pte:
        base = config.redirect.page_fault;
        break;
    case mmu_fault_code_t::read_perm:
        base = config.redirect.read_perm;
        break;
    case mmu_fault_code_t::write_perm:
        base = config.redirect.write_perm;
        break;
    case mmu_fault_code_t::exec_perm:
        base = config.redirect.exec_perm;
        break;
    case mmu_fault_code_t::ptw_bus_error:
    case mmu_fault_code_t::ptw_timeout:
        base = config.redirect.ptw_error;
        break;
    case mmu_fault_code_t::none:
    default:
        break;
    }
    if(config.redirect.offset_enable) {
        base += va & MMU_PAGE_OFFSET_MASK;
    }
    return mask_pa(base);
}

mmu_translation_result_t mmu_t::make_fault(
    const mmu_request_t& request, mmu_fault_code_t fault,
    const std::vector<uint64_t>& addresses, uint64_t pte_snapshot)
{
    mmu_translation_result_t result;
    result.pa = redirect_pa(request.va, fault, request.config);
    result.fault = fault;
    result.page_size = mmu_page_size_t::page_4k;
    result.redirected = true;
    result.walk_addresses = addresses;
    result.pte_snapshot = pte_snapshot;
    return result;
}

mmu_translation_result_t mmu_t::make_success(const mmu_request_t& request,
                                              const tlb_entry_t& entry)
{
    mmu_translation_result_t result;
    const uint64_t va = request.va;
    const uint64_t ppn = entry.ppn & kPpnMask;
    switch(entry.page_size) {
    case mmu_page_size_t::page_1g:
        result.pa = ((ppn << 12) & ~(page_size_bytes(entry.page_size) - 1)) |
                    (va & (page_size_bytes(entry.page_size) - 1));
        break;
    case mmu_page_size_t::page_2m:
        result.pa = ((ppn << 12) & ~(page_size_bytes(entry.page_size) - 1)) |
                    (va & (page_size_bytes(entry.page_size) - 1));
        break;
    case mmu_page_size_t::page_4k:
    default:
        result.pa = (ppn << 12) | (va & MMU_PAGE_OFFSET_MASK);
        break;
    }
    result.pa = mask_pa(result.pa);
    result.page_size = entry.page_size;
    result.global_mapping = entry.global;
    return result;
}

mmu_t::tlb_entry_t* mmu_t::l1_lookup(unsigned port, uint64_t vpn,
                                      uint8_t asid)
{
    for(tlb_entry_t& entry : l1_[port]) {
        if(entry.valid && page_match(entry, vpn) &&
           (entry.global || entry.asid == asid)) {
            return &entry;
        }
    }
    return nullptr;
}

const mmu_t::tlb_entry_t* mmu_t::l1_lookup(unsigned port, uint64_t vpn,
                                           uint8_t asid) const
{
    for(const tlb_entry_t& entry : l1_[port]) {
        if(entry.valid && page_match(entry, vpn) &&
           (entry.global || entry.asid == asid)) {
            return &entry;
        }
    }
    return nullptr;
}

void mmu_t::l1_insert(unsigned port, const tlb_entry_t& entry)
{
    if(!entry.valid) {
        return;
    }
    auto& entries = l1_[port];
    for(tlb_entry_t& old : entries) {
        if(old.valid && old.vpn == entry.vpn &&
           (old.global || old.asid == entry.asid)) {
            old.valid = false;
        }
    }
    size_t victim = entries.size();
    for(size_t i = 0; i < entries.size(); i++) {
        if(!entries[i].valid) {
            victim = i;
            break;
        }
    }
    if(victim == entries.size()) {
        victim = l1_lfsr_ % entries.size();
    }
    entries[victim] = entry;
    l1_lfsr_ = static_cast<uint16_t>((l1_lfsr_ << 1) ^
        (((l1_lfsr_ >> 15) ^ (l1_lfsr_ >> 13) ^
          (l1_lfsr_ >> 12) ^ (l1_lfsr_ >> 10)) & 1));
    if(l1_lfsr_ == 0) {
        l1_lfsr_ = 1;
    }
}

mmu_t::tlb_entry_t* mmu_t::l2_lookup(uint64_t vpn, uint8_t asid)
{
    auto lookup_bank = [vpn, asid](l2_bank_t& bank,
                                    unsigned bank_kind) -> tlb_entry_t* {
        unsigned set = 0;
        if(bank_kind == 0) {
            set = static_cast<unsigned>(vpn & 0x7f);
        } else if(bank_kind == 1) {
            set = static_cast<unsigned>((vpn >> 9) & 0xff);
        } else {
            set = static_cast<unsigned>((vpn >> 18) & 0x7f);
        }
        const size_t base = static_cast<size_t>(set) * bank.ways;
        for(unsigned way = 0; way < bank.ways; way++) {
            tlb_entry_t& entry = bank.entries[base + way];
            if(entry.valid && page_match(entry, vpn) &&
               (entry.global || entry.asid == asid)) {
                return &entry;
            }
        }
        return nullptr;
    };
    if(tlb_entry_t* entry = lookup_bank(l2_4k_, 0)) {
        return entry;
    }
    if(tlb_entry_t* entry = lookup_bank(l2_2m_, 1)) {
        return entry;
    }
    return lookup_bank(l2_1g_, 2);
}

const mmu_t::tlb_entry_t* mmu_t::l2_lookup(uint64_t vpn,
                                           uint8_t asid) const
{
    auto lookup_bank = [vpn, asid](const l2_bank_t& bank,
                                    unsigned bank_kind) -> const tlb_entry_t* {
        unsigned set = 0;
        if(bank_kind == 0) {
            set = static_cast<unsigned>(vpn & 0x7f);
        } else if(bank_kind == 1) {
            set = static_cast<unsigned>((vpn >> 9) & 0xff);
        } else {
            set = static_cast<unsigned>((vpn >> 18) & 0x7f);
        }
        const size_t base = static_cast<size_t>(set) * bank.ways;
        for(unsigned way = 0; way < bank.ways; way++) {
            const tlb_entry_t& entry = bank.entries[base + way];
            if(entry.valid && page_match(entry, vpn) &&
               (entry.global || entry.asid == asid)) {
                return &entry;
            }
        }
        return nullptr;
    };
    if(const tlb_entry_t* entry = lookup_bank(l2_4k_, 0)) {
        return entry;
    }
    if(const tlb_entry_t* entry = lookup_bank(l2_2m_, 1)) {
        return entry;
    }
    return lookup_bank(l2_1g_, 2);
}

void mmu_t::l2_insert(const tlb_entry_t& entry)
{
    if(!entry.valid) {
        return;
    }
    l2_bank_t* bank = &l2_4k_;
    unsigned set = static_cast<unsigned>(entry.vpn & 0x7f);
    if(entry.page_size == mmu_page_size_t::page_2m) {
        bank = &l2_2m_;
        set = static_cast<unsigned>((entry.vpn >> 9) & 0xff);
    } else if(entry.page_size == mmu_page_size_t::page_1g) {
        bank = &l2_1g_;
        set = static_cast<unsigned>((entry.vpn >> 18) & 0x7f);
    }
    const size_t base = static_cast<size_t>(set) * bank->ways;
    unsigned victim = bank->ways;
    for(unsigned way = 0; way < bank->ways; way++) {
        if(!bank->entries[base + way].valid) {
            victim = way;
            break;
        }
    }
    if(victim == bank->ways) {
        victim = replacement_lfsr_ % bank->ways;
    }
    bank->entries[base + victim] = entry;
    advance_lfsr();
}

mmu_t::pdc_lookup_t mmu_t::pdc_l0_lookup(uint16_t v2, uint8_t asid) const
{
    for(const pdc_entry_t& entry : pdc_l0_) {
        if(entry.valid && entry.vpn2 == v2 &&
           (entry.global || entry.asid == asid)) {
            return {true, entry.global, entry.ppn};
        }
    }
    return {};
}

mmu_t::pdc_lookup_t mmu_t::pdc_l1_lookup(uint16_t v2, uint16_t v1,
                                         uint8_t asid) const
{
    for(const pdc_entry_t& entry : pdc_l1_) {
        if(entry.valid && entry.vpn2 == v2 && entry.vpn1 == v1 &&
           (entry.global || entry.asid == asid)) {
            return {true, entry.global, entry.ppn};
        }
    }
    return {};
}

void mmu_t::pdc_l0_insert(uint16_t v2, uint8_t asid, bool global,
                          uint64_t ppn)
{
    size_t victim = pdc_l0_.size();
    for(size_t i = 0; i < pdc_l0_.size(); i++) {
        if(!pdc_l0_[i].valid) {
            victim = i;
            break;
        }
    }
    if(victim == pdc_l0_.size()) {
        victim = replacement_lfsr_ % pdc_l0_.size();
    }
    pdc_l0_[victim] = {true, v2, 0, asid, global, ppn & kPpnMask};
    advance_lfsr();
}

void mmu_t::pdc_l1_insert(uint16_t v2, uint16_t v1, uint8_t asid,
                          bool global, uint64_t ppn)
{
    size_t victim = pdc_l1_.size();
    for(size_t i = 0; i < pdc_l1_.size(); i++) {
        if(!pdc_l1_[i].valid) {
            victim = i;
            break;
        }
    }
    if(victim == pdc_l1_.size()) {
        victim = replacement_lfsr_ % pdc_l1_.size();
    }
    pdc_l1_[victim] = {true, v2, v1, asid, global, ppn & kPpnMask};
    advance_lfsr();
}

mmu_translation_result_t mmu_t::walk(const mmu_request_t& request,
                                     uint64_t vpn)
{
    std::vector<uint64_t> addresses;
    addresses.reserve(3);
    uint64_t current_ppn = (request.config.ptw_base >> kPageOffsetBits) & kPpnMask;
    bool accumulated_global = false;
    const uint16_t v2 = static_cast<uint16_t>(vpn2(vpn));
    const uint16_t v1 = static_cast<uint16_t>(vpn1(vpn));

    unsigned level = 0;
    while(level < 3) {
        if(level == 0) {
            const pdc_lookup_t hit = pdc_l0_lookup(v2, request.config.asid);
            if(hit.hit) {
                current_ppn = hit.ppn;
                level = 1;
                continue;
            }
        } else if(level == 1) {
            const pdc_lookup_t hit = pdc_l1_lookup(v2, v1,
                                                   request.config.asid);
            if(hit.hit) {
                current_ppn = hit.ppn;
                level = 2;
                continue;
            }
        }

        const uint64_t pte_pa = mask_pa((current_ppn << kPageOffsetBits) +
                                        (vpn_index(vpn, level) << 3));
        addresses.push_back(pte_pa);
        const mmu_pte_read_result_t read = reader_(pte_pa);
        if(read.response == mmu_pte_read_result_t::response_t::timeout) {
            return make_fault(request, mmu_fault_code_t::ptw_timeout,
                              addresses, 0);
        }
        if(read.response == mmu_pte_read_result_t::response_t::bus_error) {
            return make_fault(request, mmu_fault_code_t::ptw_bus_error,
                              addresses, read.data);
        }

        const uint64_t pte = read.data;
        const bool valid = (pte & (uint64_t{1} << 0)) != 0;
        const bool readable = (pte & (uint64_t{1} << 1)) != 0;
        const bool writable = (pte & (uint64_t{1} << 2)) != 0;
        const bool executable = (pte & (uint64_t{1} << 3)) != 0;
        const bool global = (pte & (uint64_t{1} << 5)) != 0;
        const uint64_t ppn = (pte >> 10) & kPpnMask;

        if(!valid) {
            return make_fault(request, mmu_fault_code_t::pte_invalid,
                              addresses, pte);
        }
        if(writable && !readable) {
            return make_fault(request, mmu_fault_code_t::illegal_pte,
                              addresses, pte);
        }
        if(readable || executable) {
            if((level == 0 && (ppn & ((uint64_t{1} << 18) - 1))) ||
               (level == 1 && (ppn & ((uint64_t{1} << 9) - 1)))) {
                return make_fault(request, mmu_fault_code_t::illegal_pte,
                                  addresses, pte);
            }
            mmu_translation_result_t result;
            result.fault = mmu_fault_code_t::none;
            result.page_size = level == 0
                ? mmu_page_size_t::page_1g
                : level == 1 ? mmu_page_size_t::page_2m
                             : mmu_page_size_t::page_4k;
            result.global_mapping = accumulated_global || global;
            result.walk_addresses = addresses;
            result.pte_snapshot = pte;
            const uint64_t low_bits = page_size_bytes(result.page_size) - 1;
            result.pa = mask_pa(((ppn << kPageOffsetBits) & ~low_bits) |
                                (request.va & low_bits));
            return result;
        }
        if(level == 2) {
            return make_fault(request, mmu_fault_code_t::illegal_pte,
                              addresses, pte);
        }

        accumulated_global = accumulated_global || global;
        const bool cached_global = global;
        if(level == 0) {
            pdc_l0_insert(v2, request.config.asid, cached_global, ppn);
        } else {
            pdc_l1_insert(v2, v1, request.config.asid, cached_global, ppn);
        }
        current_ppn = ppn;
        level++;
    }

    return make_fault(request, mmu_fault_code_t::illegal_pte,
                      addresses, 0);
}

void mmu_t::invalidate_all()
{
    for(auto& entries : l1_) {
        for(tlb_entry_t& entry : entries) {
            entry.valid = false;
        }
    }
    for(tlb_entry_t& entry : l2_4k_.entries) {
        entry.valid = false;
    }
    for(tlb_entry_t& entry : l2_2m_.entries) {
        entry.valid = false;
    }
    for(tlb_entry_t& entry : l2_1g_.entries) {
        entry.valid = false;
    }
    for(pdc_entry_t& entry : pdc_l0_) {
        entry.valid = false;
    }
    for(pdc_entry_t& entry : pdc_l1_) {
        entry.valid = false;
    }
}

void mmu_t::invalidate_asid(uint8_t asid)
{
    auto invalidate = [asid](auto& entries) {
        for(auto& entry : entries) {
            if(entry.valid && !entry.global && entry.asid == asid) {
                entry.valid = false;
            }
        }
    };
    for(auto& entries : l1_) {
        invalidate(entries);
    }
    invalidate(l2_4k_.entries);
    invalidate(l2_2m_.entries);
    invalidate(l2_1g_.entries);
    invalidate(pdc_l0_);
    invalidate(pdc_l1_);
}

void mmu_t::advance_lfsr()
{
    replacement_lfsr_ = static_cast<uint8_t>(
        (replacement_lfsr_ << 1) ^
        (((replacement_lfsr_ >> 7) ^ (replacement_lfsr_ >> 5) ^
          (replacement_lfsr_ >> 4) ^ (replacement_lfsr_ >> 3)) & 1));
    if(replacement_lfsr_ == 0) {
        replacement_lfsr_ = 1;
    }
}

void mmu_t::set_ptw_base(uint64_t value)
{
    std::unique_lock lock(mutex_);
    config_.ptw_base = mask_pa(value);
}

void mmu_t::set_global_asid(uint8_t value)
{
    std::unique_lock lock(mutex_);
    config_.asid = value;
}

void mmu_t::set_fixmap(bool n_enable, uint64_t base, uint64_t size_pages)
{
    std::unique_lock lock(mutex_);
    config_.fixmap.n_enable = n_enable;
    config_.fixmap.base = mask_pa(base);
    config_.fixmap.size_pages = mask_pa(size_pages);
}

void mmu_t::set_redirects(const mmu_fault_redirect_config_t& value)
{
    std::unique_lock lock(mutex_);
    config_.redirect = value;
    config_.redirect.page_fault = mask_pa(config_.redirect.page_fault);
    config_.redirect.read_perm = mask_pa(config_.redirect.read_perm);
    config_.redirect.write_perm = mask_pa(config_.redirect.write_perm);
    config_.redirect.exec_perm = mask_pa(config_.redirect.exec_perm);
    config_.redirect.ptw_error = mask_pa(config_.redirect.ptw_error);
    config_.redirect.default_base = mask_pa(config_.redirect.default_base);
}

void mmu_t::set_pte_reader(mmu_pte_reader_t reader)
{
    std::unique_lock lock(mutex_);
    reader_ = std::move(reader);
    if(!reader_) {
        reader_ = [](uint64_t) {
            return mmu_pte_read_result_t{
                mmu_pte_read_result_t::response_t::okay, 0};
        };
    }
    invalidate_all();
}

bool mmu_t::write_register(uint16_t address, uint32_t data)
{
    std::unique_lock lock(mutex_);
    switch(address) {
    case kPtwBaseLo:
        config_.ptw_base = (config_.ptw_base & ~uint64_t{0xffffffff}) | data;
        break;
    case kPtwBaseHi:
        config_.ptw_base = (config_.ptw_base & uint64_t{0xffffffff}) |
                            (uint64_t{data & 0x00ffffff} << 32);
        break;
    case kFixMapBaseLo:
        if((data & 0xfff) == 0) {
            config_.fixmap.base = (config_.fixmap.base & ~uint64_t{0xffffffff}) | data;
        }
        break;
    case kFixMapBaseHi:
        config_.fixmap.base = (config_.fixmap.base & uint64_t{0xffffffff}) |
                              (uint64_t{data & 0x00ffffff} << 32);
        break;
    case kNFixMapEn:
        config_.fixmap.n_enable = (data & 1) != 0;
        break;
    case kTlbFlush:
        if(data & 1) {
            if(flush_busy_) {
                flush_error_code_ = 1;
            } else {
                flush_busy_ = true;
                if(data & 2) {
                    invalidate_asid(static_cast<uint8_t>((data >> 2) & 0xff));
                } else {
                    invalidate_all();
                }
                flush_busy_ = false;
                flush_done_ = true;
                flush_error_code_ = 0;
            }
        }
        break;
    case kFlushAck:
        if(data & 1) {
            flush_done_ = false;
        }
        break;
    case kGlobalAsid:
        config_.asid = static_cast<uint8_t>(data & 0xff);
        break;
    case kErrorSlvMaskLo:
        if((data & 0xfff) == 0) {
            error_slv_mask_ = (error_slv_mask_ & ~uint64_t{0xffffffff}) | data;
        }
        break;
    case kErrorSlvMaskHi:
        error_slv_mask_ = (error_slv_mask_ & uint64_t{0xffffffff}) |
                          (uint64_t{data & 0x00ffffff} << 32);
        break;
    case kErrorSlvOffMode:
        config_.redirect.offset_enable = (data & 1) != 0;
        break;
    case kErrorSlvCfgClr:
        // Standalone top-level ERROR_SLV_CFG_ERR input is tied low.
        break;
    case kRedirPageFaultLo:
        if((data & 0xfff) == 0) {
            config_.redirect.page_fault =
                (config_.redirect.page_fault & ~uint64_t{0xffffffff}) | data;
        }
        break;
    case kRedirPageFaultHi:
        config_.redirect.page_fault =
            (config_.redirect.page_fault & uint64_t{0xffffffff}) |
            (uint64_t{data & 0x00ffffff} << 32);
        break;
    case kRedirReadPermLo:
        if((data & 0xfff) == 0) {
            config_.redirect.read_perm =
                (config_.redirect.read_perm & ~uint64_t{0xffffffff}) | data;
        }
        break;
    case kRedirReadPermHi:
        config_.redirect.read_perm =
            (config_.redirect.read_perm & uint64_t{0xffffffff}) |
            (uint64_t{data & 0x00ffffff} << 32);
        break;
    case kRedirWritePermLo:
        if((data & 0xfff) == 0) {
            config_.redirect.write_perm =
                (config_.redirect.write_perm & ~uint64_t{0xffffffff}) | data;
        }
        break;
    case kRedirWritePermHi:
        config_.redirect.write_perm =
            (config_.redirect.write_perm & uint64_t{0xffffffff}) |
            (uint64_t{data & 0x00ffffff} << 32);
        break;
    case kRedirExecPermLo:
        if((data & 0xfff) == 0) {
            config_.redirect.exec_perm =
                (config_.redirect.exec_perm & ~uint64_t{0xffffffff}) | data;
        }
        break;
    case kRedirExecPermHi:
        config_.redirect.exec_perm =
            (config_.redirect.exec_perm & uint64_t{0xffffffff}) |
            (uint64_t{data & 0x00ffffff} << 32);
        break;
    case kRedirPtwErrLo:
        if((data & 0xfff) == 0) {
            config_.redirect.ptw_error =
                (config_.redirect.ptw_error & ~uint64_t{0xffffffff}) | data;
        }
        break;
    case kRedirPtwErrHi:
        config_.redirect.ptw_error =
            (config_.redirect.ptw_error & uint64_t{0xffffffff}) |
            (uint64_t{data & 0x00ffffff} << 32);
        break;
    case kRedirDefaultLo:
        if((data & 0xfff) == 0) {
            config_.redirect.default_base =
                (config_.redirect.default_base & ~uint64_t{0xffffffff}) | data;
        }
        break;
    case kRedirDefaultHi:
        config_.redirect.default_base =
            (config_.redirect.default_base & uint64_t{0xffffffff}) |
            (uint64_t{data & 0x00ffffff} << 32);
        break;
    case kFixMapSizeLo:
        config_.fixmap.size_pages =
            (config_.fixmap.size_pages & ~uint64_t{0xffffffff}) | data;
        break;
    case kFixMapSizeHi:
        config_.fixmap.size_pages =
            (config_.fixmap.size_pages & uint64_t{0xffffffff}) |
            (uint64_t{data & 0x00ffffff} << 32);
        break;
    default:
        return false;
    }
    config_.ptw_base = mask_pa(config_.ptw_base);
    config_.fixmap.base = mask_pa(config_.fixmap.base);
    config_.fixmap.size_pages = mask_pa(config_.fixmap.size_pages);
    config_.redirect.page_fault = mask_pa(config_.redirect.page_fault);
    config_.redirect.read_perm = mask_pa(config_.redirect.read_perm);
    config_.redirect.write_perm = mask_pa(config_.redirect.write_perm);
    config_.redirect.exec_perm = mask_pa(config_.redirect.exec_perm);
    config_.redirect.ptw_error = mask_pa(config_.redirect.ptw_error);
    config_.redirect.default_base = mask_pa(config_.redirect.default_base);
    error_slv_mask_ = mask_pa(error_slv_mask_);
    return true;
}

uint32_t mmu_t::read_register(uint16_t address) const
{
    std::shared_lock lock(mutex_);
    const auto low = [](uint64_t value) {
        return static_cast<uint32_t>(value & 0xffffffff);
    };
    const auto high = [](uint64_t value) {
        return static_cast<uint32_t>((value >> 32) & 0x00ffffff);
    };
    switch(address) {
    case kPtwBaseLo: return low(config_.ptw_base);
    case kPtwBaseHi: return high(config_.ptw_base);
    case kFixMapBaseLo: return low(config_.fixmap.base);
    case kFixMapBaseHi: return high(config_.fixmap.base);
    case kNFixMapEn: return config_.fixmap.n_enable ? 1 : 0;
    case kFlushDone: return flush_done_ ? 1 : 0;
    case kFlushBusy: return flush_busy_ ? 1 : 0;
    case kFlushErrCode: return flush_error_code_;
    case kGlobalAsid: return config_.asid;
    case kErrorSlvMaskLo: return low(error_slv_mask_);
    case kErrorSlvMaskHi: return high(error_slv_mask_);
    case kErrorSlvOffMode: return config_.redirect.offset_enable ? 1 : 0;
    case kErrorSlvCfgErr: return 0;
    case kRedirPageFaultLo: return low(config_.redirect.page_fault);
    case kRedirPageFaultHi: return high(config_.redirect.page_fault);
    case kRedirReadPermLo: return low(config_.redirect.read_perm);
    case kRedirReadPermHi: return high(config_.redirect.read_perm);
    case kRedirWritePermLo: return low(config_.redirect.write_perm);
    case kRedirWritePermHi: return high(config_.redirect.write_perm);
    case kRedirExecPermLo: return low(config_.redirect.exec_perm);
    case kRedirExecPermHi: return high(config_.redirect.exec_perm);
    case kRedirPtwErrLo: return low(config_.redirect.ptw_error);
    case kRedirPtwErrHi: return high(config_.redirect.ptw_error);
    case kRedirDefaultLo: return low(config_.redirect.default_base);
    case kRedirDefaultHi: return high(config_.redirect.default_base);
    case kFixMapSizeLo: return low(config_.fixmap.size_pages);
    case kFixMapSizeHi: return high(config_.fixmap.size_pages);
    default: return 0;
    }
}

void mmu_t::flush(bool asid_selective, uint8_t asid)
{
    std::unique_lock lock(mutex_);
    if(flush_busy_) {
        flush_error_code_ = 1;
        return;
    }
    flush_busy_ = true;
    if(asid_selective) {
        invalidate_asid(asid);
    } else {
        invalidate_all();
    }
    flush_busy_ = false;
    flush_done_ = true;
    flush_error_code_ = 0;
}

bool mmu_t::flush_done() const
{
    std::shared_lock lock(mutex_);
    return flush_done_;
}

bool mmu_t::flush_busy() const
{
    std::shared_lock lock(mutex_);
    return flush_busy_;
}

uint8_t mmu_t::flush_error_code() const
{
    std::shared_lock lock(mutex_);
    return flush_error_code_;
}

void mmu_t::acknowledge_flush()
{
    std::unique_lock lock(mutex_);
    flush_done_ = false;
}

mmu_pte_reader_t mmu_t::make_dram_reader(uint8_t* dram, size_t dram_size,
                                         uint64_t dram_base)
{
    dram_base &= MMU_PA_MASK;
    return [dram, dram_size, dram_base](uint64_t physical_address) {
        mmu_pte_read_result_t result;
        const uint64_t address = physical_address & MMU_PA_MASK;
        if(!dram || address < dram_base) {
            result.response = mmu_pte_read_result_t::response_t::bus_error;
            return result;
        }
        const uint64_t offset = address - dram_base;
        if(offset > dram_size || dram_size - offset < sizeof(uint64_t)) {
            result.response = mmu_pte_read_result_t::response_t::bus_error;
            return result;
        }
        std::memcpy(&result.data, dram + offset, sizeof(result.data));
        result.response = mmu_pte_read_result_t::response_t::okay;
        return result;
    };
}

} // namespace RVTNG
