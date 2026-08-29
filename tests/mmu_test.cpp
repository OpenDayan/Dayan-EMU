#include "mmu.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <unordered_map>

namespace {

using RVTNG::MMU_PA_MASK;
using RVTNG::mmu_access_type_t;
using RVTNG::mmu_fault_code_t;
using RVTNG::mmu_pte_read_result_t;
using RVTNG::mmu_t;

constexpr uint64_t kRoot = 0x80000000ull;
constexpr uint64_t kL1 = 0x80001000ull;
constexpr uint64_t kL2 = 0x80002000ull;
constexpr uint64_t kLeaf = 0x0000000040003000ull;
constexpr uint64_t kPteV = 1ull << 0;
constexpr uint64_t kPteR = 1ull << 1;
constexpr uint64_t kPteW = 1ull << 2;
constexpr uint64_t kPteX = 1ull << 3;
constexpr uint64_t kPteG = 1ull << 5;

struct page_table_image_t {
    std::unordered_map<uint64_t, uint64_t> ptes;
    mmu_pte_read_result_t::response_t response =
        mmu_pte_read_result_t::response_t::okay;
    size_t reads = 0;

    mmu_pte_read_result_t read(uint64_t address) {
        ++reads;
        if(response != mmu_pte_read_result_t::response_t::okay) {
            return {response, 0};
        }
        const auto it = ptes.find(address);
        return {mmu_pte_read_result_t::response_t::okay,
                it == ptes.end() ? 0 : it->second};
    }
};

uint64_t pte_for(uint64_t physical_address, uint64_t flags) {
    return ((physical_address >> 12) << 10) | flags;
}

uint64_t vpn_of(uint64_t va) {
    return (va >> 12) & ((uint64_t{1} << 27) - 1);
}

uint64_t pte_address(uint64_t table, uint64_t index) {
    return table + index * sizeof(uint64_t);
}

void install_4k(page_table_image_t& image, uint64_t va,
                uint64_t physical_address, uint64_t leaf_flags,
                uint64_t root_flags = kPteV,
                uint64_t l1_flags = kPteV) {
    const uint64_t vpn = vpn_of(va);
    const uint64_t i2 = (vpn >> 18) & 0x1ff;
    const uint64_t i1 = (vpn >> 9) & 0x1ff;
    const uint64_t i0 = vpn & 0x1ff;
    image.ptes[pte_address(kRoot, i2)] = pte_for(kL1, root_flags);
    image.ptes[pte_address(kL1, i1)] = pte_for(kL2, l1_flags);
    image.ptes[pte_address(kL2, i0)] = pte_for(physical_address, leaf_flags);
}

std::unique_ptr<mmu_t> make_mmu(page_table_image_t& image) {
    auto mmu = std::make_unique<mmu_t>(
        [&image](uint64_t address) { return image.read(address); });
    mmu->set_fixmap(true, 0, 0);
    mmu->set_ptw_base(kRoot);
    return mmu;
}

void test_fixmap() {
    mmu_t mmu;
    auto result = mmu.translate(0x1234, mmu_access_type_t::read);
    assert(result.ok() && result.pa == 0x1234);

    mmu.set_fixmap(false, 0x100000, 2);
    result = mmu.translate(0x1000, mmu_access_type_t::write);
    assert(result.ok() && result.pa == 0x101000);
    result = mmu.translate(0x2000, mmu_access_type_t::read);
    assert(!result.ok());

    const uint64_t high_va =
        ((~RVTNG::MMU_VA_MASK) & MMU_PA_MASK) | (uint64_t{1} << 38);
    mmu.set_fixmap(false, 0x2000, 0);
    const uint64_t rtl_pa = mmu.translate(
        high_va, mmu_access_type_t::read).pa;
    assert(rtl_pa == ((0x2000 + (high_va & RVTNG::MMU_VA_MASK)) & MMU_PA_MASK));
}

void test_page_sizes() {
    {
        page_table_image_t image;
        constexpr uint64_t va = 0x0000000012345678ull;
        install_4k(image, va, kLeaf, kPteV | kPteR | kPteW);
        auto mmu = make_mmu(image);
        const auto result = mmu->translate(va + 0x321,
                                          mmu_access_type_t::write);
        assert(result.ok());
        assert(result.page_size == RVTNG::mmu_page_size_t::page_4k);
        assert(result.pa == (kLeaf + ((va + 0x321) & 0xfff)));
        assert(result.walk_addresses.size() == 3);
    }

    {
        page_table_image_t image;
        constexpr uint64_t va = 0x0000000012345678ull;
        const uint64_t vpn = vpn_of(va);
        image.ptes[pte_address(kRoot, (vpn >> 18) & 0x1ff)] =
            pte_for(0x80001000, kPteV);
        const uint64_t physical = 0x0000000040000000ull;
        image.ptes[pte_address(kL1, (vpn >> 9) & 0x1ff)] =
            pte_for(physical, kPteV | kPteR);
        auto mmu = make_mmu(image);
        const auto result = mmu->translate(va, mmu_access_type_t::read);
        assert(result.ok());
        assert(result.page_size == RVTNG::mmu_page_size_t::page_2m);
        assert(result.pa == physical + (va & 0x1fffff));
    }

    {
        page_table_image_t image;
        constexpr uint64_t va = 0x0000000012345678ull;
        const uint64_t physical = 0x0000000080000000ull;
        const uint64_t vpn = vpn_of(va);
        image.ptes[pte_address(kRoot, (vpn >> 18) & 0x1ff)] =
            pte_for(physical, kPteV | kPteR | kPteX);
        auto mmu = make_mmu(image);
        const auto result = mmu->translate(va, mmu_access_type_t::execute);
        assert(result.ok());
        assert(result.page_size == RVTNG::mmu_page_size_t::page_1g);
        assert(result.pa == physical + (va & 0x3fffffff));
    }
}

void test_faults_and_permissions() {
    page_table_image_t image;
    constexpr uint64_t va = 0x0000000000456789ull;
    install_4k(image, va, kLeaf, kPteV | kPteR);
    auto mmu = make_mmu(image);

    RVTNG::mmu_fault_redirect_config_t redirects;
    redirects.page_fault = 0x10000000;
    redirects.read_perm = 0x20000000;
    redirects.write_perm = 0x30000000;
    redirects.exec_perm = 0x40000000;
    redirects.ptw_error = 0x50000000;
    redirects.offset_enable = true;
    mmu->set_redirects(redirects);

    auto result = mmu->translate(va, mmu_access_type_t::read);
    assert(result.ok() && result.pa == kLeaf + (va & 0xfff));
    result = mmu->translate(va, mmu_access_type_t::write);
    assert(result.fault == mmu_fault_code_t::write_perm);
    assert(result.pa == redirects.write_perm + (va & 0xfff));
    result = mmu->translate(va, mmu_access_type_t::execute);
    assert(result.fault == mmu_fault_code_t::exec_perm);
    assert(result.pa == redirects.exec_perm + (va & 0xfff));

    page_table_image_t invalid;
    auto invalid_mmu = make_mmu(invalid);
    result = invalid_mmu->translate(va, mmu_access_type_t::read);
    assert(result.fault == mmu_fault_code_t::pte_invalid);

    page_table_image_t illegal;
    const uint64_t vpn = vpn_of(va);
    illegal.ptes[pte_address(kRoot, (vpn >> 18) & 0x1ff)] =
        pte_for(kL1, kPteV | kPteW);
    auto illegal_mmu = make_mmu(illegal);
    result = illegal_mmu->translate(va, mmu_access_type_t::read);
    assert(result.fault == mmu_fault_code_t::illegal_pte);

    page_table_image_t misaligned;
    misaligned.ptes[pte_address(kRoot, (vpn >> 18) & 0x1ff)] =
        pte_for(0x80001000 + (1ull << 12), kPteV | kPteR);
    auto misaligned_mmu = make_mmu(misaligned);
    result = misaligned_mmu->translate(va, mmu_access_type_t::read);
    assert(result.fault == mmu_fault_code_t::illegal_pte);
}

void test_negative_cache_and_flush() {
    page_table_image_t image;
    constexpr uint64_t va = 0x0000000000123456ull;
    image.response = mmu_pte_read_result_t::response_t::bus_error;
    auto mmu = make_mmu(image);
    auto result = mmu->translate(va, mmu_access_type_t::read);
    assert(result.fault == mmu_fault_code_t::ptw_bus_error);
    const size_t reads_after_fault = image.reads;

    image.response = mmu_pte_read_result_t::response_t::okay;
    install_4k(image, va, kLeaf, kPteV | kPteR);
    result = mmu->translate(va, mmu_access_type_t::read);
    assert(result.fault == mmu_fault_code_t::ptw_bus_error);
    assert(image.reads == reads_after_fault);

    mmu->flush(false, 0);
    result = mmu->translate(va, mmu_access_type_t::read);
    assert(result.ok() && result.pa == kLeaf + (va & 0xfff));
    assert(mmu->flush_done());
    mmu->acknowledge_flush();
    assert(!mmu->flush_done());

    page_table_image_t timeout;
    timeout.response = mmu_pte_read_result_t::response_t::timeout;
    auto timeout_mmu = make_mmu(timeout);
    result = timeout_mmu->translate(va, mmu_access_type_t::read);
    assert(result.fault == mmu_fault_code_t::ptw_timeout);
}

void test_apb_and_snapshot() {
    mmu_t mmu;
    assert(mmu.write_register(0x008, 0x00000123));
    assert(mmu.read_register(0x008) == 0);
    assert(mmu.write_register(0x008, 0x00123000));
    assert(mmu.read_register(0x008) == 0x00123000);
    assert(mmu.write_register(0x00c, 0x00000001));
    assert(mmu.read_register(0x00c) == 1);
    assert(mmu.write_register(0x010, 1));
    assert(mmu.read_register(0x010) == 1);
    assert(mmu.write_register(0x050, 1));
    assert(mmu.read_register(0x050) == 1);
    assert(mmu.write_register(0x048, 0x00001000));
    assert(mmu.read_register(0x048) == 0x00001000);
    assert(mmu.write_register(0x04c, 0x00000002));
    assert(mmu.read_register(0x04c) == 2);

    mmu.set_fixmap(false, 0x1000, 0);
    auto request = mmu.capture_request(0x2000, mmu_access_type_t::read);
    mmu.set_fixmap(false, 0x4000, 0);
    auto result = mmu.translate(request);
    assert(result.ok() && result.pa == 0x3000);

    assert(mmu.write_register(0x014, 1));
    assert(mmu.read_register(0x01c) == 1);
    assert(mmu.write_register(0x018, 1));
    assert(mmu.read_register(0x01c) == 0);
    assert(!mmu.write_register(0xffff, 1));
}

void test_dram_reader_bounds() {
    std::array<uint8_t, 16> bytes{};
    uint64_t value = 0x1122334455667788ull;
    std::memcpy(bytes.data(), &value, sizeof(value));
    const auto reader = mmu_t::make_dram_reader(
        bytes.data(), bytes.size(), 0x1000);
    assert(reader(0x1000).response ==
           mmu_pte_read_result_t::response_t::okay);
    assert(reader(0x1008).response ==
           mmu_pte_read_result_t::response_t::okay);
    assert(reader(0x1009).response ==
           mmu_pte_read_result_t::response_t::bus_error);
    assert(reader(0x0ff8).response ==
           mmu_pte_read_result_t::response_t::bus_error);
}

void test_asid_flush() {
    page_table_image_t image;
    constexpr uint64_t va = 0x0000000000567000ull;
    install_4k(image, va, kLeaf, kPteV | kPteR);
    auto mmu = make_mmu(image);

    mmu->set_global_asid(1);
    assert(mmu->translate(va, mmu_access_type_t::read).ok());
    mmu->set_global_asid(2);
    assert(mmu->translate(va, mmu_access_type_t::read).ok());
    const size_t before_flush = image.reads;

    mmu->flush(true, 1);
    mmu->set_global_asid(2);
    assert(mmu->translate(va, mmu_access_type_t::read).ok());
    assert(image.reads == before_flush);

    mmu->set_global_asid(1);
    assert(mmu->translate(va, mmu_access_type_t::read).ok());
    assert(image.reads > before_flush);
}

void test_global_pdc_rtl_behavior() {
    page_table_image_t image;
    constexpr uint64_t va0 = 0x0000000000011000ull;
    constexpr uint64_t va1 = 0x0000000000012000ull;
    install_4k(image, va0, kLeaf, kPteV | kPteR,
               kPteV | kPteG, kPteV);
    const uint64_t vpn1 = vpn_of(va1);
    image.ptes[pte_address(kL2, vpn1 & 0x1ff)] =
        pte_for(kLeaf + 0x1000, kPteV | kPteR);
    auto mmu = make_mmu(image);
    mmu->set_global_asid(1);
    assert(mmu->translate(va0, mmu_access_type_t::read).ok());
    mmu->set_global_asid(2);
    const auto result = mmu->translate(va1, mmu_access_type_t::read);
    assert(result.ok() && !result.global_mapping);
}

} // namespace

int main() {
    test_fixmap();
    test_page_sizes();
    test_faults_and_permissions();
    test_negative_cache_and_flush();
    test_apb_and_snapshot();
    test_dram_reader_bounds();
    test_asid_flush();
    test_global_pdc_rtl_behavior();
    std::cout << "MMU tests passed\n";
    return 0;
}
