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

#include "lsu.hpp"
#include "dayan_decoder.hpp"
#include "swizzle.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr uint64_t kDramBase = 0x80000000ull;
constexpr size_t kUramSize = 1024 * 1024;
constexpr size_t kTensorUramSize = 1536 * 1024;
constexpr uint64_t kTensorUramBase = 0x00800000ull;
constexpr uint64_t kTensorUramStride = 2ull * 1024ull * 1024ull;
constexpr size_t kDramSize = 128 * 1024;
constexpr size_t kRowBytes = 64;

class test_lsu_t : public DAYAN::lsu_t {
public:
    using DAYAN::lsu_t::check_uram_access;
    using DAYAN::lsu_t::execute_dma;
    using DAYAN::lsu_t::execute_dma_once;
    using DAYAN::lsu_t::lsu_t;

    void set_active_context(const DAYAN::tensor_context_snapshot_t& value) {
        active_context = value;
    }
};

DAYAN::lsu_tcsr_t make_control() {
    DAYAN::lsu_tcsr_t control{};
    control.inBlockStride = kRowBytes;
    control.inDim0Stride = kRowBytes;
    control.outBlockStride = kRowBytes;
    control.outDim0Stride = kRowBytes;
    control.blockTranspose = 6;
    control.requestReadSize = kRowBytes - 1;
    control.requestWriteSize = kRowBytes - 1;
    control.tableStride = 0x100;
    return control;
}

uint8_t* memory_ptr(std::vector<uint8_t>& uram, std::vector<uint8_t>& dram,
                    bool is_dram, uint64_t addr) {
    return is_dram ? dram.data() + addr - kDramBase : uram.data() + addr;
}

void store_u64(std::vector<uint8_t>& uram, uint64_t addr, uint64_t value) {
    std::memcpy(uram.data() + addr, &value, sizeof(value));
}

void store_dram_u64(std::vector<uint8_t>& dram, uint64_t addr, uint64_t value) {
    std::memcpy(dram.data() + addr - kDramBase, &value, sizeof(value));
}

std::array<uint8_t, kRowBytes> pattern(unsigned seed) {
    std::array<uint8_t, kRowBytes> data{};
    for(size_t byte = 0; byte < data.size(); byte++) {
        data[byte] = static_cast<uint8_t>((seed * 37 + byte * 13) & 0xff);
    }
    return data;
}

uint64_t pack_swizzle(uint8_t m, uint8_t b, uint8_t s) {
    return (uint64_t{m} << 16) | (uint64_t{b} << 8) | s;
}

void test_swizzle_address_bits() {
    constexpr uint64_t swizzle =
        (uint64_t{5} << 16) | (uint64_t{3} << 8) | 5;
    constexpr uint64_t high_bits = 0x00808000;

    assert(DAYAN::swizzle_addr_raw(0x1234, swizzle) == 0x12b4);
    assert(DAYAN::swizzle_addr_raw(high_bits | 0x1234, swizzle) ==
           (high_bits | 0x12b4));
    assert(DAYAN::swizzle_addr_raw(
               0x1234, pack_swizzle(0x15, 0x13, 0x15)) == 0x12b4);
}

void test_decoder() {
    constexpr uint32_t base = 0x0800000b;
    assert(IS_RVT_LSU_DMA(base));
    assert(IS_RVT_LSU_DMA(base | (1u << 23)));
    assert(IS_RVT_LSU_DMA(base | (1u << 14)));
    assert(IS_RVT_LSU_DMA(base | (1u << 23) | (1u << 14)));
    assert(DAYAN_LSU_DMA_GATHER(base | (1u << 23)) == 1);
    assert(DAYAN_LSU_DMA_SCATTER(base | (1u << 14)) == 1);
    assert(!IS_RVT_LSU_DMA(base | (1u << 24)));
}

void test_unaligned_partial_copy(test_lsu_t& lsu, std::vector<uint8_t>& uram,
                                 std::vector<uint8_t>& dram) {
    auto control = make_control();
    control.requestReadSize = 4;
    control.requestWriteSize = 10;
    const std::array<uint8_t, 8> padding = {0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7};
    std::copy(padding.begin(), padding.end(), control.paddingValue);

    const auto source = pattern(3);
    for(bool src_dram : {false, true}) {
        for(bool dst_dram : {false, true}) {
            const uint64_t src = (src_dram ? kDramBase : 0) + 0x1003;
            const uint64_t dst = (dst_dram ? kDramBase : 0) + 0x3007;
            std::copy(source.begin(), source.end(),
                      memory_ptr(uram, dram, src_dram, src));
            std::fill(memory_ptr(uram, dram, dst_dram, dst) - 4,
                      memory_ptr(uram, dram, dst_dram, dst) + kRowBytes + 4, 0xa5);

            assert(lsu.execute_dma_once(&control, src, dst, src_dram, dst_dram));
            const uint8_t* result = memory_ptr(uram, dram, dst_dram, dst);
            for(size_t byte = 0; byte <= control.requestWriteSize; byte++) {
                const uint8_t expected = byte <= control.requestReadSize
                    ? source[byte]
                    : padding[byte % padding.size()];
                assert(result[byte] == expected);
            }
            for(size_t byte = control.requestWriteSize + 1; byte < kRowBytes; byte++) {
                assert(result[byte] == 0xa5);
            }
            assert(result[-1] == 0xa5);
            assert(result[kRowBytes] == 0xa5);
        }
    }
}

void test_tail_store_sizes(test_lsu_t& lsu, std::vector<uint8_t>& uram,
                           std::vector<uint8_t>& dram) {
    auto control = make_control();
    control.requestReadSize = 7;
    control.requestWriteSize = 63;

    const uint64_t src = 0x5003;
    const uint64_t dst = kDramBase + 0x7007;
    const auto source = pattern(7);
    std::copy(source.begin(), source.end(), uram.data() + src);
    std::fill(dram.data() + dst - kDramBase,
              dram.data() + dst - kDramBase + kRowBytes, 0xa5);

    assert(lsu.execute_dma_once(&control, src, dst, false, true));
    const uint8_t* wide_result = dram.data() + dst - kDramBase;
    assert(std::equal(source.begin(), source.begin() + 8, wide_result));
    assert(std::all_of(wide_result + 8, wide_result + kRowBytes,
                       [](uint8_t value) { return value == 0; }));

    control.requestWriteSize = 7;
    std::fill(dram.data() + dst - kDramBase,
              dram.data() + dst - kDramBase + kRowBytes, 0xa5);
    assert(lsu.execute_dma_once(&control, src, dst, false, true));
    const uint8_t* tail_result = dram.data() + dst - kDramBase;
    assert(std::equal(source.begin(), source.begin() + 8, tail_result));
    assert(std::all_of(tail_result + 8, tail_result + kRowBytes,
                       [](uint8_t value) { return value == 0xa5; }));
}

void test_gather_scatter_all_directions(test_lsu_t& lsu,
                                        std::vector<uint8_t>& uram,
                                        std::vector<uint8_t>& dram) {
    constexpr uint64_t gather_table = 0x000;
    constexpr uint64_t scatter_table = 0x100;
    constexpr std::array<unsigned, 3> source_order = {2, 0, 1};
    constexpr std::array<unsigned, 3> destination_order = {1, 2, 0};

    for(bool src_dram : {false, true}) {
        for(bool dst_dram : {false, true}) {
            for(bool gather : {false, true}) {
                for(bool scatter : {false, true}) {
                    std::fill(uram.begin(), uram.end(), 0);
                    std::fill(dram.begin(), dram.end(), 0);
                    auto control = make_control();
                    control.tableSize = 2;

                    const uint64_t src_base = (src_dram ? kDramBase : 0) + 0x2003;
                    const uint64_t dst_base = (dst_dram ? kDramBase : 0) + 0x6007;
                    std::array<uint64_t, 3> sources{};
                    std::array<uint64_t, 3> destinations{};
                    std::array<std::array<uint8_t, kRowBytes>, 3> expected{};

                    for(size_t index = 0; index < sources.size(); index++) {
                        const unsigned src_slot = gather ? source_order[index] : index;
                        const unsigned dst_slot = scatter ? destination_order[index] : index;
                        sources[index] = src_base + src_slot * control.tableStride;
                        destinations[index] = dst_base + dst_slot * control.tableStride;
                        expected[index] = pattern(static_cast<unsigned>(index + 11));
                        std::copy(expected[index].begin(), expected[index].end(),
                                  memory_ptr(uram, dram, src_dram, sources[index]));
                        std::fill(memory_ptr(uram, dram, dst_dram, destinations[index]),
                                  memory_ptr(uram, dram, dst_dram, destinations[index]) + kRowBytes,
                                  0xa5);
                        if(gather) {
                            store_u64(uram, gather_table + index * sizeof(uint64_t), sources[index]);
                        }
                        if(scatter) {
                            store_u64(uram, scatter_table + index * sizeof(uint64_t), destinations[index]);
                        }
                    }

                    lsu.execute_dma(&control,
                                    gather ? gather_table : src_base,
                                    scatter ? scatter_table : dst_base,
                                    src_dram, dst_dram, gather, scatter);

                    for(size_t index = 0; index < destinations.size(); index++) {
                        const uint8_t* result = memory_ptr(uram, dram, dst_dram, destinations[index]);
                        assert(std::equal(expected[index].begin(), expected[index].end(), result));
                    }
                }
            }
        }
    }
}

void test_unaligned_uram_swizzle(std::vector<uint8_t>& uram,
                                 std::vector<uint8_t>& dram) {
    DAYAN::exec_unit_t parent;
    DAYAN::tensor_context_t context;
    DAYAN::interrupt_controller_t interrupts;
    test_lsu_t lsu(&parent, uram.data(), dram.data(), nullptr, 0,
                   &context, &interrupts);
    auto store_control = make_control();
    store_control.requestReadSize = 31;
    store_control.requestWriteSize = 31;
    store_control.uramWriteSwizzleMBS = pack_swizzle(5, 1, 1);
    const uint64_t dram_src = kDramBase + 0x3005;
    const uint64_t uram_addr = 0x0050;
    const uint64_t dram_dst = kDramBase + 0x7009;
    const auto source = pattern(19);
    std::copy(source.begin(), source.end(), memory_ptr(uram, dram, true, dram_src));
    std::fill(uram.begin() + 0x40, uram.begin() + 0x80, 0xa5);

    assert(lsu.execute_dma_once(&store_control, dram_src, uram_addr, true, false));
    const uint64_t first_port = DAYAN::swizzle_addr(0x40,
        store_control.uramWriteSwizzleMBS);
    const uint64_t second_port = DAYAN::swizzle_addr(0x60,
        store_control.uramWriteSwizzleMBS);
    assert(first_port == 0x60);
    assert(second_port == 0x40);
    assert(std::equal(source.begin(), source.begin() + 16,
                      uram.begin() + first_port + 16));
    assert(std::equal(source.begin() + 16, source.begin() + 32,
                      uram.begin() + second_port));
    assert(uram[first_port + 15] == 0xa5);
    assert(uram[second_port + 16] == 0xa5);

    auto load_control = make_control();
    load_control.requestReadSize = 31;
    load_control.requestWriteSize = 31;
    load_control.uramReadSwizzleMBS = store_control.uramWriteSwizzleMBS;
    assert(lsu.execute_dma_once(&load_control, uram_addr, dram_dst, false, true));
    assert(std::equal(source.begin(), source.begin() + 32,
                      memory_ptr(uram, dram, true, dram_dst)));
    assert(interrupts.count() == 0);
}

void test_block_rows_dominate_padding(test_lsu_t& lsu, std::vector<uint8_t>& uram,
                                      std::vector<uint8_t>& dram) {
    auto control = make_control();
    control.block = 2;
    control.blockPadding = 0;
    const uint64_t src = kDramBase + 0x9001;
    const uint64_t dst = kDramBase + 0xa007;
    for(unsigned row = 0; row < 3; row++) {
        const auto data = pattern(30 + row);
        std::copy(data.begin(), data.end(),
                  memory_ptr(uram, dram, true, src + row * kRowBytes));
    }

    assert(lsu.execute_dma_once(&control, src, dst, true, true));
    for(unsigned row = 0; row < 3; row++) {
        const auto expected = pattern(30 + row);
        assert(std::equal(expected.begin(), expected.end(),
                          memory_ptr(uram, dram, true, dst + row * kRowBytes)));
    }
}

void test_aligned_transpose(test_lsu_t& lsu, std::vector<uint8_t>& uram,
                            std::vector<uint8_t>& dram) {
    auto control = make_control();
    control.block = 1;
    control.blockPadding = 1;
    control.blockTranspose = 5;
    const uint64_t src = kDramBase + 0x4000;
    const uint64_t dst = kDramBase + 0x5000;
    auto first = pattern(21);
    auto second = pattern(22);
    std::copy(first.begin(), first.end(), memory_ptr(uram, dram, true, src));
    std::copy(second.begin(), second.end(), memory_ptr(uram, dram, true, src + kRowBytes));

    assert(lsu.execute_dma_once(&control, src, dst, true, true));
    const uint8_t* row0 = memory_ptr(uram, dram, true, dst);
    const uint8_t* row1 = row0 + kRowBytes;
    assert(std::equal(first.begin(), first.begin() + 32, row0));
    assert(std::equal(second.begin(), second.begin() + 32, row0 + 32));
    assert(std::equal(first.begin() + 32, first.end(), row1));
    assert(std::equal(second.begin() + 32, second.end(), row1 + 32));

    control.blockTranspose = 5;
    assert(!lsu.execute_dma_once(&control, src + 1, dst, true, true));
}

void test_tensor_uram_windows(std::vector<uint8_t>& dram) {
    std::array<std::vector<uint8_t>, 4> uram = {
        std::vector<uint8_t>(kTensorUramSize),
        std::vector<uint8_t>(kTensorUramSize),
        std::vector<uint8_t>(kTensorUramSize),
        std::vector<uint8_t>(kTensorUramSize),
    };
    std::array<uint8_t*, 4> windows = {
        uram[0].data(), uram[1].data(), uram[2].data(), uram[3].data(),
    };
    DAYAN::exec_unit_t parent;
    test_lsu_t lsu(&parent, uram[0].data(), dram.data(), windows.data(),
                   windows.size());
    auto control = make_control();
    const auto source = pattern(41);
    constexpr uint64_t remote_offset = 0x12340;
    constexpr uint64_t local_offset = 0x54320;
    const uint64_t remote_source =
        kTensorUramBase + 3 * kTensorUramStride + remote_offset;

    std::copy(source.begin(), source.end(),
              uram[3].data() + remote_offset);
    assert(lsu.execute_dma_once(&control, remote_source, local_offset,
                                true, false));
    assert(std::equal(source.begin(), source.end(),
                      uram[0].data() + local_offset));

    const uint64_t remote_destination =
        kTensorUramBase + kTensorUramStride + remote_offset;
    assert(lsu.execute_dma_once(&control, local_offset, remote_destination,
                                false, true));
    assert(std::equal(source.begin(), source.end(),
                      uram[1].data() + remote_offset));

    const uint64_t unmapped_gap =
        kTensorUramBase + 2 * kTensorUramStride + kTensorUramSize;
    assert(!lsu.execute_dma_once(&control, unmapped_gap, local_offset,
                                 true, false));
}

void test_mmu_dram_translation(std::vector<uint8_t>& uram,
                               std::vector<uint8_t>& dram) {
    std::fill(dram.begin(), dram.end(), 0);
    constexpr uint64_t root = kDramBase;
    constexpr uint64_t l1 = kDramBase + 0x1000;
    constexpr uint64_t l2 = kDramBase + 0x2000;
    constexpr uint64_t source_va = 0x0000000080401000ull;
    constexpr uint64_t destination_va = 0x0000000080402000ull;
    constexpr uint64_t source_pa = kDramBase + 0x9000;
    constexpr uint64_t destination_pa = kDramBase + 0xa000;
    constexpr uint64_t pte_v = 1ull << 0;
    constexpr uint64_t pte_r = 1ull << 1;
    constexpr uint64_t pte_w = 1ull << 2;

    auto vpn = [](uint64_t va) { return (va >> 12) & ((uint64_t{1} << 27) - 1); };
    auto pte = [](uint64_t pa, uint64_t flags) {
        return ((pa >> 12) << 10) | flags;
    };
    auto pte_address = [](uint64_t table, uint64_t index) {
        return table + index * sizeof(uint64_t);
    };
    const uint64_t source_vpn = vpn(source_va);
    const uint64_t destination_vpn = vpn(destination_va);
    store_dram_u64(dram, pte_address(root, (source_vpn >> 18) & 0x1ff),
                   pte(l1, pte_v));
    store_dram_u64(dram, pte_address(l1, (source_vpn >> 9) & 0x1ff),
                   pte(l2, pte_v));
    store_dram_u64(dram, pte_address(l2, source_vpn & 0x1ff),
                   pte(source_pa, pte_v | pte_r | pte_w));
    store_dram_u64(dram, pte_address(l2, destination_vpn & 0x1ff),
                   pte(destination_pa, pte_v | pte_r | pte_w));

    const auto source = pattern(73);
    std::copy(source.begin(), source.end(),
              dram.data() + source_pa - kDramBase);
    std::fill(dram.data() + destination_pa - kDramBase,
              dram.data() + destination_pa - kDramBase + kRowBytes, 0xa5);

    DAYAN::exec_unit_t parent;
    DAYAN::mmu_t mmu(dram.data(), dram.size());
    mmu.set_fixmap(true, 0, 0);
    mmu.set_ptw_base(root);
    test_lsu_t lsu(&parent, uram.data(), dram.data(), nullptr, 0, nullptr,
                   nullptr, &mmu, dram.size(), kDramBase, 4);
    auto control = make_control();
    assert(lsu.execute_dma_once(&control, source_va, destination_va,
                                true, true));
    assert(std::equal(source.begin(), source.end(),
                      dram.data() + destination_pa - kDramBase));
}

void test_mmu_page_crossing_translation(std::vector<uint8_t>& uram,
                                        std::vector<uint8_t>& dram) {
    std::fill(dram.begin(), dram.end(), 0);
    constexpr uint64_t root = kDramBase;
    constexpr uint64_t l1 = kDramBase + 0x1000;
    constexpr uint64_t l2 = kDramBase + 0x2000;
    constexpr uint64_t pte_v = 1ull << 0;
    constexpr uint64_t pte_r = 1ull << 1;
    constexpr uint64_t pte_w = 1ull << 2;

    auto vpn = [](uint64_t va) {
        return (va >> 12) & ((uint64_t{1} << 27) - 1);
    };
    auto pte = [](uint64_t pa, uint64_t flags) {
        return ((pa >> 12) << 10) | flags;
    };
    auto pte_address = [](uint64_t table, uint64_t index) {
        return table + index * sizeof(uint64_t);
    };
    auto store_leaf = [&](uint64_t va, uint64_t pa, uint64_t flags) {
        const uint64_t entry = vpn(va);
        store_dram_u64(dram, pte_address(root, (entry >> 18) & 0x1ff),
                       pte(l1, pte_v));
        store_dram_u64(dram, pte_address(l1, (entry >> 9) & 0x1ff),
                       pte(l2, pte_v));
        store_dram_u64(dram, pte_address(l2, entry & 0x1ff), pte(pa, flags));
    };

    // Every burst starts 32 B before a page end, so its 64 B transfer splits
    // into a 32 B head and a 32 B tail on the next virtual page.  Each page
    // pair maps to physical pages that are not contiguous, which only works
    // if every page is translated on its own.  All pages share one level-1
    // entry and use distinct level-2 slots (vpn[8:0]), so the leaf PTEs do
    // not collide inside the shared level-2 table.
    constexpr uint64_t source_va = 0x0000000080401fe0ull;
    constexpr uint64_t destination_va = 0x0000000080403fe0ull;
    constexpr uint64_t perm_destination_va = 0x0000000080405fe0ull;
    constexpr uint64_t unmapped_source_va = 0x0000000080407fe0ull;
    constexpr uint64_t source_pa = kDramBase + 0x9000;
    constexpr uint64_t source_next_pa = kDramBase + 0x11000;
    constexpr uint64_t destination_pa = kDramBase + 0xa000;
    constexpr uint64_t destination_next_pa = kDramBase + 0x12000;
    constexpr uint64_t scratch_pa = kDramBase + 0xb000;
    constexpr uint64_t perm_destination_pa = kDramBase + 0xc000;
    constexpr uint64_t perm_destination_next_pa = kDramBase + 0x14000;

    store_leaf(source_va, source_pa, pte_v | pte_r | pte_w);
    store_leaf(source_va + 0x20, source_next_pa, pte_v | pte_r | pte_w);
    store_leaf(destination_va, destination_pa, pte_v | pte_r | pte_w);
    store_leaf(destination_va + 0x20, destination_next_pa,
               pte_v | pte_r | pte_w);
    store_leaf(perm_destination_va, perm_destination_pa,
               pte_v | pte_r | pte_w);
    // Second page of this write burst lacks W.
    store_leaf(perm_destination_va + 0x20, perm_destination_next_pa,
               pte_v | pte_r);
    store_leaf(unmapped_source_va, scratch_pa, pte_v | pte_r | pte_w);
    // Second page of this read burst has no valid PTE at all.

    constexpr size_t cross_bytes = 0x1000 - (source_va & 0xfff);
    const auto first = pattern(81);
    const auto second = pattern(82);
    std::copy(first.begin(), first.end(),
              dram.data() + source_pa - kDramBase + (source_va & 0xfff));
    std::copy(second.begin(), second.end(),
              dram.data() + source_next_pa - kDramBase);
    std::fill(dram.data() + destination_pa - kDramBase + (destination_va & 0xfff),
              dram.data() + destination_pa - kDramBase + 0x1000, 0xa5);
    std::fill(dram.data() + destination_next_pa - kDramBase,
              dram.data() + destination_next_pa - kDramBase + cross_bytes,
              0xa5);

    DAYAN::exec_unit_t parent;
    DAYAN::mmu_t mmu(dram.data(), dram.size());
    mmu.set_fixmap(true, 0, 0);
    mmu.set_ptw_base(root);
    test_lsu_t lsu(&parent, uram.data(), dram.data(), nullptr, 0, nullptr,
                   nullptr, &mmu, dram.size(), kDramBase, 4);
    auto control = make_control();

    assert(lsu.execute_dma_once(&control, source_va, destination_va,
                                true, true));
    assert(std::equal(first.begin(), first.begin() + cross_bytes,
                      dram.data() + destination_pa - kDramBase +
                          (destination_va & 0xfff)));
    assert(std::equal(second.begin(), second.begin() + cross_bytes,
                      dram.data() + destination_next_pa - kDramBase));

    // A write whose second page is read-only must fault instead of silently
    // reusing the first page's translation.
    assert(!lsu.execute_dma_once(&control, source_va, perm_destination_va,
                                 true, true));
    // A read whose second page is unmapped must fault as well.
    assert(!lsu.execute_dma_once(&control, unmapped_source_va,
                                 destination_va, true, true));
}

void test_mmu_translation_before_tensor_window(
    std::vector<uint8_t>& dram) {
    std::array<std::vector<uint8_t>, 4> uram = {
        std::vector<uint8_t>(kTensorUramSize),
        std::vector<uint8_t>(kTensorUramSize),
        std::vector<uint8_t>(kTensorUramSize),
        std::vector<uint8_t>(kTensorUramSize),
    };
    std::array<uint8_t*, 4> windows = {
        uram[0].data(), uram[1].data(), uram[2].data(), uram[3].data(),
    };

    std::fill(dram.begin(), dram.end(), 0);
    constexpr uint64_t root = kDramBase;
    constexpr uint64_t l1 = kDramBase + 0x1000;
    constexpr uint64_t l2 = kDramBase + 0x2000;
    // The source VA is above the 2 GiB DRAM base and numerically inside
    // the shared DRAM window, but Sv39 maps it to core 3's remote tensor
    // URAM window: the transfer can only land there if the window lookup
    // runs on the translated PA.  The destination stays an ordinary
    // shared DRAM page.
    constexpr uint64_t source_va = 0x0000000080401000ull;
    constexpr uint64_t destination_va = 0x0000000080402000ull;
    constexpr uint64_t source_pa =
        kTensorUramBase + 3 * kTensorUramStride + 0x1000;
    constexpr uint64_t destination_pa = kDramBase + 0xa000;
    constexpr uint64_t pte_v = 1ull << 0;
    constexpr uint64_t pte_r = 1ull << 1;
    constexpr uint64_t pte_w = 1ull << 2;

    auto vpn = [](uint64_t va) {
        return (va >> 12) & ((uint64_t{1} << 27) - 1);
    };
    auto pte = [](uint64_t pa, uint64_t flags) {
        return ((pa >> 12) << 10) | flags;
    };
    auto pte_address = [](uint64_t table, uint64_t index) {
        return table + index * sizeof(uint64_t);
    };
    const uint64_t source_vpn = vpn(source_va);
    const uint64_t destination_vpn = vpn(destination_va);
    // The two VAs share the same root and level-1 entries.
    store_dram_u64(dram, pte_address(root, (source_vpn >> 18) & 0x1ff),
                   pte(l1, pte_v));
    store_dram_u64(dram, pte_address(l1, (source_vpn >> 9) & 0x1ff),
                   pte(l2, pte_v));
    store_dram_u64(dram, pte_address(l2, source_vpn & 0x1ff),
                   pte(source_pa, pte_v | pte_r | pte_w));
    store_dram_u64(dram, pte_address(l2, destination_vpn & 0x1ff),
                   pte(destination_pa, pte_v | pte_r | pte_w));

    const auto source = pattern(97);
    std::copy(source.begin(), source.end(), uram[3].data() + 0x1000);
    std::fill(dram.data() + destination_pa - kDramBase,
              dram.data() + destination_pa - kDramBase + kRowBytes, 0xa5);

    DAYAN::exec_unit_t parent;
    DAYAN::mmu_t mmu(dram.data(), dram.size());
    mmu.set_fixmap(true, 0, 0);
    mmu.set_ptw_base(root);
    test_lsu_t lsu(&parent, uram[0].data(), dram.data(), windows.data(),
                   windows.size(), nullptr, nullptr, &mmu, dram.size(),
                   kDramBase, 4);
    auto control = make_control();
    assert(lsu.execute_dma_once(&control, source_va, destination_va,
                                true, true));

    assert(std::equal(source.begin(), source.end(),
                      dram.data() + destination_pa - kDramBase));
    assert(std::equal(source.begin(), source.end(),
                      uram[3].data() + 0x1000));
}

void test_context_uram_interrupts(std::vector<uint8_t>& uram,
                                  std::vector<uint8_t>& dram) {
    DAYAN::exec_unit_t parent;
    DAYAN::tensor_context_t context;
    DAYAN::interrupt_controller_t interrupts;
    test_lsu_t lsu(&parent, uram.data(), dram.data(), nullptr, 0,
                   &context, &interrupts);
    auto snapshot = context.snapshot();
    snapshot.uram_range = DAYAN::uram_range_t{4, 5};
    lsu.set_active_context(snapshot);

    constexpr uint64_t begin = 4 * DAYAN::URAM_CONTEXT_GRANULE;
    constexpr uint64_t end = 5 * DAYAN::URAM_CONTEXT_GRANULE;
    assert(lsu.check_uram_access(
        begin, 32,
        DAYAN::interrupt_source_t::neodma_uram_read_range,
        DAYAN::interrupt_source_t::neodma_uram_read_align));
    assert(!lsu.check_uram_access(
        begin - 32, 32,
        DAYAN::interrupt_source_t::neodma_uram_read_range,
        DAYAN::interrupt_source_t::neodma_uram_read_align));
    assert(!lsu.check_uram_access(
        begin + 1, 32,
        DAYAN::interrupt_source_t::neodma_uram_read_range,
        DAYAN::interrupt_source_t::neodma_uram_read_align));
    assert(!lsu.check_uram_access(
        end - 32, 64,
        DAYAN::interrupt_source_t::neodma_uram_read_range,
        DAYAN::interrupt_source_t::neodma_uram_read_align));

    const uint64_t swizzle = pack_swizzle(8, 1, 5);
    assert(lsu.check_uram_access(
        begin, 32,
        DAYAN::interrupt_source_t::neodma_uram_read_range,
        DAYAN::interrupt_source_t::neodma_uram_read_align,
        32, swizzle));
    assert(!lsu.check_uram_access(
        begin - 32, 32,
        DAYAN::interrupt_source_t::neodma_uram_read_range,
        DAYAN::interrupt_source_t::neodma_uram_read_align,
        32, swizzle));

    assert(interrupts.count() == 4);
    assert((interrupts.read(1) >> 32) == static_cast<uint16_t>(
        DAYAN::interrupt_source_t::neodma_uram_read_range));
    assert(static_cast<uint32_t>(interrupts.read(3)) == end);
    assert(static_cast<uint32_t>(interrupts.read(4)) == begin - 9 * 32);
}

} // namespace

int main() {
    std::vector<uint8_t> uram(kUramSize);
    std::vector<uint8_t> dram(kDramSize);
    DAYAN::exec_unit_t parent;
    test_lsu_t lsu(&parent, uram.data(), dram.data());

    test_decoder();
    test_swizzle_address_bits();
    test_unaligned_partial_copy(lsu, uram, dram);
    test_tail_store_sizes(lsu, uram, dram);
    test_gather_scatter_all_directions(lsu, uram, dram);
    test_unaligned_uram_swizzle(uram, dram);
    test_block_rows_dominate_padding(lsu, uram, dram);
    test_aligned_transpose(lsu, uram, dram);
    test_tensor_uram_windows(dram);
    test_mmu_dram_translation(uram, dram);
    test_mmu_page_crossing_translation(uram, dram);
    test_mmu_translation_before_tensor_window(dram);
    test_context_uram_interrupts(uram, dram);
    std::cout << "LSU DMA tests passed\n";
    return 0;
}
