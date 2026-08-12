#include "lsu.hpp"
#include "rvt_decoder.hpp"

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

class test_lsu_t : public RVTNG::lsu_t {
public:
    using RVTNG::lsu_t::execute_dma;
    using RVTNG::lsu_t::execute_dma_once;
    using RVTNG::lsu_t::lsu_t;
};

RVTNG::lsu_tcsr_t make_control() {
    RVTNG::lsu_tcsr_t control{};
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

void test_decoder() {
    constexpr uint32_t base = 0x0800000b;
    assert(IS_RVT_LSU_DMA(base));
    assert(IS_RVT_LSU_DMA(base | (1u << 23)));
    assert(IS_RVT_LSU_DMA(base | (1u << 14)));
    assert(IS_RVT_LSU_DMA(base | (1u << 23) | (1u << 14)));
    assert(RVT_LSU_DMA_GATHER(base | (1u << 23)) == 1);
    assert(RVT_LSU_DMA_SCATTER(base | (1u << 14)) == 1);
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

void test_unaligned_uram_swizzle(test_lsu_t& lsu, std::vector<uint8_t>& uram,
                                 std::vector<uint8_t>& dram) {
    auto store_control = make_control();
    store_control.uramWriteSwizzleMBS = pack_swizzle(5, 3, 5);
    const uint64_t dram_src = kDramBase + 0x3005;
    const uint64_t uram_addr = 0x4003;
    const uint64_t dram_dst = kDramBase + 0x7009;
    const auto source = pattern(19);
    std::copy(source.begin(), source.end(), memory_ptr(uram, dram, true, dram_src));

    assert(lsu.execute_dma_once(&store_control, dram_src, uram_addr, true, false));
    auto load_control = make_control();
    load_control.uramReadSwizzleMBS = store_control.uramWriteSwizzleMBS;
    assert(lsu.execute_dma_once(&load_control, uram_addr, dram_dst, false, true));
    assert(std::equal(source.begin(), source.end(), memory_ptr(uram, dram, true, dram_dst)));
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
    RVTNG::exec_unit_t parent;
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

} // namespace

int main() {
    std::vector<uint8_t> uram(kUramSize);
    std::vector<uint8_t> dram(kDramSize);
    RVTNG::exec_unit_t parent;
    test_lsu_t lsu(&parent, uram.data(), dram.data());

    test_decoder();
    test_unaligned_partial_copy(lsu, uram, dram);
    test_tail_store_sizes(lsu, uram, dram);
    test_gather_scatter_all_directions(lsu, uram, dram);
    test_unaligned_uram_swizzle(lsu, uram, dram);
    test_block_rows_dominate_padding(lsu, uram, dram);
    test_aligned_transpose(lsu, uram, dram);
    test_tensor_uram_windows(dram);
    std::cout << "LSU DMA tests passed\n";
    return 0;
}
