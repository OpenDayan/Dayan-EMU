#include "interrupt.hpp"
#include "tensor_context.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdint>
#include <vector>

namespace {

void test_context_encoding()
{
    const auto defaults = RVTNG::tensor_context_snapshot_t::defaults();
    const uint64_t expected =
        (uint64_t{48} << 8) |
        (uint64_t{8} << 16) |
        (uint64_t{8} << 26) |
        (uint64_t{1} << 36) |
        (uint64_t{8} << 46);
    assert(defaults.encode() == expected);

    auto context = defaults;
    context.uram_range = RVTNG::uram_range_t{3, 7};
    context.lsu_tmap = RVTNG::tcsr_map_t{2, 5};
    context.mat_tmap = RVTNG::tcsr_map_t{4, 12};
    context.vec_tmap = RVTNG::tcsr_map_t{1, 3};
    context.macro_tmap = RVTNG::tcsr_map_t{3, 14};

    auto decoded = RVTNG::tensor_context_snapshot_t::decode(
        context.encode() | (uint64_t{0xff} << 56));
    assert(decoded.encode() == context.encode());
    assert(decoded.uram_range.begin() == 3 * RVTNG::URAM_CONTEXT_GRANULE);
    assert(decoded.uram_range.end() == 7 * RVTNG::URAM_CONTEXT_GRANULE);
    assert(decoded.uram_range.contains(decoded.uram_range.begin(), 32));
    assert(!decoded.uram_range.contains(decoded.uram_range.end(), 1));

    assert(decoded.physical_group(RVTNG::tensor_unit_t::lsu, 0, 16) == 5);
    assert(decoded.physical_group(RVTNG::tensor_unit_t::lsu, 1, 16) == 6);
    assert(!decoded.physical_group(RVTNG::tensor_unit_t::lsu, 2, 16));
    assert(decoded.physical_group(RVTNG::tensor_unit_t::mat, 3, 16) == 15);
    assert(decoded.physical_group(RVTNG::tensor_unit_t::macro_unit, 2, 16) == 0);

    decoded.local_resources = RVTNG::resource_map_t{7, 3};
    decoded.shared_resources = RVTNG::resource_map_t{9, 11};
    RVTNG::tensor_context_snapshot_t resource_decoded = defaults;
    resource_decoded.decode_resource_context(decoded.encode_resource_context());
    assert(resource_decoded.local_resources.count == 7);
    assert(resource_decoded.local_resources.base == 3);
    assert(resource_decoded.shared_resources.count == 9);
    assert(resource_decoded.shared_resources.base == 11);
    assert(resource_decoded.local_resources.physical(6) == 9);
    assert(!resource_decoded.local_resources.physical(7));
}

void test_interrupt_records()
{
    std::vector<bool> levels;
    RVTNG::interrupt_controller_t interrupts(
        [&](bool level) { levels.push_back(level); });

    interrupts.raise(RVTNG::interrupt_source_t::neodma_uram_read_range,
                     0x12345678);
    assert(interrupts.pending());
    assert(interrupts.count() == 1);
    assert(levels.size() == 1 && levels.back());
    assert((interrupts.read(0) & 1) == 1);
    assert(((interrupts.read(0) >> 4) & 0x1f) == 1);
    assert(((interrupts.read(0) >> 9) & 0x1f) == 1);
    assert(interrupts.read(1) ==
           (uint64_t{static_cast<uint16_t>(
                RVTNG::interrupt_source_t::neodma_uram_read_range)} << 32 |
            0x12345678));

    for(uint32_t info = 1; info <= 20; ++info) {
        interrupts.raise(RVTNG::interrupt_source_t::instruction_decode_error,
                         info);
    }
    assert(interrupts.count() == 21);
    assert(((interrupts.read(0) >> 4) & 0x1f) == 16);
    assert(((interrupts.read(0) >> 9) & 0x1f) == 16);
    assert((interrupts.read(17) & 0xffffffffu) == 5);
    assert((interrupts.read(32) & 0xffffffffu) == 20);

    interrupts.write(1, 1);
    assert(interrupts.pending());
    interrupts.write(0, 0);
    assert(interrupts.pending());
    interrupts.write(0, 1);
    assert(!interrupts.pending());
    assert(levels.size() == 2 && !levels.back());
    assert(interrupts.count() == 21);
}

void test_decoder_helpers()
{
    const uint64_t context_read =
        (RVTNG::TENSOR_CONTEXT_ADDRESS << 13) | (uint64_t{1} << 12) | 0x0b;
    const uint64_t context_write = context_read & ~(uint64_t{1} << 12);
    const uint64_t resource_context_read =
        (RVTNG::TENSOR_RESOURCE_CONTEXT_ADDRESS << 13) |
        (uint64_t{1} << 12) | 0x0b;
    const uint64_t interrupt_read =
        (uint64_t{RVTNG::INTERRUPT_TCSR_UNIT} << 23) |
        (uint64_t{33} << 13) | (uint64_t{1} << 12) | 0x0b;
    assert(RVTNG::is_context_read(context_read));
    assert(RVTNG::is_context_write(context_write));
    assert(RVTNG::is_context_read(resource_context_read));
    assert(RVTNG::is_interrupt_tcsr(interrupt_read));
    assert(RVTNG::interrupt_source_name(
        RVTNG::interrupt_source_t::instruction_decode_error) ==
        "instruction_decode_error");
    assert(RVTNG::interrupt_source_name(
        RVTNG::interrupt_source_t::neomat_datatype_error) ==
        "neomat_datatype_error");
    assert(RVTNG::interrupt_source_name(
        RVTNG::interrupt_source_t::resource_scheduler_error) ==
        "resource_scheduler_error");
}

void test_queued_context_snapshot()
{
    RVTNG::tensor_context_t context;
    RVTNG::queued_tensor_inst_t before;
    before.context = context.snapshot();

    auto updated = before.context;
    updated.uram_range = RVTNG::uram_range_t{8, 16};
    updated.lsu_tmap = RVTNG::tcsr_map_t{2, 7};
    context.write(updated.encode());
    updated.local_resources = RVTNG::resource_map_t{4, 2};
    updated.shared_resources = RVTNG::resource_map_t{5, 7};
    context.write(RVTNG::TENSOR_RESOURCE_CONTEXT_ADDRESS,
                  updated.encode_resource_context());

    RVTNG::queued_tensor_inst_t context_write;
    context_write.context = context.snapshot();
    context_write.context_write = true;
    RVTNG::queued_tensor_inst_t after;
    after.context = context.snapshot();

    assert(before.context.uram_range.lo == 0);
    assert(before.context.uram_range.hi == 48);
    assert(before.context.lsu_tmap.base == 0);
    assert(context_write.context_write);
    assert(context_write.context.encode() == updated.encode());
    assert(after.context.encode() == updated.encode());
    assert(after.context.local_resources.count == 4);
    assert(after.context.local_resources.base == 2);
    assert(after.context.shared_resources.count == 5);
    assert(after.context.shared_resources.base == 7);
}

void test_invalid_resource_context_write()
{
    RVTNG::tensor_context_t context;
    const auto address = RVTNG::TENSOR_RESOURCE_CONTEXT_ADDRESS;
    const uint64_t initial = context.read(address);
    assert(initial == 0x200400);
    assert(!context.write(address, 31 | (uint64_t{2} << 5)));
    assert(!context.write(address, (uint64_t{31} << 11) | (uint64_t{2} << 16)));
    assert(!context.write(address, uint64_t{33} << 5));
    assert(context.read(address) == initial);
    assert(context.write(address, 0));
    assert(!context.snapshot().local_resources.physical(0));
}

} // namespace

int main()
{
    test_context_encoding();
    test_interrupt_records();
    test_decoder_helpers();
    test_queued_context_snapshot();
    test_invalid_resource_context_write();
    return 0;
}
