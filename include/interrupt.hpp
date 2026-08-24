#pragma once

#include "tensor_context.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string_view>

namespace RVTNG {

enum class interrupt_source_t : uint16_t {
    neomat_uram_a_read_range = 0,
    neomat_uram_a_read_align = 1,
    neomat_uram_b_read_range = 2,
    neomat_uram_b_read_align = 3,
    neomat_uram_scale_read_range = 4,
    neomat_uram_scale_read_align = 5,
    neomat_uram_c_read_range = 6,
    neomat_uram_c_read_align = 7,
    neomat_uram_c_write_range = 8,
    neomat_uram_c_write_align = 9,
    neomat_tcsr_access_error = 10,
    neomat_decode_error = 11,
    neodma_tcsr_access_error = 12,
    neodma_uram_table_range = 13,
    neodma_uram_table_align = 14,
    neodma_uram_read_range = 15,
    neodma_uram_read_align = 16,
    neodma_uram_write_range = 17,
    neodma_uram_write_align = 18,
    neodma_decode_error = 19,
    flexvec_decode_error = 20,
    flexvec_uram_group_0_range = 21,
    flexvec_uram_group_0_align = 22,
    flexvec_uram_group_1_range = 23,
    flexvec_uram_group_1_align = 24,
    neovec_uram_lda_range = 25,
    neovec_uram_lda_align = 26,
    neovec_uram_ldb_range = 27,
    neovec_uram_ldb_align = 28,
    neovec_uram_store_range = 29,
    neovec_uram_store_align = 30,
    neovec_tcsr_access_error = 31,
    neovec_decode_error = 32,
    instruction_decode_error = 33,
};

constexpr uint8_t INTERRUPT_TCSR_UNIT = 5;
constexpr size_t INTERRUPT_RECORD_COUNT = 16;

std::string_view interrupt_source_name(interrupt_source_t source);

class interrupt_controller_t {
public:
    using level_callback_t = std::function<void(bool)>;

    explicit interrupt_controller_t(level_callback_t callback = {});

    void set_level_callback(level_callback_t callback);
    void raise(interrupt_source_t source, uint32_t info);
    uint64_t read(uint16_t address) const;
    void write(uint16_t address, uint64_t value);
    bool pending() const;
    uint64_t count() const;

private:
    static uint64_t make_record(interrupt_source_t source, uint32_t info);

    mutable std::mutex mutex_;
    mutable std::mutex callback_mutex_;
    std::array<uint64_t, INTERRUPT_RECORD_COUNT> first_{};
    std::array<uint64_t, INTERRUPT_RECORD_COUNT> recent_{};
    uint8_t first_count_ = 0;
    uint8_t recent_count_ = 0;
    uint8_t recent_write_ptr_ = 0;
    bool pending_ = false;
    uint64_t interrupt_count_ = 0;
    level_callback_t level_callback_;
};

inline bool is_interrupt_tcsr(uint64_t inst)
{
    return (inst & 0x7f) == 0x0b && ((inst >> 27) & 0x1f) == 0 &&
        ((inst >> 23) & 0x0f) == INTERRUPT_TCSR_UNIT;
}

} // namespace RVTNG
