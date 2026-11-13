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

#include "interrupt.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace DAYAN {
namespace {

constexpr std::array<std::string_view, 35> kInterruptSourceNames = {
    "neomat_uram_a_read_range",
    "neomat_uram_a_read_align",
    "neomat_uram_b_read_range",
    "neomat_uram_b_read_align",
    "neomat_uram_scale_read_range",
    "neomat_uram_scale_read_align",
    "neomat_uram_c_read_range",
    "neomat_uram_c_read_align",
    "neomat_uram_c_write_range",
    "neomat_uram_c_write_align",
    "neomat_tcsr_access_error",
    "neomat_decode_error",
    "neodma_tcsr_access_error",
    "neodma_uram_table_range",
    "neodma_uram_table_align",
    "neodma_uram_read_range",
    "neodma_uram_read_align",
    "neodma_uram_write_range",
    "neodma_uram_write_align",
    "neodma_decode_error",
    "flexvec_decode_error",
    "flexvec_uram_group_0_range",
    "flexvec_uram_group_0_align",
    "flexvec_uram_group_1_range",
    "flexvec_uram_group_1_align",
    "neovec_uram_lda_range",
    "neovec_uram_lda_align",
    "neovec_uram_ldb_range",
    "neovec_uram_ldb_align",
    "neovec_uram_store_range",
    "neovec_uram_store_align",
    "neovec_tcsr_access_error",
    "neovec_decode_error",
    "instruction_decode_error",
    "neomat_datatype_error",
};

constexpr uint64_t kCountMask = (uint64_t{1} << 48) - 1;

} // namespace

std::string_view interrupt_source_name(interrupt_source_t source)
{
    // Resource 使用默认 RTL 的编号，旧单元的名称仍按原有连续编号表解析。
    if(source == interrupt_source_t::resource_scheduler_error) {
        return "resource_scheduler_error";
    }
    const size_t index = static_cast<size_t>(source);
    return index < kInterruptSourceNames.size()
        ? kInterruptSourceNames[index] : std::string_view{"unknown"};
}

interrupt_controller_t::interrupt_controller_t(level_callback_t callback)
    : level_callback_(std::move(callback))
{
}

void interrupt_controller_t::set_level_callback(level_callback_t callback)
{
    std::lock_guard<std::mutex> callback_lock(callback_mutex_);
    level_callback_t installed;
    bool level;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        level_callback_ = std::move(callback);
        installed = level_callback_;
        level = pending_;
    }
    if(installed) {
        installed(level);
    }
}

uint64_t interrupt_controller_t::make_record(interrupt_source_t source,
                                             uint32_t info)
{
    return (uint64_t{static_cast<uint16_t>(source)} << 32) | info;
}

void interrupt_controller_t::raise(interrupt_source_t source, uint32_t info)
{
    std::lock_guard<std::mutex> callback_lock(callback_mutex_);
    level_callback_t callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const uint64_t record = make_record(source, info);
        interrupt_count_ = (interrupt_count_ + 1) & kCountMask;

        if(first_count_ < INTERRUPT_RECORD_COUNT) {
            first_[first_count_++] = record;
        }
        recent_[recent_write_ptr_] = record;
        recent_write_ptr_ = (recent_write_ptr_ + 1) % INTERRUPT_RECORD_COUNT;
        recent_count_ = std::min<uint8_t>(recent_count_ + 1,
                                          INTERRUPT_RECORD_COUNT);

        if(!pending_) {
            pending_ = true;
            callback = level_callback_;
        }
    }
    if(callback) {
        callback(true);
    }
}

uint64_t interrupt_controller_t::read(uint16_t address) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    if(address == 0) {
        return uint64_t{pending_}
            | (uint64_t{first_count_} << 4)
            | (uint64_t{recent_count_} << 9)
            | (uint64_t{recent_write_ptr_} << 14);
    }
    if(address >= 1 && address <= INTERRUPT_RECORD_COUNT) {
        return first_[address - 1];
    }
    if(address >= 17 && address <= 32) {
        const size_t logical = address - 17;
        const size_t oldest = recent_count_ == INTERRUPT_RECORD_COUNT
            ? recent_write_ptr_ : 0;
        return recent_[(oldest + logical) % INTERRUPT_RECORD_COUNT];
    }
    if(address == 33) {
        return interrupt_count_;
    }
    return 0;
}

void interrupt_controller_t::write(uint16_t address, uint64_t value)
{
    if(address != 0 || (value & 1) == 0) {
        return;
    }

    std::lock_guard<std::mutex> callback_lock(callback_mutex_);
    level_callback_t callback;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if(pending_) {
            pending_ = false;
            callback = level_callback_;
        }
    }
    if(callback) {
        callback(false);
    }
}

bool interrupt_controller_t::pending() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_;
}

uint64_t interrupt_controller_t::count() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return interrupt_count_;
}

} // namespace DAYAN
