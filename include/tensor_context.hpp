#pragma once

#include "rvtng-intf.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace RVTNG {

constexpr uint64_t TENSOR_CONTEXT_ADDRESS = 0x010;
constexpr uint64_t TENSOR_CONTEXT_MASK = (uint64_t{1} << 56) - 1;
constexpr uint64_t URAM_CONTEXT_GRANULE = 32 * 1024;
constexpr uint8_t LOGICAL_TCSR_GROUPS = 8;

enum class tensor_unit_t : uint8_t {
    lsu,
    mat,
    vec,
    macro_unit,
};

struct tcsr_map_t {
    uint8_t count = 0;
    uint8_t base = 0;
};

struct uram_range_t {
    uint8_t lo = 0;
    uint8_t hi = 0;

    uint64_t begin() const
    {
        return uint64_t{lo} * URAM_CONTEXT_GRANULE;
    }

    uint64_t end() const
    {
        return uint64_t{hi} * URAM_CONTEXT_GRANULE;
    }

    bool contains(uint64_t address, size_t size = 1) const
    {
        if(size == 0) {
            return address >= begin() && address <= end();
        }
        if(address < begin() || address >= end()) {
            return false;
        }
        const uint64_t bytes = static_cast<uint64_t>(size);
        return bytes <= end() - address;
    }
};

struct tensor_context_snapshot_t {
    uram_range_t uram_range;
    tcsr_map_t lsu_tmap;
    tcsr_map_t mat_tmap;
    tcsr_map_t vec_tmap;
    tcsr_map_t macro_tmap;

    static tensor_context_snapshot_t decode(uint64_t value)
    {
        tensor_context_snapshot_t context;
        value &= TENSOR_CONTEXT_MASK;
        context.uram_range.lo = value & 0xff;
        context.uram_range.hi = (value >> 8) & 0xff;
        context.lsu_tmap.count = (value >> 16) & 0x0f;
        context.lsu_tmap.base = (value >> 20) & 0x3f;
        context.mat_tmap.count = (value >> 26) & 0x0f;
        context.mat_tmap.base = (value >> 30) & 0x3f;
        context.vec_tmap.count = (value >> 36) & 0x0f;
        context.vec_tmap.base = (value >> 40) & 0x3f;
        context.macro_tmap.count = (value >> 46) & 0x0f;
        context.macro_tmap.base = (value >> 50) & 0x3f;
        return context;
    }

    static tensor_context_snapshot_t defaults()
    {
        tensor_context_snapshot_t context;
        context.uram_range = uram_range_t{0, 48};
        context.lsu_tmap = tcsr_map_t{8, 0};
        context.mat_tmap = tcsr_map_t{8, 0};
        context.vec_tmap = tcsr_map_t{1, 0};
        context.macro_tmap = tcsr_map_t{8, 0};
        return context;
    }

    uint64_t encode() const
    {
        return uint64_t{uram_range.lo}
            | (uint64_t{uram_range.hi} << 8)
            | (static_cast<uint64_t>(lsu_tmap.count & 0x0f) << 16)
            | (static_cast<uint64_t>(lsu_tmap.base & 0x3f) << 20)
            | (static_cast<uint64_t>(mat_tmap.count & 0x0f) << 26)
            | (static_cast<uint64_t>(mat_tmap.base & 0x3f) << 30)
            | (static_cast<uint64_t>(vec_tmap.count & 0x0f) << 36)
            | (static_cast<uint64_t>(vec_tmap.base & 0x3f) << 40)
            | (static_cast<uint64_t>(macro_tmap.count & 0x0f) << 46)
            | (static_cast<uint64_t>(macro_tmap.base & 0x3f) << 50);
    }

    const tcsr_map_t& tmap(tensor_unit_t unit) const
    {
        switch(unit) {
        case tensor_unit_t::lsu:
            return lsu_tmap;
        case tensor_unit_t::mat:
            return mat_tmap;
        case tensor_unit_t::vec:
            return vec_tmap;
        case tensor_unit_t::macro_unit:
            return macro_tmap;
        }
        return lsu_tmap;
    }

    std::optional<size_t> physical_group(tensor_unit_t unit,
                                         uint8_t logical_group,
                                         size_t physical_banks) const
    {
        const tcsr_map_t& map = tmap(unit);
        if(logical_group >= LOGICAL_TCSR_GROUPS || logical_group >= map.count ||
           physical_banks == 0) {
            return std::nullopt;
        }
        return (static_cast<size_t>(map.base) + logical_group) % physical_banks;
    }
};

class tensor_context_t {
public:
    tensor_context_t()
        : value_(tensor_context_snapshot_t::defaults().encode())
    {
    }

    tensor_context_snapshot_t snapshot() const
    {
        return tensor_context_snapshot_t::decode(
            value_.load(std::memory_order_acquire));
    }

    void write(uint64_t value)
    {
        value_.store(value & TENSOR_CONTEXT_MASK, std::memory_order_release);
    }

    uint64_t read() const
    {
        return value_.load(std::memory_order_acquire);
    }

private:
    std::atomic<uint64_t> value_;
};

struct queued_tensor_inst_t {
    rvt_pack_t inst{};
    tensor_context_snapshot_t context = tensor_context_snapshot_t::defaults();
    bool context_write = false;
};

inline bool is_context_access(uint64_t inst)
{
    return (inst & 0x7f) == 0x0b && ((inst >> 27) & 0x1f) == 0 &&
        ((inst >> 23) & 0x0f) == 0 &&
        ((inst >> 13) & 0x3ff) == TENSOR_CONTEXT_ADDRESS;
}

inline bool is_context_read(uint64_t inst)
{
    return is_context_access(inst) && ((inst >> 12) & 1) != 0;
}

inline bool is_context_write(uint64_t inst)
{
    return is_context_access(inst) && ((inst >> 12) & 1) == 0;
}

} // namespace RVTNG
