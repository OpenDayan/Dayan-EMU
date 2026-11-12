#pragma once

#include "interrupt.hpp"
#include "tensor_context.hpp"
#include "thread_safe_queue.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

namespace DAYAN {

constexpr uint16_t RESOURCE_PRODUCE_BASE = 0x020;
constexpr uint8_t RESOURCE_NO_OUTPUT = 0xff;
constexpr uint16_t RESOURCE_WAIT_FREE = 0x030;
constexpr uint16_t RESOURCE_WAIT_READY = 0x031;
constexpr uint16_t RESOURCE_WAIT_QOS = 0x032;
constexpr uint16_t RESOURCE_WAIT = 0x033;
constexpr uint16_t RESOURCE_CONSUME_READY = 0x034;
constexpr uint16_t LOCAL_RESOURCE_INIT = 0x035;
constexpr uint16_t RESOURCE_TOKEN_EXT = 0x036;
constexpr uint16_t RESOURCE_WAIT_PHASE = 0x037;
constexpr size_t RESOURCE_WAIT_FIFO_CAPACITY = 64;
constexpr size_t RESOURCE_ISSUE_QUEUE_CAPACITY = 16;

// 完整地址译码与 RTL 一致；Resource Produce 的读操作由核级 CSR 路径直接处理。
inline bool is_unit_scoreboard(uint64_t instruction, uint8_t unit)
{
    const uint16_t address = (instruction >> 13) & 0x3ff;
    return (instruction & 0x7f) == 0x0b &&
        ((instruction >> 23) & 0x1ff) == 0 &&
        (address == unit ||
         (address == RESOURCE_PRODUCE_BASE + unit &&
          ((instruction >> 12) & 1) == 0));
}

enum class resource_role_t : uint8_t {
    normal = 0,
    producer = 1,
    consumer = 2,
};

struct scoreboard_entry_t {
    uint64_t token = 0;
    uint64_t extension = 0;
    bool resource = false;
    uint32_t local_resources = 0;
    uint32_t shared_resources = 0;
};

enum class resource_state_t : uint8_t {
    free,
    busy,
    ready,
};

struct resource_entry_t {
    resource_state_t state = resource_state_t::free;
    uint8_t consumers = 0;
    // Lifecycle generation: stable from FREE through BUSY/READY, toggled on
    // the final READY -> FREE consumption.
    bool phase = false;
};

class resource_scheduler_t;

class resource_cluster_t {
public:
    void add(resource_scheduler_t* scheduler);
    void remove(resource_scheduler_t* scheduler);
    bool clear(uint32_t mask);

private:
    friend class resource_scheduler_t;

    void schedule_all_locked();

    std::mutex mutex_;
    std::condition_variable wait_space_;
    std::array<resource_entry_t, RESOURCE_TABLE_ENTRIES> shared_{};
    std::vector<resource_scheduler_t*> schedulers_;
};

class resource_scheduler_t {
public:
    resource_scheduler_t(resource_cluster_t& cluster,
                         MPMCQueue<scoreboard_entry_t, 32>& output,
                         interrupt_controller_t* interrupts = nullptr);
    ~resource_scheduler_t();

    bool handles(uint16_t address) const;
    uint64_t read(uint16_t address);
    bool write(uint16_t address, uint64_t value,
               const tensor_context_snapshot_t& context);
    void complete_produce(uint8_t unit);
    void output_popped(const scoreboard_entry_t& entry);

private:
    friend class resource_cluster_t;

    struct mapped_mask_t {
        uint32_t local = 0;
        uint32_t shared = 0;
    };

    struct wait_entry_t {
        mapped_mask_t wait_free;
        mapped_mask_t wait_ready;
        mapped_mask_t phase;
        bool election_shared = false;
        uint8_t election_index = 0;
        bool election_phase = false;
        int8_t qos = 0;
        uint64_t age = 0;
        uint64_t token = 0;
    };

    struct produce_entry_t {
        mapped_mask_t consume_ready;
        bool has_output = true;
        bool output_shared = false;
        uint8_t output_index = 0;
        uint8_t consumers = 0;
    };

    bool map_mask_locked(uint64_t logical,
                         const tensor_context_snapshot_t& context,
                         mapped_mask_t& mapped) const;
    bool map_resource_locked(uint8_t logical,
                             const tensor_context_snapshot_t& context,
                             bool& shared, uint8_t& physical) const;
    bool register_wait_locked(uint64_t token,
                              const tensor_context_snapshot_t& context,
                              std::unique_lock<std::mutex>& lock);
    bool register_produce_locked(uint8_t unit, uint64_t value,
                                 const tensor_context_snapshot_t& context);
    bool wait_ready_locked(const wait_entry_t& wait,
                           resource_role_t& role) const;
    bool reserve_wait_locked(const wait_entry_t& wait,
                             resource_role_t role);
    bool complete_produce_locked(const produce_entry_t& produce);
    void refill_issue_locked();
    void schedule_locked();
    void raise_locked(uint16_t address) const;
    bool in_use_locked(uint32_t mask, bool shared) const;
    resource_entry_t& entry_locked(bool shared, uint8_t index);
    const resource_entry_t& entry_locked(bool shared, uint8_t index) const;

    resource_cluster_t& cluster_;
    MPMCQueue<scoreboard_entry_t, 32>& output_;
    interrupt_controller_t* interrupts_;
    std::array<resource_entry_t, RESOURCE_TABLE_ENTRIES> local_{};
    std::array<std::deque<produce_entry_t>, 16> pending_produce_{};
    std::deque<wait_entry_t> wait_fifo_;
    std::vector<wait_entry_t> issue_queue_;
    uint64_t wait_free_ = 0;
    uint64_t wait_ready_ = 0;
    uint64_t wait_qos_ = 0;
    uint64_t wait_phase_ = 0;
    uint64_t consume_ready_ = 0;
    uint64_t token_ext_ = 0;
    uint64_t next_age_ = 0;
    size_t pending_tokens_ = 0;
    std::array<size_t, RESOURCE_TABLE_ENTRIES> local_tokens_{};
    std::array<size_t, RESOURCE_TABLE_ENTRIES> shared_tokens_{};
};

} // namespace DAYAN
