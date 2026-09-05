#include "resource_scheduler.hpp"
#include "rvt_decoder.hpp"

#include <algorithm>
#include <cassert>
#include <limits>

namespace RVTNG {

void resource_cluster_t::add(resource_scheduler_t* scheduler)
{
    std::lock_guard<std::mutex> lock(mutex_);
    schedulers_.push_back(scheduler);
}

void resource_cluster_t::remove(resource_scheduler_t* scheduler)
{
    std::lock_guard<std::mutex> lock(mutex_);
    schedulers_.erase(std::remove(schedulers_.begin(), schedulers_.end(),
                                  scheduler),
                      schedulers_.end());
}

void resource_cluster_t::schedule_all_locked()
{
    for(resource_scheduler_t* scheduler : schedulers_) {
        scheduler->schedule_locked();
    }
}

resource_scheduler_t::resource_scheduler_t(
    resource_cluster_t& cluster, MPMCQueue<scoreboard_entry_t, 32>& output,
    interrupt_controller_t* interrupts)
    : cluster_(cluster), output_(output), interrupts_(interrupts)
{
    cluster_.add(this);
}

resource_scheduler_t::~resource_scheduler_t()
{
    cluster_.remove(this);
}

bool resource_scheduler_t::handles(uint16_t address) const
{
    return (address >= RESOURCE_PRODUCE_BASE &&
            address < RESOURCE_PRODUCE_BASE + 16) ||
           (address >= RESOURCE_WAIT_FREE &&
            address <= RESOURCE_TOKEN_EXT);
}

resource_entry_t& resource_scheduler_t::entry_locked(bool shared,
                                                      uint8_t index)
{
    return shared ? cluster_.shared_[index] : local_[index];
}

const resource_entry_t& resource_scheduler_t::entry_locked(
    bool shared, uint8_t index) const
{
    return shared ? cluster_.shared_[index] : local_[index];
}

void resource_scheduler_t::raise_locked(uint16_t address) const
{
    if(interrupts_) {
        interrupts_->raise(interrupt_source_t::resource_scheduler_error,
                           address);
    }
}

bool resource_scheduler_t::map_resource_locked(
    uint8_t logical, const tensor_context_snapshot_t& context,
    bool& shared, uint8_t& physical) const
{
    shared = logical >= 32;
    const uint8_t offset = shared ? static_cast<uint8_t>(logical - 32)
                                  : logical;
    const std::optional<uint8_t> mapped = shared
        ? context.shared_resources.physical(offset)
        : context.local_resources.physical(offset);
    if(!mapped) {
        return false;
    }
    physical = *mapped;
    return true;
}

bool resource_scheduler_t::map_mask_locked(
    uint64_t logical, const tensor_context_snapshot_t& context,
    mapped_mask_t& mapped) const
{
    mapped = {};
    for(uint8_t resource = 0; resource < 64; ++resource) {
        if((logical & (uint64_t{1} << resource)) == 0) {
            continue;
        }
        bool shared;
        uint8_t physical;
        if(!map_resource_locked(resource, context, shared, physical)) {
            return false;
        }
        uint32_t& mask = shared ? mapped.shared : mapped.local;
        mask |= uint32_t{1} << physical;
    }
    return true;
}

uint64_t resource_scheduler_t::read(uint16_t address)
{
    std::lock_guard<std::mutex> lock(cluster_.mutex_);
    if(address >= RESOURCE_PRODUCE_BASE && address < RESOURCE_PRODUCE_BASE + 16 &&
       (address < RESOURCE_PRODUCE_BASE + RVT_UNIT_LSU ||
        address > RESOURCE_PRODUCE_BASE + RVT_UNIT_MACRO)) {
        raise_locked(address);
    }
    switch(address) {
    case RESOURCE_WAIT_FREE:
        return wait_free_;
    case RESOURCE_WAIT_READY:
        return wait_ready_;
    case RESOURCE_WAIT_QOS:
        return wait_qos_;
    case RESOURCE_CONSUME_READY:
        return consume_ready_;
    case RESOURCE_TOKEN_EXT:
        return token_ext_;
    default:
        return 0;
    }
}

bool resource_scheduler_t::register_wait_locked(
    uint64_t token, const tensor_context_snapshot_t& context,
    std::unique_lock<std::mutex>& lock)
{
    mapped_mask_t mapped_free;
    mapped_mask_t mapped_ready;
    const uint64_t election = wait_free_ & wait_ready_;
    if((election != 0 && (election & (election - 1)) != 0) ||
       !map_mask_locked(wait_free_, context, mapped_free) ||
       !map_mask_locked(wait_ready_, context, mapped_ready)) {
        raise_locked(RESOURCE_WAIT);
        return false;
    }

    while(wait_fifo_.size() + issue_queue_.size() >=
          RESOURCE_WAIT_FIFO_CAPACITY) {
        cluster_.wait_space_.wait(lock);
    }

    wait_entry_t wait;
    wait.wait_free = mapped_free;
    wait.wait_ready = mapped_ready;
    wait.qos = static_cast<int8_t>(wait_qos_ & 7);
    if(wait.qos & 4) {
        wait.qos = static_cast<int8_t>(wait.qos - 8);
    }
    wait.age = next_age_++;
    wait.token = token;
    if(election) {
        const uint8_t logical = static_cast<uint8_t>(__builtin_ctzll(election));
        if(!map_resource_locked(logical, context, wait.election_shared,
                                wait.election_index)) {
            raise_locked(RESOURCE_WAIT);
            return false;
        }
        wait.election_phase = entry_locked(wait.election_shared,
                                            wait.election_index).phase;
    }
    wait_fifo_.push_back(wait);
    wait_free_ = 0;
    wait_ready_ = 0;
    wait_qos_ = 0;
    refill_issue_locked();
    cluster_.schedule_all_locked();
    return true;
}

bool resource_scheduler_t::register_produce_locked(
    uint8_t unit, uint64_t value,
    const tensor_context_snapshot_t& context)
{
    produce_entry_t produce;
    const uint8_t output = static_cast<uint8_t>(value);
    produce.consumers = static_cast<uint8_t>(value >> 8);
    produce.has_output = output != RESOURCE_NO_OUTPUT;
    if(unit < RVT_UNIT_LSU || unit > RVT_UNIT_MACRO || (value >> 16) != 0 ||
       !map_mask_locked(consume_ready_, context, produce.consume_ready) ||
       (!produce.has_output &&
        (produce.consumers != 0 || consume_ready_ == 0)) ||
       (produce.has_output &&
        (!produce.consumers ||
         !map_resource_locked(output, context, produce.output_shared,
                              produce.output_index)))) {
        raise_locked(static_cast<uint16_t>(RESOURCE_PRODUCE_BASE + unit));
        return false;
    }
    if(produce.has_output) {
        const uint32_t output_bit = uint32_t{1} << produce.output_index;
        if((produce.output_shared
                ? produce.consume_ready.shared
                : produce.consume_ready.local) & output_bit) {
            raise_locked(static_cast<uint16_t>(RESOURCE_PRODUCE_BASE + unit));
            return false;
        }
    }
    pending_produce_[unit].push_back(produce);
    consume_ready_ = 0;
    return true;
}

bool resource_scheduler_t::write(
    uint16_t address, uint64_t value,
    const tensor_context_snapshot_t& context)
{
    std::unique_lock<std::mutex> lock(cluster_.mutex_);
    if(address >= RESOURCE_PRODUCE_BASE &&
       address < RESOURCE_PRODUCE_BASE + 16) {
        return register_produce_locked(
            static_cast<uint8_t>(address - RESOURCE_PRODUCE_BASE), value,
            context);
    }
    switch(address) {
    case RESOURCE_WAIT_FREE:
        wait_free_ = value;
        break;
    case RESOURCE_WAIT_READY:
        wait_ready_ = value;
        break;
    case RESOURCE_WAIT_QOS:
        wait_qos_ = value & 7;
        break;
    case RESOURCE_WAIT:
        return register_wait_locked(value, context, lock);
    case RESOURCE_CONSUME_READY:
        consume_ready_ = value;
        break;
    case LOCAL_RESOURCE_INIT: {
        // 与 LocalScheduler 一样检查内部未完成项，其他资源使用者由软件负责排空。
        bool quiescent = wait_fifo_.empty() && issue_queue_.empty() && pending_tokens_ == 0;
        for(const auto& pending : pending_produce_) {
            quiescent = quiescent && pending.empty();
        }
        if(!quiescent) {
            raise_locked(LOCAL_RESOURCE_INIT);
            return false;
        }
        local_.fill(resource_entry_t{});
        break;
    }
    case RESOURCE_TOKEN_EXT:
        raise_locked(RESOURCE_TOKEN_EXT);
        return false;
    default:
        return false;
    }
    return true;
}

bool resource_scheduler_t::wait_ready_locked(
    const wait_entry_t& wait, resource_role_t& role) const
{
    role = resource_role_t::normal;
    if(wait.wait_free.local & wait.wait_ready.local ||
       wait.wait_free.shared & wait.wait_ready.shared) {
        const resource_entry_t& election = entry_locked(
            wait.election_shared, wait.election_index);
        if(election.state == resource_state_t::free &&
           election.phase == wait.election_phase) {
            role = resource_role_t::producer;
        } else if(election.state == resource_state_t::ready &&
                  election.phase != wait.election_phase) {
            role = resource_role_t::consumer;
        } else {
            return false;
        }
    }

    auto test_mask = [&](uint32_t mask, bool shared,
                         resource_state_t state,
                         bool skip_election) {
        while(mask) {
            const uint8_t index = static_cast<uint8_t>(__builtin_ctz(mask));
            mask &= mask - 1;
            if(skip_election && shared == wait.election_shared &&
               index == wait.election_index) {
                continue;
            }
            if(entry_locked(shared, index).state != state) {
                return false;
            }
        }
        return true;
    };
    const bool election = role != resource_role_t::normal;
    return test_mask(wait.wait_free.local, false, resource_state_t::free,
                     election) &&
           test_mask(wait.wait_free.shared, true, resource_state_t::free,
                     election) &&
           test_mask(wait.wait_ready.local, false, resource_state_t::ready,
                     election) &&
           test_mask(wait.wait_ready.shared, true, resource_state_t::ready,
                     election);
}

bool resource_scheduler_t::reserve_wait_locked(
    const wait_entry_t& wait, resource_role_t role)
{
    scoreboard_entry_t output{wait.token, static_cast<uint64_t>(role), true};
    if(!output_.try_enqueue(output)) {
        return false;
    }
    ++pending_tokens_;
    auto reserve_mask = [&](uint32_t mask, bool shared) {
        while(mask) {
            const uint8_t index = static_cast<uint8_t>(__builtin_ctz(mask));
            mask &= mask - 1;
            if(role == resource_role_t::consumer &&
               shared == wait.election_shared &&
               index == wait.election_index) {
                continue;
            }
            resource_entry_t& resource = entry_locked(shared, index);
            resource.state = resource_state_t::busy;
            resource.phase = !resource.phase;
        }
    };
    reserve_mask(wait.wait_free.local, false);
    reserve_mask(wait.wait_free.shared, true);
    return true;
}

void resource_scheduler_t::refill_issue_locked()
{
    while(issue_queue_.size() < RESOURCE_ISSUE_QUEUE_CAPACITY &&
          !wait_fifo_.empty()) {
        issue_queue_.push_back(wait_fifo_.front());
        wait_fifo_.pop_front();
    }
}

void resource_scheduler_t::schedule_locked()
{
    refill_issue_locked();
    size_t selected = issue_queue_.size();
    resource_role_t selected_role = resource_role_t::normal;
    for(size_t index = 0; index < issue_queue_.size(); ++index) {
        resource_role_t role;
        if(!wait_ready_locked(issue_queue_[index], role)) {
            continue;
        }
        if(selected == issue_queue_.size() ||
           issue_queue_[index].qos > issue_queue_[selected].qos ||
           (issue_queue_[index].qos == issue_queue_[selected].qos &&
            issue_queue_[index].age < issue_queue_[selected].age)) {
            selected = index;
            selected_role = role;
        }
    }
    if(selected == issue_queue_.size() ||
       !reserve_wait_locked(issue_queue_[selected], selected_role)) {
        return;
    }
    issue_queue_.erase(issue_queue_.begin() + selected);
    refill_issue_locked();
    cluster_.wait_space_.notify_all();
}

bool resource_scheduler_t::complete_produce_locked(
    const produce_entry_t& produce)
{
    if(produce.has_output) {
        const resource_entry_t& output = entry_locked(
            produce.output_shared, produce.output_index);
        if(output.state != resource_state_t::busy || !produce.consumers) {
            return false;
        }
    }
    auto validate_inputs = [&](uint32_t mask, bool shared) {
        while(mask) {
            const uint8_t index = static_cast<uint8_t>(__builtin_ctz(mask));
            mask &= mask - 1;
            const resource_entry_t& input = entry_locked(shared, index);
            if(input.state != resource_state_t::ready || !input.consumers) {
                return false;
            }
        }
        return true;
    };
    if(!validate_inputs(produce.consume_ready.local, false) ||
       !validate_inputs(produce.consume_ready.shared, true)) {
        return false;
    }
    auto consume_inputs = [&](uint32_t mask, bool shared) {
        while(mask) {
            const uint8_t index = static_cast<uint8_t>(__builtin_ctz(mask));
            mask &= mask - 1;
            resource_entry_t& input = entry_locked(shared, index);
            if(--input.consumers == 0) {
                input.state = resource_state_t::free;
            }
        }
    };
    consume_inputs(produce.consume_ready.local, false);
    consume_inputs(produce.consume_ready.shared, true);
    if(produce.has_output) {
        resource_entry_t& mutable_output = entry_locked(
            produce.output_shared, produce.output_index);
        mutable_output.consumers = produce.consumers;
        mutable_output.state = resource_state_t::ready;
    }
    return true;
}

void resource_scheduler_t::complete_produce(uint8_t unit)
{
    std::lock_guard<std::mutex> lock(cluster_.mutex_);
    if(unit >= pending_produce_.size() || pending_produce_[unit].empty()) {
        raise_locked(static_cast<uint16_t>(RESOURCE_PRODUCE_BASE + unit));
        return;
    }
    const produce_entry_t produce = pending_produce_[unit].front();
    pending_produce_[unit].pop_front();
    if(!complete_produce_locked(produce)) {
        raise_locked(static_cast<uint16_t>(RESOURCE_PRODUCE_BASE + unit));
        return;
    }
    cluster_.schedule_all_locked();
}

void resource_scheduler_t::output_popped(const scoreboard_entry_t& entry)
{
    std::lock_guard<std::mutex> lock(cluster_.mutex_);
    token_ext_ = entry.extension & 3;
    if(entry.resource) {
        assert(pending_tokens_ > 0);
        --pending_tokens_;
    }
    cluster_.schedule_all_locked();
}

} // namespace RVTNG
