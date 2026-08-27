#ifndef __VEC_H__
#define __VEC_H__

#include "exec_unit.hpp"
#include "thread_safe_queue.h"
#include "tensor_context.hpp"
#include "interrupt.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace RVTNG {

static constexpr uint32_t VEC_INST_QUEUE_CAPACITY = 512;
static constexpr uint32_t VEC_TCSR_COUNT = 69;

struct vec_iterator_t {
    uint64_t base = 0;
    uint64_t dim0Stride = 0;
    uint64_t dim1Stride = 0;
    uint64_t dim2Stride = 0;
    uint64_t dim3Stride = 0;
    uint64_t dim4Stride = 0;
    uint64_t dim5Stride = 0;
};

struct vec_tcsr_t {
    uint64_t control = 0;
    uint64_t vec_sync = 0;
    uint64_t dim0 = 0;
    uint64_t dim1 = 0;
    uint64_t dim2 = 0;
    uint64_t dim3 = 0;
    uint64_t dim4 = 0;
    uint64_t dim5 = 0;
    uint64_t implDefHint = 0;
    vec_iterator_t s[4]{};
    vec_iterator_t v[4]{};
    uint64_t uramChannelStride[4] = {32, 32, 32, 32};
};

static_assert(sizeof(vec_tcsr_t) == VEC_TCSR_COUNT * sizeof(uint64_t));

struct vec_inst_t {
    uint64_t inst = 0;
    uint64_t rd = 0;
    uint64_t rs1 = 0;
    uint64_t rs2 = 0;
    uint64_t timestamp = 0;
};

struct vec_vfork_control_t {
    uint8_t prologue0_end = 0;
    uint8_t prologue1_end = 0;
    uint8_t loop_end = 0;
    uint8_t epilogue1_end = 0;
    uint8_t epilogue0_end = 0;
    bool dynamic = false;
};

struct vec_coord_t {
    uint64_t d0 = 0;
    uint64_t d1 = 0;
    uint64_t d2 = 0;
    uint64_t d3 = 0;
    uint64_t d4 = 0;
    uint64_t d5 = 0;
};

inline uint64_t vec_outer_coord_count(const vec_tcsr_t& tcsr)
{
    uint64_t total = 1;
    for(uint64_t dim : {tcsr.dim2, tcsr.dim3, tcsr.dim4, tcsr.dim5}) {
        const uint64_t extent = dim + 1;
        if(total > std::numeric_limits<uint64_t>::max() / extent) {
            return std::numeric_limits<uint64_t>::max();
        }
        total *= extent;
    }
    return total;
}

inline vec_coord_t vec_outer_coord(const vec_tcsr_t& tcsr, uint64_t index)
{
    vec_coord_t coord;
    const uint64_t dim2_extent = tcsr.dim2 + 1;
    const uint64_t dim3_extent = tcsr.dim3 + 1;
    const uint64_t dim4_extent = tcsr.dim4 + 1;
    coord.d2 = index % dim2_extent;
    index /= dim2_extent;
    coord.d3 = index % dim3_extent;
    index /= dim3_extent;
    coord.d4 = index % dim4_extent;
    index /= dim4_extent;
    coord.d5 = index;
    return coord;
}

struct vec_vfork_req_t {
    uint64_t block_id = 0;
    uint64_t start_slot = 0;
    uint64_t inst_count = 0;
    vec_vfork_control_t control;
    vec_tcsr_t tcsr_snapshot;
    uram_range_t uram_range;
};

struct vec_resp_t {
    uint64_t block_id = 0;
    uint64_t start_slot = 0;
    uint64_t inst_count = 0;
};

struct vec_task_req_t {
    uint64_t block_id = 0;
    uint64_t start_slot = 0;
    uint64_t inst_count = 0;
    uint64_t outer_begin = 0;
    uint64_t outer_end = 1;
    uint64_t outer_stride = 1;
    vec_vfork_control_t control;
    vec_tcsr_t tcsr_snapshot;
    uram_range_t uram_range;
};

struct vec_task_resp_t {
    uint64_t block_id = 0;
    uint64_t executed_insts = 0;
};

// Tracks the monotonically increasing vFork issue/retire sequence.  A
// scoreboard instruction snapshots issued_blocks(); later vForks must not
// extend that fence, and a snapshot of zero must be immediately ready.
class vec_block_retirement_t {
public:
    uint64_t issue()
    {
        return issued_blocks_++;
    }

    void retire(uint64_t block_id)
    {
        if(block_id != retired_blocks_) {
            throw std::logic_error("vector blocks retired out of order");
        }
        retired_blocks_++;
    }

    uint64_t snapshot() const
    {
        return issued_blocks_;
    }

    bool reached(uint64_t target) const
    {
        if(target > issued_blocks_) {
            throw std::logic_error("vector retirement target was never issued");
        }
        return retired_blocks_ >= target;
    }

    template<typename PollOne>
    void wait_until(uint64_t target, PollOne&& poll_one) const
    {
        while(!reached(target)) {
            std::forward<PollOne>(poll_one)();
        }
    }

    uint64_t retired_blocks() const
    {
        return retired_blocks_;
    }

private:
    uint64_t issued_blocks_ = 0;
    uint64_t retired_blocks_ = 0;
};

enum class vec_scoreboard_fence_kind_t : uint8_t {
    produce,
    bar
};

struct vec_scoreboard_fence_t {
    uint64_t target = 0;
    vec_scoreboard_fence_kind_t kind = vec_scoreboard_fence_kind_t::produce;
    uint64_t data = 0;
    uint64_t timestamp = 0;
    uint8_t rd = 0;
};

// Scoreboard markers complete in input order once their issue-count snapshot
// has retired.  Keeping the marker queue separate from the Vec issue loop lets
// later requests proceed without extending an older marker's fence.
class vec_scoreboard_fences_t {
public:
    void enqueue(vec_scoreboard_fence_t fence)
    {
        fences_.push_back(std::move(fence));
    }

    template<typename Reached, typename Complete>
    void drain_ready(Reached&& reached, Complete&& complete)
    {
        while(!fences_.empty() && reached(fences_.front().target)) {
            vec_scoreboard_fence_t fence = std::move(fences_.front());
            fences_.pop_front();
            complete(fence);
        }
    }

    bool empty() const
    {
        return fences_.empty();
    }

private:
    std::deque<vec_scoreboard_fence_t> fences_;
};

struct vec_sparse_memory_t {
    uint32_t read_u32(uint64_t addr);
    void write_u32(uint64_t addr, uint32_t value);
    uint32_t atomic_update_u32(uint64_t addr, uint32_t value, uint8_t op);

private:
    std::mutex mutex;
    std::unordered_map<uint64_t, uint32_t> words;
};

class vec_executer_t {
public:
    vec_executer_t(MPMCQueue<vec_vfork_req_t, 16>* vfork_queue,
                   MPMCQueue<vec_resp_t, 16>* resp_queue,
                   const vec_inst_t* inst_queue,
                   uint8_t* uram,
                   vec_sparse_memory_t* sparse_memory,
                   size_t lane_count,
                   size_t thread_count = 4,
                   size_t tracked_blocks = 4,
                   interrupt_controller_t* interrupts = nullptr);
    ~vec_executer_t();

    void start();
    void stop();

private:
    struct active_block_t {
        vec_vfork_req_t req;
        uint64_t total_tasks = 0;
        uint64_t finished_tasks = 0;
        bool dispatched = false;
        bool done = false;
    };

    void run();
    void worker_loop();
    void dispatch_block(active_block_t& block);
    void drain_task_responses();
    bool retire_ready_blocks();

    MPMCQueue<vec_vfork_req_t, 16>* vfork_queue;
    MPMCQueue<vec_resp_t, 16>* resp_queue;
    MPMCQueue<vec_task_req_t, 256> task_req_queue;
    MPMCQueue<vec_task_resp_t, 256> task_resp_queue;
    const vec_inst_t* inst_queue;
    uint8_t* uram;
    vec_sparse_memory_t* sparse_memory;
    interrupt_controller_t* interrupts;
    size_t lane_count;
    size_t thread_count;
    size_t tracked_blocks;
    std::atomic<bool> running;
    std::thread scheduler_thread;
    std::vector<std::thread> worker_threads;
    std::deque<active_block_t> active_blocks;
};

class vec_t: exec_unit_t {
public:
    ulong issue(const rvt_pack_t* inst) override;
    vec_t(exec_unit_t* parrent, uint8_t *uram, ulong lane_width,
          tensor_context_t* context = nullptr,
          interrupt_controller_t* interrupts = nullptr);
    ~vec_t();
    void execute(void);
    void enqueue_context(const rvt_pack_t& inst,
                         const tensor_context_snapshot_t& context);

protected:
    static constexpr uint32_t INST_QUEUE_CAPACITY = VEC_INST_QUEUE_CAPACITY;

    void handle_tcsr_write(const rvt_pack_t& inst, uint32_t tcsr_id,
                           const tensor_context_snapshot_t& context);
    void handle_tcsr_read(const rvt_pack_t& inst, uint32_t tcsr_id,
                          const tensor_context_snapshot_t& context);
    void handle_vfork(uint64_t payload,
                      const tensor_context_snapshot_t& context);
    void handle_vec_sync();
    void poll_responses(bool wait_one);
    void wait_for_retirement(uint64_t target);
    void drain_scoreboard_fences();
    uint64_t inst_queue_free_count() const;
    bool inst_queue_has_space() const;
    vec_vfork_control_t decode_control(uint64_t payload) const;

    MPMCQueue<queued_tensor_inst_t, 128> issue_queue;
    MPMCQueue<vec_vfork_req_t, 16> vfork_queue;
    MPMCQueue<vec_resp_t, 16> resp_queue;
    vec_sparse_memory_t sparse_memory;
    vec_executer_t executer;

    vec_inst_t inst_queue[INST_QUEUE_CAPACITY];
    uint64_t inst_queue_head;
    uint64_t inst_queue_fork_head;
    uint64_t inst_queue_tail;
    vec_block_retirement_t block_retirement;
    std::deque<vec_vfork_req_t> inflight_blocks;
    vec_scoreboard_fences_t scoreboard_fences;

    vec_tcsr_t tcsr[4];
    uint8_t* uram;
    ulong lane_width;
    tensor_context_t* context;
    interrupt_controller_t* interrupts;
    tensor_context_snapshot_t active_context;

    const char tcsr_width[VEC_TCSR_COUNT] = {
        41, 0, 24, 24, 24, 24, 24, 24, 64,
        20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20
    };
};

}

#endif
