#include "resource_scheduler.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdint>

namespace {

using queue_t = RVTNG::MPMCQueue<RVTNG::scoreboard_entry_t, 32>;

RVTNG::scoreboard_entry_t pop(queue_t& queue,
                              RVTNG::resource_scheduler_t& scheduler)
{
    const auto entry = queue.try_dequeue();
    assert(entry);
    scheduler.output_popped(*entry);
    return *entry;
}

void write_wait(RVTNG::resource_scheduler_t& scheduler,
                const RVTNG::tensor_context_snapshot_t& context,
                uint64_t wait_free, uint64_t wait_ready, uint64_t token)
{
    scheduler.write(RVTNG::RESOURCE_WAIT_FREE, wait_free, context);
    scheduler.write(RVTNG::RESOURCE_WAIT_READY, wait_ready, context);
    scheduler.write(RVTNG::RESOURCE_WAIT, token, context);
}

void test_local_resource_flow()
{
    RVTNG::resource_cluster_t cluster;
    queue_t output;
    RVTNG::interrupt_controller_t interrupts;
    RVTNG::resource_scheduler_t scheduler(cluster, output, &interrupts);
    const auto context = RVTNG::tensor_context_snapshot_t::defaults();

    write_wait(scheduler, context, uint64_t{1} << 0, 0, 0x100);
    auto entry = pop(output, scheduler);
    assert(entry.token == 0x100 && entry.extension == 0);

    scheduler.write(RVTNG::RESOURCE_PRODUCE_BASE + 1,
                    uint64_t{1} << 8, context);
    scheduler.complete_produce(1);

    write_wait(scheduler, context, 0, uint64_t{1} << 0, 0x101);
    entry = pop(output, scheduler);
    assert(entry.token == 0x101);

    write_wait(scheduler, context, uint64_t{1} << 1,
               uint64_t{1} << 0, 0x102);
    entry = pop(output, scheduler);
    assert(entry.token == 0x102);

    scheduler.write(RVTNG::RESOURCE_CONSUME_READY,
                    uint64_t{1} << 0, context);
    scheduler.write(RVTNG::RESOURCE_PRODUCE_BASE + 2,
                    (uint64_t{1} << 8) | 1, context);
    scheduler.complete_produce(2);

    write_wait(scheduler, context, uint64_t{1} << 0, 0, 0x103);
    entry = pop(output, scheduler);
    assert(entry.token == 0x103);
    write_wait(scheduler, context, 0, uint64_t{1} << 1, 0x104);
    entry = pop(output, scheduler);
    assert(entry.token == 0x104);
    assert(!interrupts.pending());
}

void test_shared_resource_flow()
{
    RVTNG::resource_cluster_t cluster;
    queue_t output_a;
    queue_t output_b;
    RVTNG::resource_scheduler_t scheduler_a(cluster, output_a);
    RVTNG::resource_scheduler_t scheduler_b(cluster, output_b);
    const auto context = RVTNG::tensor_context_snapshot_t::defaults();
    const uint64_t shared_zero = uint64_t{1} << 32;

    write_wait(scheduler_a, context, shared_zero, 0, 0x200);
    assert(pop(output_a, scheduler_a).token == 0x200);
    write_wait(scheduler_b, context, shared_zero, 0, 0x201);
    assert(output_b.empty());

    scheduler_a.write(RVTNG::RESOURCE_PRODUCE_BASE + 1,
                      (uint64_t{1} << 8) | 32, context);
    scheduler_a.complete_produce(1);
    assert(output_b.empty());

    write_wait(scheduler_b, context, 0, shared_zero, 0x202);
    assert(pop(output_b, scheduler_b).token == 0x202);

    write_wait(scheduler_b, context, uint64_t{1} << 0,
               shared_zero, 0x203);
    assert(pop(output_b, scheduler_b).token == 0x203);
    scheduler_b.write(RVTNG::RESOURCE_CONSUME_READY, shared_zero, context);
    scheduler_b.write(RVTNG::RESOURCE_PRODUCE_BASE + 1,
                      uint64_t{1} << 8, context);
    scheduler_b.complete_produce(1);

    assert(pop(output_b, scheduler_b).token == 0x201);
}

void test_terminal_consume()
{
    RVTNG::resource_cluster_t cluster;
    queue_t output;
    RVTNG::interrupt_controller_t interrupts;
    RVTNG::resource_scheduler_t scheduler(cluster, output, &interrupts);
    const auto context = RVTNG::tensor_context_snapshot_t::defaults();
    const uint64_t resource_zero = uint64_t{1} << 0;

    write_wait(scheduler, context, resource_zero, 0, 0x250);
    assert(pop(output, scheduler).token == 0x250);
    scheduler.write(RVTNG::RESOURCE_PRODUCE_BASE + 1,
                    uint64_t{1} << 8, context);
    scheduler.complete_produce(1);

    scheduler.write(RVTNG::RESOURCE_CONSUME_READY, resource_zero, context);
    scheduler.write(RVTNG::RESOURCE_PRODUCE_BASE + 1,
                    RVTNG::RESOURCE_NO_OUTPUT, context);
    scheduler.complete_produce(1);

    write_wait(scheduler, context, resource_zero, 0, 0x251);
    assert(pop(output, scheduler).token == 0x251);
    assert(!interrupts.pending());
}

void test_election_roles()
{
    RVTNG::resource_cluster_t cluster;
    queue_t output_a;
    queue_t output_b;
    RVTNG::resource_scheduler_t scheduler_a(cluster, output_a);
    RVTNG::resource_scheduler_t scheduler_b(cluster, output_b);
    const auto context = RVTNG::tensor_context_snapshot_t::defaults();
    const uint64_t elected = uint64_t{1} << 32;

    for(uint64_t i = 0; i < 32; ++i) {
        assert(output_a.try_enqueue({0x1000 + i, 0}));
        assert(output_b.try_enqueue({0x2000 + i, 0}));
    }
    write_wait(scheduler_a, context, elected, elected, 0x300);
    write_wait(scheduler_b, context, elected, elected, 0x301);

    bool saw_producer = false;
    for(size_t i = 0; i < 33; ++i) {
        const auto entry = pop(output_a, scheduler_a);
        if(entry.token == 0x300) {
            assert(entry.extension ==
                   static_cast<uint64_t>(RVTNG::resource_role_t::producer));
            saw_producer = true;
            break;
        }
    }
    assert(saw_producer);

    scheduler_a.write(RVTNG::RESOURCE_PRODUCE_BASE + 1,
                      (uint64_t{1} << 8) | 32, context);
    scheduler_a.complete_produce(1);

    bool saw_consumer = false;
    for(size_t i = 0; i < 33; ++i) {
        const auto entry = pop(output_b, scheduler_b);
        if(entry.token == 0x301) {
            assert(entry.extension ==
                   static_cast<uint64_t>(RVTNG::resource_role_t::consumer));
            saw_consumer = true;
            break;
        }
    }
    assert(saw_consumer);
    assert(scheduler_b.read(RVTNG::RESOURCE_TOKEN_EXT) ==
           static_cast<uint64_t>(RVTNG::resource_role_t::consumer));
}

void test_election_phase_lifecycle()
{
    RVTNG::resource_cluster_t cluster;
    queue_t producer_output;
    queue_t consumer_output;
    queue_t ready_consumer_output;
    RVTNG::resource_scheduler_t producer(cluster, producer_output);
    RVTNG::resource_scheduler_t consumer(cluster, consumer_output);
    RVTNG::resource_scheduler_t ready_consumer(cluster, ready_consumer_output);
    const auto context = RVTNG::tensor_context_snapshot_t::defaults();
    const uint64_t elected = uint64_t{1} << 32;

    write_wait(producer, context, elected, 0, 0x350);
    assert(pop(producer_output, producer).token == 0x350);

    // A waiter arriving after FREE -> BUSY still observes this lifecycle and
    // must be elected as a consumer once the resource becomes READY.
    write_wait(consumer, context, elected, elected, 0x351);
    assert(consumer_output.empty());
    producer.write(RVTNG::RESOURCE_PRODUCE_BASE + 1,
                   (uint64_t{2} << 8) | 32, context);
    producer.complete_produce(1);
    const auto elected_consumer = pop(consumer_output, consumer);
    assert(elected_consumer.token == 0x351);
    assert(elected_consumer.extension ==
           static_cast<uint64_t>(RVTNG::resource_role_t::consumer));

    write_wait(ready_consumer, context, elected, elected, 0x352);
    const auto ready_elected_consumer = pop(ready_consumer_output,
                                             ready_consumer);
    assert(ready_elected_consumer.token == 0x352);
    assert(ready_elected_consumer.extension ==
           static_cast<uint64_t>(RVTNG::resource_role_t::consumer));

    for(uint64_t i = 0; i < 32; ++i) {
        assert(consumer_output.try_enqueue({0x4000 + i, 0}));
    }
    write_wait(consumer, context, elected, elected, 0x353);

    for(size_t i = 0; i < 2; ++i) {
        producer.write(RVTNG::RESOURCE_CONSUME_READY, elected, context);
        producer.write(RVTNG::RESOURCE_PRODUCE_BASE + 1,
                       RVTNG::RESOURCE_NO_OUTPUT, context);
        producer.complete_produce(1);
    }

    // The final consumption changes the lifecycle.  A waiter that missed the
    // READY window must not claim the now-FREE resource as a new producer.
    for(size_t i = 0; i < 32; ++i) {
        assert(pop(consumer_output, consumer).token == 0x4000 + i);
    }
    assert(consumer_output.empty());

    write_wait(consumer, context, elected, elected, 0x354);
    const auto next_producer = pop(consumer_output, consumer);
    assert(next_producer.token == 0x354);
    assert(next_producer.extension ==
           static_cast<uint64_t>(RVTNG::resource_role_t::producer));
}

void test_invalid_context_window()
{
    RVTNG::resource_cluster_t cluster;
    queue_t output;
    RVTNG::interrupt_controller_t interrupts;
    RVTNG::resource_scheduler_t scheduler(cluster, output, &interrupts);
    auto context = RVTNG::tensor_context_snapshot_t::defaults();
    context.local_resources = RVTNG::resource_map_t{1, 0};

    write_wait(scheduler, context, uint64_t{1} << 1, 0, 0x400);
    assert(output.empty());
    assert(interrupts.pending());
    assert(interrupts.count() == 1);
}

void test_csr_acceptance_and_init()
{
    RVTNG::resource_cluster_t cluster;
    queue_t output;
    RVTNG::interrupt_controller_t interrupts;
    RVTNG::resource_scheduler_t scheduler(cluster, output, &interrupts);
    const auto context = RVTNG::tensor_context_snapshot_t::defaults();

    assert(scheduler.write(RVTNG::RESOURCE_WAIT_QOS, 0xff, context));
    assert(scheduler.read(RVTNG::RESOURCE_WAIT_QOS) == 7);
    assert(scheduler.write(RVTNG::RESOURCE_CONSUME_READY, 2, context));
    assert(scheduler.read(RVTNG::RESOURCE_CONSUME_READY) == 2);
    assert(!scheduler.write(RVTNG::RESOURCE_PRODUCE_BASE + 1, 0, context));
    assert(!scheduler.write(RVTNG::RESOURCE_PRODUCE_BASE, 0x100, context));
    assert(!scheduler.write(RVTNG::RESOURCE_PRODUCE_BASE + 5, 0x100, context));
    assert(!scheduler.write(RVTNG::RESOURCE_PRODUCE_BASE + 1, 0x10100, context));
    assert(scheduler.read(RVTNG::RESOURCE_CONSUME_READY) == 2);
    assert(scheduler.read(RVTNG::RESOURCE_PRODUCE_BASE + 1) == 0);
    assert(output.empty());
    assert(interrupts.count() == 4);
    assert(scheduler.write(RVTNG::LOCAL_RESOURCE_INIT, 0, context));

    // 零值也是合法选举 token；只有实际弹出才更新 TOKEN_EXT 和释放初始化检查额度。
    assert(scheduler.write(RVTNG::RESOURCE_CONSUME_READY, 0, context));
    write_wait(scheduler, context, 1, 1, 0);
    assert(scheduler.read(RVTNG::RESOURCE_TOKEN_EXT) == 0);
    assert(!scheduler.write(RVTNG::LOCAL_RESOURCE_INIT, 0, context));
    const auto token = pop(output, scheduler);
    assert(token.token == 0 && token.extension == 1 && token.resource);
    assert(scheduler.read(RVTNG::RESOURCE_TOKEN_EXT) == 1);
    assert(scheduler.write(RVTNG::RESOURCE_PRODUCE_BASE + 1, 0x100, context));
    assert(!scheduler.write(RVTNG::LOCAL_RESOURCE_INIT, 0, context));
    scheduler.complete_produce(1);
    assert(output.empty());
    assert(scheduler.write(RVTNG::LOCAL_RESOURCE_INIT, 0, context));
    write_wait(scheduler, context, 0, 1, 0x700);
    assert(!scheduler.write(RVTNG::LOCAL_RESOURCE_INIT, 0, context));
    assert(interrupts.count() == 7);
}

void test_scoreboard_decode()
{
    for(uint8_t unit = 1; unit <= 4; ++unit) {
        for(uint16_t address = 0; address < 0x400; ++address) {
            const uint64_t inst = (uint64_t{address} << 13) | 0x0b;
            assert(RVTNG::is_unit_scoreboard(inst, unit) ==
                   (address == unit || address == RVTNG::RESOURCE_PRODUCE_BASE + unit));
            assert(RVTNG::is_unit_scoreboard(inst | 0x1000, unit) == (address == unit));
        }
        assert(!RVTNG::is_unit_scoreboard((uint64_t{unit} << 13), unit));
        assert(!RVTNG::is_unit_scoreboard((uint64_t{unit} << 13) | (1ull << 23) | 0x0b, unit));
    }
}

} // namespace

int main()
{
    test_local_resource_flow();
    test_shared_resource_flow();
    test_terminal_consume();
    test_election_roles();
    test_election_phase_lifecycle();
    test_invalid_context_window();
    test_csr_acceptance_and_init();
    test_scoreboard_decode();
    return 0;
}
