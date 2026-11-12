#include "resource_scheduler.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdint>

namespace {

using queue_t = DAYAN::MPMCQueue<DAYAN::scoreboard_entry_t, 32>;

DAYAN::scoreboard_entry_t pop(queue_t& queue,
                              DAYAN::resource_scheduler_t& scheduler)
{
    const auto entry = queue.try_dequeue();
    assert(entry);
    scheduler.output_popped(*entry);
    return *entry;
}

void write_wait(DAYAN::resource_scheduler_t& scheduler,
                const DAYAN::tensor_context_snapshot_t& context,
                uint64_t wait_free, uint64_t wait_ready, uint64_t token)
{
    scheduler.write(DAYAN::RESOURCE_WAIT_FREE, wait_free, context);
    scheduler.write(DAYAN::RESOURCE_WAIT_READY, wait_ready, context);
    scheduler.write(DAYAN::RESOURCE_WAIT, token, context);
}

void write_phase_wait(DAYAN::resource_scheduler_t& scheduler,
                      const DAYAN::tensor_context_snapshot_t& context,
                      uint64_t free, uint64_t ready, uint64_t phases,
                      uint64_t token)
{
    assert(scheduler.write(DAYAN::RESOURCE_WAIT_PHASE, phases, context));
    write_wait(scheduler, context, free, ready, token);
    assert(scheduler.read(DAYAN::RESOURCE_WAIT_PHASE) == 0);
}

void test_phase_ready_generations(bool shared)
{
    DAYAN::resource_cluster_t cluster;
    queue_t producer_output, peer_output;
    DAYAN::interrupt_controller_t interrupts;
    DAYAN::resource_scheduler_t producer(cluster, producer_output, &interrupts);
    DAYAN::resource_scheduler_t peer(cluster, peer_output, &interrupts);
    auto& consumer = shared ? peer : producer;
    auto& consumer_output = shared ? peer_output : producer_output;
    const auto context = DAYAN::tensor_context_snapshot_t::defaults();
    const uint8_t index = shared ? 35 : 3;
    const uint64_t mask = uint64_t{1} << index;
    const uint16_t produce = DAYAN::RESOURCE_PRODUCE_BASE + 2;

    write_wait(producer, context, mask, 0, 0x800);
    assert(pop(producer_output, producer).token == 0x800);
    assert(producer.write(produce, (2 << 8) | index, context));
    producer.complete_produce(2);
    for(unsigned generation = 0; generation < 4; ++generation) {
        const uint64_t phase = generation & 1 ? mask : 0;
        write_phase_wait(consumer, context, 0, mask, phase ^ mask, 0x810 + generation);
        assert(consumer_output.empty());

        // Both queued consumes still belong to the old READY generation.
        for(unsigned use = 0; use < 2; ++use) {
            assert(producer.write(DAYAN::RESOURCE_CONSUME_READY, mask, context));
            assert(producer.write(produce, DAYAN::RESOURCE_NO_OUTPUT, context));
            assert(consumer_output.empty());
            producer.complete_produce(2);
            assert(consumer_output.empty());
        }
        write_wait(producer, context, mask, 0, 0x820 + generation);
        assert(pop(producer_output, producer).token == 0x820 + generation);
        assert(producer.write(produce, (2 << 8) | index, context));
        assert(consumer_output.empty());
        producer.complete_produce(2);
        assert(pop(consumer_output, consumer).token == 0x810 + generation);
    }
    assert(!interrupts.pending());
}

void test_phase_context_mapping()
{
    DAYAN::resource_cluster_t cluster;
    queue_t output;
    DAYAN::interrupt_controller_t interrupts;
    DAYAN::resource_scheduler_t scheduler(cluster, output, &interrupts);
    auto context = DAYAN::tensor_context_snapshot_t::defaults();
    context.local_resources = {2, 5};
    context.shared_resources = {1, 9};
    const uint64_t shared = uint64_t{1} << 32;

    // Local input stays at phase zero; shared input advances to phase one.
    write_wait(scheduler, context, 1 | shared, 0, 0x900);
    assert(pop(output, scheduler).token == 0x900);
    assert(scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1, 0x100, context));
    assert(scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1, 0x120, context));
    scheduler.complete_produce(1);
    scheduler.complete_produce(1);
    write_phase_wait(scheduler, context, 2, 1 | shared, shared, 0x901);
    assert(output.empty());

    // Later staging writes and another Context must not alter the queued wait.
    auto other = context;
    other.local_resources.base = 20;
    other.shared_resources.base = 25;
    assert(scheduler.write(DAYAN::RESOURCE_WAIT_PHASE, 1, other));
    assert(scheduler.write(DAYAN::RESOURCE_CONSUME_READY, shared, context));
    assert(scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                           DAYAN::RESOURCE_NO_OUTPUT, context));
    scheduler.complete_produce(1);
    assert(output.empty());

    // Reset only the new staging values; the pending wait keeps its snapshot.
    assert(scheduler.write(DAYAN::RESOURCE_WAIT_PHASE, 0, context));
    write_wait(scheduler, context, shared, 0, 0x902);
    assert(pop(output, scheduler).token == 0x902);
    assert(scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1, 0x120, context));
    scheduler.complete_produce(1);
    assert(pop(output, scheduler).token == 0x901);
    assert(!interrupts.pending());
}

void test_explicit_election_phase()
{
    DAYAN::resource_cluster_t cluster;
    queue_t output;
    DAYAN::resource_scheduler_t scheduler(cluster, output);
    const auto context = DAYAN::tensor_context_snapshot_t::defaults();
    write_wait(scheduler, context, 1, 0, 0xa00);
    assert(pop(output, scheduler).token == 0xa00);
    assert(scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1, 0x100, context));
    scheduler.complete_produce(1);
    write_phase_wait(scheduler, context, 1, 1, 1, 0xa01);
    assert(output.empty());
    assert(scheduler.write(DAYAN::RESOURCE_CONSUME_READY, 1, context));
    assert(scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                           DAYAN::RESOURCE_NO_OUTPUT, context));
    scheduler.complete_produce(1);
    auto token = pop(output, scheduler);
    assert(token.token == 0xa01 && token.extension == 1);
    write_phase_wait(scheduler, context, 1, 1, 1, 0xa02);
    assert(output.empty());
    assert(scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1, 0x100, context));
    scheduler.complete_produce(1);
    token = pop(output, scheduler);
    assert(token.token == 0xa02 && token.extension == 2);
}

void test_phase_register_validation()
{
    DAYAN::resource_cluster_t cluster;
    queue_t output;
    DAYAN::interrupt_controller_t interrupts;
    DAYAN::resource_scheduler_t scheduler(cluster, output, &interrupts);
    auto context = DAYAN::tensor_context_snapshot_t::defaults();
    context.local_resources = {1, 0};
    assert(scheduler.handles(DAYAN::RESOURCE_WAIT_PHASE));
    assert(!scheduler.handles(DAYAN::RESOURCE_WAIT_PHASE + 1));
    assert(scheduler.write(DAYAN::RESOURCE_WAIT_PHASE, UINT64_MAX, context));
    assert(scheduler.write(DAYAN::RESOURCE_WAIT_READY, 2, context));
    assert(!scheduler.write(DAYAN::RESOURCE_WAIT, 0xb00, context));
    assert(scheduler.read(DAYAN::RESOURCE_WAIT_PHASE) == UINT64_MAX);
    assert(scheduler.read(DAYAN::RESOURCE_WAIT_READY) == 2);
    assert(output.empty());
    assert(interrupts.count() == 1);

    // Phase bits outside WAIT_READY are ignored, including inaccessible bits.
    assert(scheduler.write(DAYAN::RESOURCE_WAIT_READY, 1, context));
    assert(scheduler.write(DAYAN::RESOURCE_WAIT, 0xb02, context));
    assert(scheduler.read(DAYAN::RESOURCE_WAIT_PHASE) == 0);
    assert(interrupts.count() == 1);
}

void test_masked_resource_init(bool shared)
{
    DAYAN::resource_cluster_t cluster;
    queue_t output;
    DAYAN::resource_scheduler_t scheduler(cluster, output);
    auto context = DAYAN::tensor_context_snapshot_t::defaults();
    context.local_resources = {2, 3};
    context.shared_resources = {2, 7};
    const unsigned base = shared ? 7 : 3;
    const unsigned logical = shared ? 32 : 0;
    const uint64_t a = uint64_t{1} << logical;
    const uint64_t b = a << 1;
    const auto clear = [&](uint32_t mask) {
        return shared ? cluster.clear(mask << base)
                      : scheduler.write(DAYAN::LOCAL_RESOURCE_INIT,
                                         uint64_t{mask} << base, context);
    };
    const auto publish = [&](unsigned index) {
        assert(scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                               0x100 | (logical + index), context));
        scheduler.complete_produce(1);
    };
    write_wait(scheduler, context, a | b, 0, 0xc00);
    assert(pop(output, scheduler).token == 0xc00);
    publish(0);
    publish(1);
    assert(scheduler.write(DAYAN::RESOURCE_CONSUME_READY, a | b, context));
    assert(scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                           DAYAN::RESOURCE_NO_OUTPUT, context));
    scheduler.complete_produce(1);
    write_wait(scheduler, context, a | b, 0, 0xc01);
    assert(pop(output, scheduler).token == 0xc01);
    publish(0);
    publish(1);
    assert(clear(0));
    write_phase_wait(scheduler, context, 0, a | b, a | b, 0xc02);
    assert(pop(output, scheduler).token == 0xc02);

    // A new allocation can clear A while B has an unrelated pending wait.
    write_phase_wait(scheduler, context, 0, b, 0, 0xc03);
    assert(output.empty());
    assert(!clear(3));
    write_phase_wait(scheduler, context, 0, a, a, 0xc04);
    assert(pop(output, scheduler).token == 0xc04);
    assert(clear(1));
    write_wait(scheduler, context, a, 0, 0xc05);
    assert(pop(output, scheduler).token == 0xc05);
    publish(0);
    write_phase_wait(scheduler, context, 0, a, 0, 0xc06);
    assert(pop(output, scheduler).token == 0xc06);

    assert(scheduler.write(DAYAN::RESOURCE_CONSUME_READY, b, context));
    assert(scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                           DAYAN::RESOURCE_NO_OUTPUT, context));
    assert(!clear(2));
    scheduler.complete_produce(1);
    write_wait(scheduler, context, b, 0, 0xc07);
    assert(pop(output, scheduler).token == 0xc07);
    assert(scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                           0x100 | (logical + 1), context));
    assert(!clear(2));
    scheduler.complete_produce(1);
    assert(!clear(2));
    assert(pop(output, scheduler).token == 0xc03);
    assert(clear(2));
    write_phase_wait(scheduler, context, 0, a, 0, 0xc08);
    assert(pop(output, scheduler).token == 0xc08);
}

void test_local_resource_flow()
{
    DAYAN::resource_cluster_t cluster;
    queue_t output;
    DAYAN::interrupt_controller_t interrupts;
    DAYAN::resource_scheduler_t scheduler(cluster, output, &interrupts);
    const auto context = DAYAN::tensor_context_snapshot_t::defaults();

    write_wait(scheduler, context, uint64_t{1} << 0, 0, 0x100);
    auto entry = pop(output, scheduler);
    assert(entry.token == 0x100 && entry.extension == 0);

    scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                    uint64_t{1} << 8, context);
    scheduler.complete_produce(1);

    write_wait(scheduler, context, 0, uint64_t{1} << 0, 0x101);
    entry = pop(output, scheduler);
    assert(entry.token == 0x101);

    write_wait(scheduler, context, uint64_t{1} << 1,
               uint64_t{1} << 0, 0x102);
    entry = pop(output, scheduler);
    assert(entry.token == 0x102);

    scheduler.write(DAYAN::RESOURCE_CONSUME_READY,
                    uint64_t{1} << 0, context);
    scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 2,
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
    DAYAN::resource_cluster_t cluster;
    queue_t output_a;
    queue_t output_b;
    DAYAN::resource_scheduler_t scheduler_a(cluster, output_a);
    DAYAN::resource_scheduler_t scheduler_b(cluster, output_b);
    const auto context = DAYAN::tensor_context_snapshot_t::defaults();
    const uint64_t shared_zero = uint64_t{1} << 32;

    write_wait(scheduler_a, context, shared_zero, 0, 0x200);
    assert(pop(output_a, scheduler_a).token == 0x200);
    write_wait(scheduler_b, context, shared_zero, 0, 0x201);
    assert(output_b.empty());

    scheduler_a.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                      (uint64_t{1} << 8) | 32, context);
    scheduler_a.complete_produce(1);
    assert(output_b.empty());

    write_wait(scheduler_b, context, 0, shared_zero, 0x202);
    assert(pop(output_b, scheduler_b).token == 0x202);

    write_wait(scheduler_b, context, uint64_t{1} << 0,
               shared_zero, 0x203);
    assert(pop(output_b, scheduler_b).token == 0x203);
    scheduler_b.write(DAYAN::RESOURCE_CONSUME_READY, shared_zero, context);
    scheduler_b.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                      uint64_t{1} << 8, context);
    scheduler_b.complete_produce(1);

    assert(pop(output_b, scheduler_b).token == 0x201);
}

void test_terminal_consume()
{
    DAYAN::resource_cluster_t cluster;
    queue_t output;
    DAYAN::interrupt_controller_t interrupts;
    DAYAN::resource_scheduler_t scheduler(cluster, output, &interrupts);
    const auto context = DAYAN::tensor_context_snapshot_t::defaults();
    const uint64_t resource_zero = uint64_t{1} << 0;

    write_wait(scheduler, context, resource_zero, 0, 0x250);
    assert(pop(output, scheduler).token == 0x250);
    scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                    uint64_t{1} << 8, context);
    scheduler.complete_produce(1);

    scheduler.write(DAYAN::RESOURCE_CONSUME_READY, resource_zero, context);
    scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                    DAYAN::RESOURCE_NO_OUTPUT, context);
    scheduler.complete_produce(1);

    write_wait(scheduler, context, resource_zero, 0, 0x251);
    assert(pop(output, scheduler).token == 0x251);
    assert(!interrupts.pending());
}

void test_election_roles()
{
    DAYAN::resource_cluster_t cluster;
    queue_t output_a;
    queue_t output_b;
    DAYAN::resource_scheduler_t scheduler_a(cluster, output_a);
    DAYAN::resource_scheduler_t scheduler_b(cluster, output_b);
    const auto context = DAYAN::tensor_context_snapshot_t::defaults();
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
                   static_cast<uint64_t>(DAYAN::resource_role_t::producer));
            saw_producer = true;
            break;
        }
    }
    assert(saw_producer);

    scheduler_a.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                      (uint64_t{1} << 8) | 32, context);
    scheduler_a.complete_produce(1);

    bool saw_consumer = false;
    for(size_t i = 0; i < 33; ++i) {
        const auto entry = pop(output_b, scheduler_b);
        if(entry.token == 0x301) {
            assert(entry.extension ==
                   static_cast<uint64_t>(DAYAN::resource_role_t::consumer));
            saw_consumer = true;
            break;
        }
    }
    assert(saw_consumer);
    assert(scheduler_b.read(DAYAN::RESOURCE_TOKEN_EXT) ==
           static_cast<uint64_t>(DAYAN::resource_role_t::consumer));
}

void test_election_phase_lifecycle()
{
    DAYAN::resource_cluster_t cluster;
    queue_t producer_output;
    queue_t consumer_output;
    queue_t ready_consumer_output;
    DAYAN::resource_scheduler_t producer(cluster, producer_output);
    DAYAN::resource_scheduler_t consumer(cluster, consumer_output);
    DAYAN::resource_scheduler_t ready_consumer(cluster, ready_consumer_output);
    const auto context = DAYAN::tensor_context_snapshot_t::defaults();
    const uint64_t elected = uint64_t{1} << 32;

    write_wait(producer, context, elected, 0, 0x350);
    assert(pop(producer_output, producer).token == 0x350);

    // A waiter requesting phase zero during BUSY joins that same lifecycle.
    write_wait(consumer, context, elected, elected, 0x351);
    assert(consumer_output.empty());
    producer.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                   (uint64_t{2} << 8) | 32, context);
    producer.complete_produce(1);
    const auto elected_consumer = pop(consumer_output, consumer);
    assert(elected_consumer.token == 0x351);
    assert(elected_consumer.extension ==
           static_cast<uint64_t>(DAYAN::resource_role_t::consumer));

    write_wait(ready_consumer, context, elected, elected, 0x352);
    const auto ready_elected_consumer = pop(ready_consumer_output,
                                             ready_consumer);
    assert(ready_elected_consumer.token == 0x352);
    assert(ready_elected_consumer.extension ==
           static_cast<uint64_t>(DAYAN::resource_role_t::consumer));

    for(uint64_t i = 0; i < 32; ++i) {
        assert(consumer_output.try_enqueue({0x4000 + i, 0}));
    }
    write_wait(consumer, context, elected, elected, 0x353);

    for(size_t i = 0; i < 2; ++i) {
        producer.write(DAYAN::RESOURCE_CONSUME_READY, elected, context);
        producer.write(DAYAN::RESOURCE_PRODUCE_BASE + 1,
                       DAYAN::RESOURCE_NO_OUTPUT, context);
        producer.complete_produce(1);
    }

    // The final consumption changes the lifecycle.  A waiter that missed the
    // READY window must not claim the now-FREE resource as a new producer.
    for(size_t i = 0; i < 32; ++i) {
        assert(pop(consumer_output, consumer).token == 0x4000 + i);
    }
    assert(consumer_output.empty());

    write_phase_wait(consumer, context, elected, elected, elected, 0x354);
    const auto next_producer = pop(consumer_output, consumer);
    assert(next_producer.token == 0x354);
    assert(next_producer.extension ==
           static_cast<uint64_t>(DAYAN::resource_role_t::producer));
}

void test_invalid_context_window()
{
    DAYAN::resource_cluster_t cluster;
    queue_t output;
    DAYAN::interrupt_controller_t interrupts;
    DAYAN::resource_scheduler_t scheduler(cluster, output, &interrupts);
    auto context = DAYAN::tensor_context_snapshot_t::defaults();
    context.local_resources = DAYAN::resource_map_t{1, 0};

    write_wait(scheduler, context, uint64_t{1} << 1, 0, 0x400);
    assert(output.empty());
    assert(interrupts.pending());
    assert(interrupts.count() == 1);
}

void test_csr_acceptance_and_init()
{
    DAYAN::resource_cluster_t cluster;
    queue_t output;
    DAYAN::interrupt_controller_t interrupts;
    DAYAN::resource_scheduler_t scheduler(cluster, output, &interrupts);
    const auto context = DAYAN::tensor_context_snapshot_t::defaults();

    assert(scheduler.write(DAYAN::RESOURCE_WAIT_QOS, 0xff, context));
    assert(scheduler.read(DAYAN::RESOURCE_WAIT_QOS) == 7);
    assert(scheduler.write(DAYAN::RESOURCE_CONSUME_READY, 2, context));
    assert(scheduler.read(DAYAN::RESOURCE_CONSUME_READY) == 2);
    assert(!scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1, 0, context));
    assert(!scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE, 0x100, context));
    assert(!scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 5, 0x100, context));
    assert(!scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1, 0x10100, context));
    assert(scheduler.read(DAYAN::RESOURCE_CONSUME_READY) == 2);
    assert(scheduler.read(DAYAN::RESOURCE_PRODUCE_BASE + 1) == 0);
    assert(output.empty());
    assert(interrupts.count() == 4);
    assert(scheduler.write(DAYAN::LOCAL_RESOURCE_INIT, UINT32_MAX, context));

    // 零值也是合法选举 token；只有实际弹出才更新 TOKEN_EXT 和释放初始化检查额度。
    assert(scheduler.write(DAYAN::RESOURCE_CONSUME_READY, 0, context));
    write_wait(scheduler, context, 1, 1, 0);
    assert(scheduler.read(DAYAN::RESOURCE_TOKEN_EXT) == 0);
    assert(!scheduler.write(DAYAN::LOCAL_RESOURCE_INIT, UINT32_MAX, context));
    const auto token = pop(output, scheduler);
    assert(token.token == 0 && token.extension == 1 && token.resource);
    assert(scheduler.read(DAYAN::RESOURCE_TOKEN_EXT) == 1);
    assert(scheduler.write(DAYAN::RESOURCE_PRODUCE_BASE + 1, 0x100, context));
    assert(!scheduler.write(DAYAN::LOCAL_RESOURCE_INIT, UINT32_MAX, context));
    scheduler.complete_produce(1);
    assert(output.empty());
    assert(scheduler.write(DAYAN::LOCAL_RESOURCE_INIT, UINT32_MAX, context));
    write_wait(scheduler, context, 0, 1, 0x700);
    assert(!scheduler.write(DAYAN::LOCAL_RESOURCE_INIT, UINT32_MAX, context));
    assert(interrupts.count() == 7);
}

void test_scoreboard_decode()
{
    for(uint8_t unit = 1; unit <= 4; ++unit) {
        for(uint16_t address = 0; address < 0x400; ++address) {
            const uint64_t inst = (uint64_t{address} << 13) | 0x0b;
            assert(DAYAN::is_unit_scoreboard(inst, unit) ==
                   (address == unit || address == DAYAN::RESOURCE_PRODUCE_BASE + unit));
            assert(DAYAN::is_unit_scoreboard(inst | 0x1000, unit) == (address == unit));
        }
        assert(!DAYAN::is_unit_scoreboard((uint64_t{unit} << 13), unit));
        assert(!DAYAN::is_unit_scoreboard((uint64_t{unit} << 13) | (1ull << 23) | 0x0b, unit));
    }
}

} // namespace

int main()
{
    test_phase_ready_generations(false);
    test_phase_ready_generations(true);
    test_phase_context_mapping();
    test_explicit_election_phase();
    test_phase_register_validation();
    test_masked_resource_init(false);
    test_masked_resource_init(true);
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
