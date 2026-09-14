#include "core_context.hpp"
#include "rvt_decoder.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <thread>
#include <vector>

using namespace RVTNG;

// 使用现有 IPC 入口和真实执行线程，不直接调用资源完成回调。
class resource_core_test_t {
public:
    resource_core_test_t()
    {
        assert(socketpair(AF_UNIX, RVT_SOCKET_TYPE, 0, request_) == 0);
        assert(socketpair(AF_UNIX, RVT_SOCKET_TYPE, 0, response_) == 0);
        assert(socketpair(AF_UNIX, RVT_SOCKET_TYPE, 0, message_) == 0);
        for(auto& memory : uram_) memory.resize(1536 * 1024);
        dram_.resize(128 * 1024);
        memories_ = {uram_[0].data(), uram_[1].data()};
        for(int i = 0; i < 2; ++i) {
            cores_[i] = new core_context_t(i, message_[1], message_mutex_, dram_.data(),
                memories_[i], memories_.data(), memories_.size(), cluster_, dram_.size());
        }
    }

    void write(int core, uint16_t address, uint64_t value = 0, uint8_t unit = 0)
    {
        send(core, address, value, false, unit);
    }

    uint64_t read(int core, uint16_t address, uint8_t unit = 0)
    {
        send(core, address, 0, true, unit);
        for(;;) {
            const auto opcode = receive(message_[0]);
            const auto source = receive(message_[0]);
            if(opcode == 1) {
                receive(message_[0]);
                continue;
            }
            assert(opcode == 0 && source == static_cast<uint64_t>(core));
            assert(receive(message_[0]) == 7);
            const auto value = receive(message_[0]);
            // 重复处理 CSR 会留下旧 timestamp 的回复，下一次读取必须检测出来。
            assert(receive(message_[0]) == timestamp_);
            return value;
        }
    }

    void wait(int core, uint64_t free, uint64_t ready, uint64_t token)
    {
        write(core, RESOURCE_WAIT_FREE, free);
        write(core, RESOURCE_WAIT_READY, ready);
        write(core, RESOURCE_WAIT, token);
    }

    void pop(int core, uint64_t expected, uint64_t role = 0)
    {
        uint64_t value = 0;
        for(int attempt = 0; attempt < 1000; ++attempt) {
            value = read(core, 0);
            if(value != 0 || expected == 0) break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        assert(value == expected);
        assert(read(core, RESOURCE_TOKEN_EXT) == role);
    }

    void test_mmu_resource_chain()
    {
        constexpr uint64_t root = MMU_DRAM_BASE;
        constexpr uint64_t l1 = root + 0x1000;
        constexpr uint64_t l2 = root + 0x2000;
        constexpr uint64_t source_va = 0x80401fe0;
        constexpr uint64_t destination_va = 0x80403fe0;
        constexpr uint64_t shared = uint64_t{1} << 33;
        auto store_pte = [&](uint64_t address, uint64_t pa, uint64_t flags) {
            const uint64_t pte = ((pa >> 12) << 10) | flags;
            std::memcpy(dram_.data() + address - root, &pte, sizeof(pte));
        };
        store_pte(root + ((source_va >> 30) & 0x1ff) * 8, l1, 1);
        store_pte(l1 + ((source_va >> 21) & 0x1ff) * 8, l2, 1);
        // Both 64-byte transfers cross pages backed by noncontiguous DRAM.
        store_pte(l2 + 1 * 8, root + 0x9000, 7);
        store_pte(l2 + 2 * 8, root + 0x11000, 7);
        store_pte(l2 + 3 * 8, root + 0xa000, 7);
        store_pte(l2 + 4 * 8, root + 0x12000, 7);
        std::array<uint8_t, 64> expected;
        for(size_t i = 0; i < expected.size(); ++i) {
            expected[i] = static_cast<uint8_t>(37 + 13 * i);
        }
        std::copy_n(expected.begin(), 32, dram_.data() + 0x9fe0);
        std::copy_n(expected.begin() + 32, 32, dram_.data() + 0x11000);
        for(int core = 0; core < 2; ++core) {
            cores_[core]->mmu->set_fixmap(true, 0, 0);
            cores_[core]->mmu->set_ptw_base(root);
            write(core, 12, 6, RVT_UNIT_LSU);
            write(core, 14, 63, RVT_UNIT_LSU);
            write(core, 18, 63, RVT_UNIT_LSU);
        }

        wait(0, shared, 0, 0x500);
        pop(0, 0x500);
        wait(1, 0, shared, 0x501);
        assert(read(1, 0) == 0);
        dispatch(0, RVT_LSU_DMA_PATTERN | (RVT_LSU_DMA_DIR_D2U << 12),
                 0x1000, source_va);
        write(0, RESOURCE_PRODUCE_BASE + RVT_UNIT_LSU, 0x121);
        pop(1, 0x501);
        assert(std::equal(expected.begin(), expected.end(), uram_[0].data() + 0x1000));

        // Core 1 consumes core 0's URAM through the untranslated low window.
        wait(0, shared, 0, 0x502);
        assert(read(0, 0) == 0);
        dispatch(1, RVT_LSU_DMA_PATTERN | (RVT_LSU_DMA_DIR_D2D << 12),
                 destination_va, 0x00801000);
        write(1, RESOURCE_CONSUME_READY, shared);
        write(1, RESOURCE_PRODUCE_BASE + RVT_UNIT_LSU, RESOURCE_NO_OUTPUT);
        pop(0, 0x502);
        assert(std::equal(expected.begin(), expected.begin() + 32, dram_.data() + 0xafe0));
        assert(std::equal(expected.begin() + 32, expected.end(), dram_.data() + 0x12000));
        assert(read(0, 33, INTERRUPT_TCSR_UNIT) == 0);
        assert(read(1, 33, INTERRUPT_TCSR_UNIT) == 0);
    }

    void test_phase_reuse()
    {
        constexpr uint8_t index = 10;
        constexpr uint64_t mask = uint64_t{1} << index;
        wait(0, mask, 0, 0x600);
        pop(0, 0x600);
        write(0, RESOURCE_PRODUCE_BASE + RVT_UNIT_MAT, 0x100 | index);
        for(unsigned generation = 0; generation < 4; ++generation) {
            const uint64_t phase = generation & 1 ? mask : 0;
            write(0, RESOURCE_WAIT_PHASE, phase);
            wait(0, 0, mask, 0x610 + generation);
            pop(0, 0x610 + generation);
            write(0, RESOURCE_WAIT_PHASE, phase ^ mask);
            wait(0, 0, mask, 0x620 + generation);
            assert(read(0, RESOURCE_WAIT_PHASE) == 0);
            assert(read(0, 0) == 0);
            write(0, RESOURCE_CONSUME_READY, mask);
            write(0, RESOURCE_PRODUCE_BASE + RVT_UNIT_LSU, RESOURCE_NO_OUTPUT);
            wait(0, mask, 0, 0x630 + generation);
            pop(0, 0x630 + generation);
            assert(read(0, 0) == 0);
            write(0, RESOURCE_PRODUCE_BASE + RVT_UNIT_MAT, 0x100 | index);
            pop(0, 0x620 + generation);
        }
        write(0, RESOURCE_CONSUME_READY, mask);
        write(0, RESOURCE_PRODUCE_BASE + RVT_UNIT_LSU, RESOURCE_NO_OUTPUT);
        write(0, RVT_UNIT_LSU, 0x640);
        pop(0, 0x640);
        write(0, LOCAL_RESOURCE_INIT, mask);
    }

private:
    static uint64_t receive(int fd)
    {
        pollfd ready{fd, POLLIN, 0};
        assert(poll(&ready, 1, 2000) == 1);
        uint64_t value;
        assert(rvt_socket_pop(fd, &value) == 0);
        return value;
    }

    void send(int core, uint16_t address, uint64_t data, bool read, uint8_t unit = 0)
    {
        const uint64_t inst = (uint64_t{unit} << 23) | (uint64_t{address} << 13) |
            (uint64_t{read} << 12) | (uint64_t{7} << 7) | 0x0b;
        dispatch(core, inst, data);
    }

    void dispatch(int core, uint64_t inst, uint64_t rd, uint64_t rs1 = 0)
    {
        const std::array<uint64_t, 6> words = {1, inst, rd, rs1, 0, ++timestamp_};
        for(auto word : words) assert(rvt_socket_push(response_[0], word) == 0);
        cores_[core]->fetch_one(request_[1], response_[1], fetch_mutex_);
        assert(receive(request_[0]) == static_cast<uint64_t>(core));
    }

    int request_[2], response_[2], message_[2];
    std::mutex message_mutex_, fetch_mutex_;
    std::array<std::vector<uint8_t>, 2> uram_;
    std::array<uint8_t*, 2> memories_;
    std::vector<uint8_t> dram_;
    resource_cluster_t cluster_;
    std::array<core_context_t*, 2> cores_;
    uint64_t timestamp_ = 0;
};

int main()
{
    alarm(30);
    resource_core_test_t test;
    test.test_phase_reuse();
    test.test_mmu_resource_chain();
    assert(test.read(0, TENSOR_RESOURCE_CONTEXT_ADDRESS) == 0x200400);
    test.write(0, RESOURCE_WAIT_QOS, 0xff);
    assert(test.read(0, RESOURCE_WAIT_QOS) == 7);
    test.wait(0, 1, 0, 0x100);
    test.pop(0, 0x100);
    assert(test.read(0, RESOURCE_WAIT_FREE) == 0);
    assert(test.read(0, RESOURCE_WAIT_QOS) == 0);

    // 与 RTL ResourceIntegrationSpec 相同的 LSU -> MAT -> VEC -> MACRO 资源依赖链。
    for(uint8_t unit = 1; unit <= 4; ++unit) {
        if(unit > 1) {
            test.wait(0, 1ull << (unit - 1), 1ull << (unit - 2), 0x100 + unit);
            test.pop(0, 0x100 + unit);
            test.write(0, RESOURCE_CONSUME_READY, 1ull << (unit - 2));
        }
        test.write(0, RESOURCE_PRODUCE_BASE + unit, 0x100 | (unit - 1));
        assert(test.read(0, RESOURCE_CONSUME_READY) == 0);
        assert(test.read(0, RESOURCE_PRODUCE_BASE + unit) == 0);
        test.wait(0, 0, 1ull << (unit - 1), 0x110 + unit);
        test.pop(0, 0x110 + unit);
        assert(test.read(0, 0) == 0);
        test.write(0, unit, 0x120 + unit);
        test.pop(0, 0x120 + unit);
    }
    test.write(0, RESOURCE_CONSUME_READY, 8);
    test.write(0, RESOURCE_PRODUCE_BASE + 4, 0xff);
    test.wait(0, 8, 0, 0x130);
    test.pop(0, 0x130);
    test.write(0, LOCAL_RESOURCE_INIT, UINT32_MAX);
    test.wait(0, 1, 1, 0);
    test.write(0, 0, 0xdead);
    assert(test.read(0, RESOURCE_TOKEN_EXT) == 0);
    test.pop(0, 0, 1);
    assert(test.read(0, 0) == 0);
    assert(test.read(0, RESOURCE_TOKEN_EXT) == 1);
    assert(test.read(0, 33, INTERRUPT_TCSR_UNIT) == 0);

    test.write(0, TENSOR_RESOURCE_CONTEXT_ADDRESS, 31 | (2 << 5));
    assert(test.read(0, TENSOR_RESOURCE_CONTEXT_ADDRESS) == 0x200400);
    test.write(0, RESOURCE_PRODUCE_BASE + 1, 0x10100);
    test.write(0, RESOURCE_PRODUCE_BASE + 1, 0x100);
    test.wait(0, 0, 1, 0x140);
    test.pop(0, 0x140);
    assert(test.read(0, 33, INTERRUPT_TCSR_UNIT) == 2);
    assert(test.read(0, 1, INTERRUPT_TCSR_UNIT) ==
        (uint64_t{static_cast<uint16_t>(interrupt_source_t::resource_scheduler_error)} << 32 |
         TENSOR_RESOURCE_CONTEXT_ADDRESS));
    test.write(0, RESOURCE_PRODUCE_BASE + 1, 0x100);
    assert(test.read(0, 1) == 0);
    assert(test.read(0, 33, INTERRUPT_TCSR_UNIT) == 3);
    assert(test.read(0, 3, INTERRUPT_TCSR_UNIT) ==
        (uint64_t{static_cast<uint16_t>(interrupt_source_t::resource_scheduler_error)} << 32 |
         (RESOURCE_PRODUCE_BASE + 1)));

    // 两核共享链也与 RTL 一致，B 的 FREE 等待必须晚于最后一次输入消费。
    const uint64_t shared = 1ull << 32;
    test.wait(0, shared, 0, 0x200);
    test.pop(0, 0x200);
    test.wait(1, shared, 0, 0x201);
    assert(test.read(1, 0) == 0);
    test.write(0, RESOURCE_PRODUCE_BASE + 1, 0x120);
    test.wait(1, 0, shared, 0x202);
    test.pop(1, 0x202);
    test.wait(1, 1, shared, 0x203);
    test.pop(1, 0x203);
    test.write(1, RESOURCE_CONSUME_READY, shared);
    test.write(1, RESOURCE_PRODUCE_BASE + 1, 0x100);
    test.pop(1, 0x201);
    assert(test.read(1, 33, INTERRUPT_TCSR_UNIT) == 0);
    test.write(1, LOCAL_RESOURCE_INIT, UINT32_MAX);
    for(unsigned index = 0; index < 32; ++index) test.wait(1, 1ull << index, 0, 0x400 + index);
    test.write(1, RESOURCE_PRODUCE_BASE + 1, 0x11f);
    assert(test.read(1, 1) == 0);
    assert(test.read(1, 33, INTERRUPT_TCSR_UNIT) == 0);
    test.write(1, LOCAL_RESOURCE_INIT, UINT32_MAX);
    assert(test.read(1, 33, INTERRUPT_TCSR_UNIT) == 1);
    for(unsigned index = 0; index < 32; ++index) test.pop(1, 0x400 + index);
    std::cout << "Resource core integration passed (four units, two cores, MMU DMA)" << std::endl;
    // 模型执行线程是永久服务循环；测试进程结束时统一退出，不调用等待线程的核析构函数。
    std::_Exit(0);
}
