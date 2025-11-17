#ifndef __THREAD_SAFE_QUEUE_H__
#define __THREAD_SAFE_QUEUE_H__

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <type_traits>

namespace RVTNG {

// 单生产者单消费者队列 (SPSC)
template<typename T, size_t Capacity>
class SPSCQueue {
    static_assert(std::is_copy_assignable_v<T>, "T must be copy assignable");
    static_assert(std::is_copy_constructible_v<T>, "T must be copy constructible");

private:
    alignas(64) std::atomic<size_t> head_{0};    // 生产者写入位置
    alignas(64) std::atomic<size_t> tail_{0};    // 消费者读取位置
    alignas(64) T buffer_[Capacity];
    
    // 缓存优化：避免false sharing
    char padding_[64 - sizeof(std::atomic<size_t>)];

public:
    SPSCQueue() = default;
    ~SPSCQueue() = default;

    // 禁止拷贝
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    // 禁止移动（简化实现）
    SPSCQueue(SPSCQueue&&) = delete;
    SPSCQueue& operator=(SPSCQueue&&) = delete;

    // 非阻塞入队
    bool try_enqueue(const T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (current_head + 1) % Capacity;
        
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false; // 队列满
        }
        
        buffer_[current_head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    // 非阻塞出队
    bool try_dequeue(T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        
        if (current_tail == head_.load(std::memory_order_acquire)) {
            return false; // 队列空
        }
        
        item = buffer_[current_tail];
        tail_.store((current_tail + 1) % Capacity, std::memory_order_release);
        return true;
    }

    // 阻塞入队
    void enqueue(const T& item) {
        while (!try_enqueue(item)) {
            std::this_thread::yield();
        }
    }

    // 阻塞出队
    T dequeue() {
        T item;
        while (!try_dequeue(item)) {
            std::this_thread::yield();
        }
        return item;
    }

    // 带超时的阻塞入队
    template<typename Rep, typename Period>
    bool enqueue_for(const T& item, const std::chrono::duration<Rep, Period>& timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (try_enqueue(item)) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        return false;
    }

    // 带超时的阻塞出队
    template<typename Rep, typename Period>
    bool dequeue_for(T& item, const std::chrono::duration<Rep, Period>& timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (try_dequeue(item)) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        return false;
    }

    // 检查队列是否为空
    bool empty() const {
        return tail_.load(std::memory_order_acquire) == head_.load(std::memory_order_acquire);
    }

    // 检查队列是否为满
    bool full() const {
        const size_t next_head = (head_.load(std::memory_order_acquire) + 1) % Capacity;
        return next_head == tail_.load(std::memory_order_acquire);
    }

    // 获取队列大小
    size_t size() const {
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail + Capacity) % Capacity;
    }

    // 获取队列容量
    constexpr size_t capacity() const {
        return Capacity;
    }
};

// 多生产者多消费者队列 (MPMC)
template<typename T, size_t Capacity>
class MPMCQueue {
    static_assert(std::is_copy_assignable_v<T>, "T must be copy assignable");
    static_assert(std::is_copy_constructible_v<T>, "T must be copy constructible");

private:
    alignas(64) std::atomic<size_t> head_{0};    // 生产者写入位置
    alignas(64) std::atomic<size_t> tail_{0};    // 消费者读取位置
    alignas(64) std::mutex head_mutex_;          // 保护head_
    alignas(64) std::mutex tail_mutex_;          // 保护tail_
    std::condition_variable not_full_;           // 队列不满条件变量
    std::condition_variable not_empty_;          // 队列不空条件变量
    T buffer_[Capacity];

    // 私有辅助方法
    bool is_full() const {
        const size_t next_head = (head_.load(std::memory_order_acquire) + 1) % Capacity;
        return next_head == tail_.load(std::memory_order_acquire);
    }

    bool is_empty() const {
        return tail_.load(std::memory_order_acquire) == head_.load(std::memory_order_acquire);
    }

public:
    MPMCQueue() = default;
    ~MPMCQueue() = default;

    // 禁止拷贝
    MPMCQueue(const MPMCQueue&) = delete;
    MPMCQueue& operator=(const MPMCQueue&) = delete;

    // 禁止移动（简化实现）
    MPMCQueue(MPMCQueue&&) = delete;
    MPMCQueue& operator=(MPMCQueue&&) = delete;

    // 非阻塞入队
    bool try_enqueue(const T& item) {
        std::unique_lock<std::mutex> lock(head_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            return false; // 无法获取锁
        }

        if (is_full()) {
            return false; // 队列满
        }

        const size_t current_head = head_.load(std::memory_order_relaxed);
        buffer_[current_head] = item;
        head_.store((current_head + 1) % Capacity, std::memory_order_release);
        lock.unlock();
        
        not_empty_.notify_one();
        return true;
    }

    // 非阻塞出队
    bool try_dequeue(T& item) {
        std::unique_lock<std::mutex> lock(tail_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            return false; // 无法获取锁
        }

        if (is_empty()) {
            return false; // 队列空
        }

        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        item = buffer_[current_tail];
        tail_.store((current_tail + 1) % Capacity, std::memory_order_release);
        lock.unlock();
        
        not_full_.notify_one();
        return true;
    }

    // 阻塞入队
    void enqueue(const T& item) {
        std::unique_lock<std::mutex> lock(head_mutex_);
        
        not_full_.wait(lock, [this]() {
            return !is_full();
        });
        
        const size_t current_head = head_.load(std::memory_order_relaxed);
        buffer_[current_head] = item;
        head_.store((current_head + 1) % Capacity, std::memory_order_release);
        lock.unlock();
        
        not_empty_.notify_one();
    }

    // 阻塞出队
    T dequeue() {
        std::unique_lock<std::mutex> lock(tail_mutex_);
        
        not_empty_.wait(lock, [this]() {
            return !is_empty();
        });
        
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        T item = buffer_[current_tail];
        tail_.store((current_tail + 1) % Capacity, std::memory_order_release);
        lock.unlock();
        
        not_full_.notify_one();
        return item;
    }

    // 带超时的阻塞入队
    template<typename Rep, typename Period>
    bool enqueue_for(const T& item, const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(head_mutex_);
        
        if (!not_full_.wait_for(lock, timeout, [this]() {
            return !is_full();
        })) {
            return false; // 超时
        }
        
        const size_t current_head = head_.load(std::memory_order_relaxed);
        buffer_[current_head] = item;
        head_.store((current_head + 1) % Capacity, std::memory_order_release);
        lock.unlock();
        
        not_empty_.notify_one();
        return true;
    }

    // 带超时的阻塞出队
    template<typename Rep, typename Period>
    bool dequeue_for(T& item, const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(tail_mutex_);
        
        if (!not_empty_.wait_for(lock, timeout, [this]() {
            return !is_empty();
        })) {
            return false; // 超时
        }
        
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        item = buffer_[current_tail];
        tail_.store((current_tail + 1) % Capacity, std::memory_order_release);
        lock.unlock();
        
        not_full_.notify_one();
        return true;
    }

    // 检查队列是否为空
    bool empty() const {
        std::lock_guard<std::mutex> lock(tail_mutex_);
        return is_empty();
    }

    // 检查队列是否为满
    bool full() const {
        std::lock_guard<std::mutex> lock(head_mutex_);
        return is_full();
    }

    // 获取队列大小
    size_t size() const {
        std::lock_guard<std::mutex> head_lock(head_mutex_);
        std::lock_guard<std::mutex> tail_lock(tail_mutex_);
        
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail + Capacity) % Capacity;
    }

    // 获取队列容量
    constexpr size_t capacity() const {
        return Capacity;
    }
};

} // namespace thread_safe

#endif // THREAD_SAFE_QUEUE_H
