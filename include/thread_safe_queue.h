#ifndef __THREAD_SAFE_QUEUE_H__
#define __THREAD_SAFE_QUEUE_H__

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <optional>

namespace RVTNG {

template<typename T, size_t Capacity>
class MPMCQueue {
public:
    // 阻塞式入队
    void enqueue(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this]() { return queue_.size() < Capacity; });
        queue_.push(item);
        not_empty_.notify_one();
    }

    // 阻塞式入队（移动语义）
    void enqueue(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this]() { return queue_.size() < Capacity; });
        queue_.push(std::move(item));
        not_empty_.notify_one();
    }

    // 阻塞式出队
    T dequeue() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]() { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    // 非阻塞式入队
    bool try_enqueue(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock() || queue_.size() >= Capacity) {
            return false;
        }
        queue_.push(item);
        not_empty_.notify_one();
        return true;
    }

    // 非阻塞式入队（移动语义）
    bool try_enqueue(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock() || queue_.size() >= Capacity) {
            return false;
        }
        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    // 非阻塞式出队
    std::optional<T> try_dequeue() {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock() || queue_.empty()) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    // 带超时的入队
    template<typename Rep, typename Period>
    bool try_enqueue_for(const T& item, const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_full_.wait_for(lock, timeout, [this]() { return queue_.size() < Capacity; })) {
            return false;
        }
        queue_.push(item);
        not_empty_.notify_one();
        return true;
    }

    // 带超时的出队
    template<typename Rep, typename Period>
    std::optional<T> try_dequeue_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_empty_.wait_for(lock, timeout, [this]() { return !queue_.empty(); })) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    // 获取队列当前大小
    size_t size() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

    // 检查队列是否为空
    bool empty() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    // 检查队列是否已满
    bool full() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size() >= Capacity;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
};

}

#endif // THREAD_SAFE_QUEUE_H
