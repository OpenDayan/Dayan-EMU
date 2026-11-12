#ifndef DAYAN_THREAD_SAFE_QUEUE_H
#define DAYAN_THREAD_SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <optional>
#include <utility>
#include <stdexcept>

namespace DAYAN {

template<typename T, size_t Capacity>
class MPMCQueue {
public:
    static_assert(Capacity > 0, "MPMCQueue Capacity must be greater than 0");

    // 停止队列，唤醒所有阻塞等待中的线程。
    // stop() 之后，所有 enqueue/dequeue/try_* 操作都会抛出 std::runtime_error。
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }

        not_empty_.notify_all();
        not_full_.notify_all();
    }

    // 阻塞式入队
    void enqueue(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);

        not_full_.wait(lock, [this]() {
            return stopped_ || queue_.size() < Capacity;
        });

        throw_if_stopped_locked_();

        queue_.push(item);

        lock.unlock();
        not_empty_.notify_one();
    }

    // 阻塞式入队（移动语义）
    void enqueue(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);

        not_full_.wait(lock, [this]() {
            return stopped_ || queue_.size() < Capacity;
        });

        throw_if_stopped_locked_();

        queue_.push(std::move(item));

        lock.unlock();
        not_empty_.notify_one();
    }

    // 阻塞式出队
    T dequeue() {
        std::unique_lock<std::mutex> lock(mutex_);

        not_empty_.wait(lock, [this]() {
            return stopped_ || !queue_.empty();
        });

        throw_if_stopped_locked_();

        T item = std::move(queue_.front());
        queue_.pop();

        lock.unlock();
        not_full_.notify_one();

        return item;
    }

    // 非阻塞式入队
    bool try_enqueue(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);

        if (!lock.owns_lock()) {
            return false;
        }

        throw_if_stopped_locked_();

        if (queue_.size() >= Capacity) {
            return false;
        }

        queue_.push(item);

        lock.unlock();
        not_empty_.notify_one();

        return true;
    }

    // 非阻塞式入队（移动语义）
    bool try_enqueue(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);

        if (!lock.owns_lock()) {
            return false;
        }

        throw_if_stopped_locked_();

        if (queue_.size() >= Capacity) {
            return false;
        }

        queue_.push(std::move(item));

        lock.unlock();
        not_empty_.notify_one();

        return true;
    }

    // 非阻塞式出队
    std::optional<T> try_dequeue() {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);

        if (!lock.owns_lock()) {
            return std::nullopt;
        }

        throw_if_stopped_locked_();

        if (queue_.empty()) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();

        lock.unlock();
        not_full_.notify_one();

        return item;
    }

    // 带超时的入队
    template<typename Rep, typename Period>
    bool try_enqueue_for(
        const T& item,
        const std::chrono::duration<Rep, Period>& timeout
    ) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!not_full_.wait_for(lock, timeout, [this]() {
            return stopped_ || queue_.size() < Capacity;
        })) {
            return false;
        }

        throw_if_stopped_locked_();

        queue_.push(item);

        lock.unlock();
        not_empty_.notify_one();

        return true;
    }

    // 带超时的出队
    template<typename Rep, typename Period>
    std::optional<T> try_dequeue_for(
        const std::chrono::duration<Rep, Period>& timeout
    ) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!not_empty_.wait_for(lock, timeout, [this]() {
            return stopped_ || !queue_.empty();
        })) {
            return std::nullopt;
        }

        throw_if_stopped_locked_();

        T item = std::move(queue_.front());
        queue_.pop();

        lock.unlock();
        not_full_.notify_one();

        return item;
    }

    // 获取队列当前大小
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    // 检查队列是否为空
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    // 检查队列是否已满
    bool full() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size() >= Capacity;
    }

private:
    void throw_if_stopped_locked_() const {
        if (stopped_) {
            throw std::runtime_error("queue stopped");
        }
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    bool stopped_ = false;
};

} // namespace DAYAN

#endif // DAYAN_THREAD_SAFE_QUEUE_H
