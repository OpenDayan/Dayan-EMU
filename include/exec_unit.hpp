#ifndef __EXEC_UNIT_H__
#define __EXEC_UNIT_H__

#include <vector>
#include <cstdint>
#include <functional>
#include "rvtng-intf.h"

namespace RVTNG {

using commit_call_back_f = std::function<void(const rvt_commit_t*)>;
using scoreboard_call_back_f = std::function<void(const uint64_t)>;
using issue_call_back_f = std::function<ulong(const rvt_pack_t*)>;

class exec_unit_t {
private:
    std::vector<commit_call_back_f> commit_cb;
    std::vector<scoreboard_call_back_f> scoreboard_cb;
    std::vector<issue_call_back_f> issue_cb;
public:
    void scoreboard(const uint64_t value) {
        for (const auto& cb : scoreboard_cb) {
            cb(value);
        }
    }
    void commit(const rvt_commit_t* inst) {
        for (const auto& cb : commit_cb) {
            cb(inst);
        }
    }
    // 初始化资源
    exec_unit_t() {};
    // 阻塞式的发射一条指令，返回值，发射了这条指令的单元数量
    virtual ulong issue(const rvt_pack_t* inst) {
        ulong ret = 0;
        for (const auto& cb : issue_cb) {
            ret += cb(inst);
        }
        return ret;
    }
    // 阻塞式指令提交回调注册
    template<typename T>
    void register_commit_callback(T* obj, void (T::*method)(const rvt_commit_t*)) {
        commit_cb.emplace_back(std::bind(method, obj, std::placeholders::_1));
    }
    void register_commit_callback(commit_call_back_f method) {
        commit_cb.emplace_back(method);
    }
    // 阻塞式 Scoreboard 写入回调注册
    template<typename T>
    void register_scoreboard_callback(T* obj, void (T::*method)(const uint64_t)) {
        scoreboard_cb.emplace_back(std::bind(method, obj, std::placeholders::_1));
    }
    void register_scoreboard_callback(scoreboard_call_back_f method) {
        scoreboard_cb.emplace_back(method);
    }
    // 指令发射回调注册
    template<typename T>
    void register_issue_callback(T* obj, ulong (T::*method)(const rvt_pack_t*)) {
        issue_cb.emplace_back(std::bind(method, obj, std::placeholders::_1));
    }
    void register_issue_callback(issue_call_back_f method) {
        issue_cb.emplace_back(method);
    }
};

}

#endif
