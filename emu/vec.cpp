#include "vec.hpp"
#include "swizzle.hpp"

namespace RVTNG {
    void vec_t::exec_once(rvt_pack_t inst) {
        // TODO:
    }
    vec_t::vec_t(exec_unit_t* parrent, ulong lane_width) {
        // resQ初始化
        this->lane_width = lane_width;
        resQ.clear();
        resQ.resize(2);
        for (size_t vb = 0; vb < 2; ++vb) {
            resQ[vb].resize(RESQ_SIZE);
            for (size_t i = 0; i < RESQ_SIZE; ++i) {
                resQ[vb][i].resize(this->lane_width);
            }
        }
        parrent->register_issue_callback(this, &vec_t::issue);
        memset(this->tcsr, 0, sizeof(this->tcsr));
        // 控制状态初始化
        this->mode = OUTER_MODE;
        this->curBlock = 0;
        this->curDim0 = 0;
        this->curDim1 = 0;
        this->inst_queue_block.clear();
    }
    void vec_t::issue(const rvt_pack_t* inst) {
        bool is_vec_tcsr = (inst->inst & 0xf6000000) == 0x02000000;
        bool is_vec = is_vec_tcsr | ((inst->inst & 0xe0000000) == 0x80000000) | ((inst->inst & 0xe0000000) == 0x40000000);
        if(!is_vec) {
            // ignore
            return;
        }
        this->inst_queue.enqueue(*inst);
        printf("VEC: Enqueue VEC Instruction %x!\n",(uint32_t)inst->inst);
    }
    void vec_t::execute(void) {
        while(1) {
            // Fetch
            auto inst = inst_queue.dequeue();
            // Decode
            bool is_vec_tcsr_write = (inst.inst & 0xfe004000) == 0x02000000;
            bool is_vec_tcsr_read  = (inst.inst & 0xfe004000) == 0x02004000;
            bool is_vec_sb_produce = (inst.inst & 0xfe004000) == 0x0a000000;
            bool is_vec_sb_bar     = (inst.inst & 0xfe004000) == 0x0a004000;
            if(this->mode == OUTER_MODE) {
                CSR:
                if(is_vec_tcsr_read | is_vec_tcsr_write) {
                    uint32_t tcsr_id = (inst.inst >> 15) & 0x3ff;
                    printf("VEC: tcsr read write %d:\n", tcsr_id);
                    if(tcsr_id > 8 * 32) {
                        printf("* Illgal id.\n");
                        continue;
                    }
                    uint64_t* _tcsr = (uint64_t*) &this->tcsr;
                    if(is_vec_tcsr_write) {
                        if((tcsr_id % 32) != 31)
                            _tcsr[tcsr_id] = inst.rd & ((1 << this->tcsr_width[tcsr_id & 0x1f]) - 1);
                        this->tcsr[tcsr_id / 32].checksum += 1;
                        printf("* write %lx -> %lx.\n", inst.rd, _tcsr[tcsr_id]);
                    } else {
                        rvt_commit_t cmt_pack;
                        cmt_pack.timestamp = inst.timestamp;
                        cmt_pack.rd = (inst.inst >> 7) & 0x1f;
                        cmt_pack.data = _tcsr[tcsr_id];
                        commit(&cmt_pack);
                        printf("* read %lx -> %lx.\n", _tcsr[tcsr_id], cmt_pack.rd);
                    }
                    continue;
                }
                if(is_vec_sb_produce) {
                    printf("VEC: scoreboard produce %lx:\n", inst.rd);
                    scoreboard(inst.rd);
                    continue;
                }
                if(is_vec_sb_bar) {
                    rvt_commit_t cmt_pack;
                    cmt_pack.timestamp = inst.timestamp;
                    cmt_pack.rd = (inst.inst >> 7) & 0x1f;
                    cmt_pack.data = inst.rd;
                    printf("VEC: scoreboard bar %lx:\n", cmt_pack.rd);
                    commit(&cmt_pack);
                    continue;
                }
            }
            // VEC 指令执行
            {      
                int block = this->tcsr[0].block;
                int dim0 = this->tcsr[0].dim0;
                int dim1 = this->tcsr[0].dim1;

                bool isCsr = is_vec_tcsr_read | is_vec_tcsr_write;

                if(mode == INNER_MODE) {
                    if(isCsr) {
                        // 执行vec指令
                        for(int i = 0; i < dim1; i++) {
                            for(int j =0; j < dim0; j++) {
                                for(int k = 0; k < block; k++) {
                                    for(auto& replay_inst : this -> inst_queue_block) {
                                        exec_once(replay_inst);
                                        this->curBlock = k;
                                        this->curDim0 = j;
                                        this->curDim1 = i;
                                    }      
                                }
                            }
                        }
                        // 清除指令缓存
                        this->inst_queue_block.clear();
                        // 转为外循环模式
                        this->mode = OUTER_MODE;
                        // 执行遇到的配置指令
                        goto CSR;
                    } else {
                        // 只压入不执行
                        this->inst_queue_block.push_back(inst);
                    }
                } else {
                    if(isCsr) {
                        // 直接执行
                    } else {
                        this->mode = INNER_MODE;
                        this->inst_queue_block.push_back(inst);
                    }
                }
                continue;
            }
        }
    }
}