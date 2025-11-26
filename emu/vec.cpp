#include "vec.hpp"
#include "swizzle.hpp"
#include <cassert>

namespace RVTNG {
    #define VV_INT32 0b100001
    #define VV_FLOAT32 0b110000
    #define LOAD (0b010000)
    #define STORE (0b010001)
    void vv_int32_exec_once(ResQ* resQ, rvt_pack_t inst) {
        enum __TX_VEC_VV_INT32_OP_ENUM {
            TX_INT32_ROL = 0,
            TX_INT32_SLL = 1,
            TX_INT32_SRA = 2,
            TX_INT32_SRL = 3,
            TX_INT32_ADD = 4,
            TX_INT32_SUB = 5,
            TX_INT32_MUL = 6,
        };
        uint8_t vb = (inst.inst >> 25) & 0x1;
        int vrs1 = resQ->get((inst.inst >> 11) & 0x1, (inst.inst >> 7) & 0xf);
        int vrs2 = resQ->get((inst.inst >> 19) & 0x1, (inst.inst >> 15) & 0xf);
        uint8_t op = (inst.inst >> 12) & 0x7;

        int result;
        switch (op) {
        case 0: // TODO:
            printf("VEC: add \n\t vrs1 %x, vrs2 %x\n", vrs1, vrs2);
            result = vrs1 + vrs2;
            resQ->enqueue(vb, result);
            break;
        default:
            assert(0);
        }
    }
    void v_load_exec_once(ResQ* resQ, rvt_pack_t inst, uint64_t addr, uint64_t swizzleMBS, uint8_t* uram) {
        uint8_t vb = (inst.inst >> 25) & 0x1;
        uint32_t data = *(uint32_t *)(uram + swizzle_addr(addr, swizzleMBS));
        printf("VEC: load \n\t swizzleAddr %lx, rdata %x\n", swizzle_addr(addr, swizzleMBS), data);
        resQ->enqueue(vb, data);
    }
    void v_store_exec_once(ResQ* resQ, rvt_pack_t inst, uint64_t addr, uint64_t swizzleMBS, uint8_t* uram) {
        uint8_t vb = (inst.inst >> 19) & 0x1; // vrs2的最高位
        int vrs2 = resQ->get(vb, (inst.inst >> 15) & 0xf);
        printf("VEC: store \n\t swizzleAddr %lx, wdata %x\n", swizzle_addr(addr, swizzleMBS), vrs2);
        *(uint32_t *)(uram + swizzle_addr(addr, swizzleMBS)) = (uint32_t)vrs2;
    }
    void vec_t::exec_once(rvt_pack_t inst) {
        uint64_t rs = inst.rs2;
        uint8_t sel = (inst.inst >> 9) & 0x3;
        const uint64_t vstride_dim0[] = { 
            tcsr[0].dim0VStride0, tcsr[0].dim0VStride1, 
            tcsr[0].dim0VStride2, tcsr[0].dim0VStride3 
        };
        const uint64_t vstride_dim1[] = { 
            tcsr[0].dim1VStride0, tcsr[0].dim1VStride1, 
            tcsr[0].dim1VStride2, tcsr[0].dim1VStride3 
        };
        const uint64_t swizzleMBS[] = {
            tcsr[0].swizzleMBS0, tcsr[0].swizzleMBS1,
            tcsr[0].swizzleMBS2, tcsr[0].swizzleMBS3
        };
        uint64_t addr = rs + curBlock * 4 + curDim0 * vstride_dim0[sel] + curDim1 * vstride_dim1[sel]; // TODO: check
        
        switch ((inst.inst >> 26) & 0b111111) {
            case VV_INT32: {
                vv_int32_exec_once(&resQ, inst);
                break;
            }
            case LOAD: {
                v_load_exec_once(&resQ, inst, addr, swizzleMBS[sel], uram);
                break;
            }
            case STORE: {
                v_store_exec_once(&resQ, inst, addr, swizzleMBS[sel], uram);
                break;
            }
            default:
                assert(0);
        }
    }
    vec_t::vec_t(exec_unit_t* parrent, uint8_t* uram, ulong lane_width):uram(uram), lane_width(lane_width), resQ(lane_width) {
        // resQ初始化
        parrent->register_issue_callback(this, &vec_t::issue);
        this->register_commit_callback(parrent, &exec_unit_t::commit);
        this->register_scoreboard_callback(parrent, &exec_unit_t::scoreboard);
        // 控制状态初始化
        this->mode = OUTER_MODE;
        this->curBlock = 0;
        this->curDim0 = 0;
        this->curDim1 = 0;
        this->inst_queue_block.clear();
        memset(this->tcsr, 0, sizeof(this->tcsr));
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
                            _tcsr[tcsr_id] = inst.rd & ((1ull << this->tcsr_width[tcsr_id & 0x1f]) - 1);
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
                printf("VEC: block=%d, dim0=%d, dim1=%d\n", block, dim0, dim1);

                bool isCsr = is_vec_tcsr_read | is_vec_tcsr_write | is_vec_sb_produce | is_vec_sb_bar;

                if(mode == INNER_MODE) {
                    if(isCsr) {
                        printf("VEC: block ended!\n");
                        // 执行vec指令
                        for(int i = 0; i <= dim1; i++) {
                            for(int j =0; j <= dim0; j++) {
                                for(int k = 0; k <= block; k++) {
                                    for(auto& replay_inst : this -> inst_queue_block) {
                                        exec_once(replay_inst);
                                    }
                                    this->curBlock = k;
                                    this->curDim0 = j;
                                    this->curDim1 = i;
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
                        printf("VEC: block size %ld\n", this->inst_queue_block.size());
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