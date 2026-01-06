#include "utils.hpp"
#include "common.hpp"
#include "mat.hpp"
#include "fputils.hpp"
#include "swizzle.hpp"

namespace RVTNG {
    mat_t::mat_t(exec_unit_t* parrent, uint8_t* uram):uram(uram) {
        // TODO：分配资源
        parrent->register_issue_callback(this, &mat_t::issue);
        this->register_commit_callback(parrent, &exec_unit_t::commit);
        this->register_scoreboard_callback(parrent, &exec_unit_t::scoreboard);
        memset(this->tcsr, 0, sizeof(this->tcsr));
    }
    ulong mat_t::issue(const rvt_pack_t* inst) {
        bool is_mat_tcsr = (inst->inst & 0xf600007f) == (0x000000000 | RVTNG_OPCODE);
        bool is_mat = is_mat_tcsr | ((inst->inst & 0xf000007f) == (0x10000000 | RVTNG_OPCODE));
        if(!is_mat) {
            // ignore
            return 0;
        }
        this->inst_queue.enqueue(*inst);
        debugf("MAT: Enqueue MAT Instruction %x!\n",(uint32_t)inst->inst);
        return 1;
    }
    void mat_t::execute(void) {
        while(1) {
            // Fetch
            auto inst = inst_queue.dequeue();
            // Decode
            bool is_mat_tcsr_write = (inst.inst & 0xfe004000) == 0x00000000;
            bool is_mat_tcsr_read  = (inst.inst & 0xfe004000) == 0x00004000;
            bool is_mat_sb_produce = (inst.inst & 0xfe004000) == 0x08000000;
            bool is_mat_sb_bar     = (inst.inst & 0xfe004000) == 0x08004000;
            if(is_mat_tcsr_write | is_mat_tcsr_read) {
                uint32_t tcsr_id = (inst.inst >> 15) & 0x3ff;
                debugf("MAT: tcsr read write %d:\n", tcsr_id);
                if(tcsr_id > 8 * 32) {
                    debugf("* Illgal id.\n");
                    continue;
                }
                uint64_t* _tcsr = (uint64_t*) &this->tcsr;
                if(is_mat_tcsr_write) {
                    if((tcsr_id % 32) != 31)
                        _tcsr[tcsr_id] = inst.rd & ((1 << this->tcsr_width[tcsr_id & 0x1f]) - 1);
                    this->tcsr[tcsr_id / 32].checksum += 1;
                    debugf("* write %lx -> %lx.\n", inst.rd, _tcsr[tcsr_id]);
                } else {
                    rvt_commit_t cmt_pack;
                    cmt_pack.timestamp = inst.timestamp;
                    cmt_pack.rd = (inst.inst >> 7) & 0x1f;
                    cmt_pack.data = _tcsr[tcsr_id];
                    commit(&cmt_pack);
                    debugf("* read %lx -> %lx.\n", _tcsr[tcsr_id], cmt_pack.rd);
                }
                continue;
            }
            if(is_mat_sb_produce) {
                debugf("MAT: scoreboard produce %lx:\n", inst.rd);
                scoreboard(inst.rd);
                continue;
            }
            if(is_mat_sb_bar) {
                rvt_commit_t cmt_pack;
                cmt_pack.timestamp = inst.timestamp;
                cmt_pack.rd = (inst.inst >> 7) & 0x1f;
                cmt_pack.data = inst.rd;
                debugf("MAT: scoreboard bar %lx:\n", cmt_pack.rd);
                commit(&cmt_pack);
                continue;
            }
            // MAT 指令执行
            {
                uint64_t group = (inst.inst >> 25) & 0x7;
                uint64_t a = inst.rs1;
                uint64_t b = inst.rs2;
                uint64_t c = inst.rd;
                if(tcsr[group].dataType == 0) {
                    execute_mm_bf16_once(tcsr + group, a, b, c, !((inst.inst >> 12) & 0x1));
                } else {
                    debugf("MAT: FIXME, add support for type %ld!\n", tcsr[group].dataType);
                }
                continue;
            }
        }
    }

    // TODO: SIMD Optimization this
    template <ulong len = 128>
    static inline void fetch_row_bf16(uint64_t swizzleMBS, uint8_t* uram, uint64_t base, float* buf, ulong left) {
        #pragma GCC ivdep
        for(ulong subK = 0 ; subK < len; subK += 4) {
            if(subK < left) {
                uint64_t v = *(uint64_t*)(uram + swizzle_addr(base + subK * 2, swizzleMBS));
                buf[subK + 0] = BF16ToFP32((v >>  0) & 0xffff);
                buf[subK + 1] = BF16ToFP32((v >> 16) & 0xffff);
                buf[subK + 2] = BF16ToFP32((v >> 32) & 0xffff);
                buf[subK + 3] = BF16ToFP32((v >> 48) & 0xffff);
            } else {
                buf[subK] = 0.f;
                buf[subK + 1] = 0.f;
                buf[subK + 2] = 0.f;
                buf[subK + 3] = 0.f;
            }
        }
    }

    void mat_t::execute_mm_bf16_once(mat_tcsr_t* tcsr, uint64_t a, uint64_t b, uint64_t c, bool acc) {
        ulong M = tcsr->m + 1;
        ulong N = tcsr->n + 1;
        ulong K = tcsr->k + 1;
        // 注意： K 对齐到 16, N 对齐到 8。
        if(((K % 16) != 0) || ((N % 8) != 0)) {
            debugf("MAT: ERROR K, N, K should be aligned to 16; N should be aligned to 8");
        } else {
            debugf("MAT: Calculating M%ld N%ld K%ld\n", M, N, K);
        }
        // A buffer == float[128]
        // B buffer == float[128]
        // 开始切片循环
        float A[128];
        float B[128];
        for(ulong k = 0; k < K; k += 128) {
            for(ulong m = 0; m < M; m += 1) {
                ulong leftK = K - k;
                // Fill A
                fetch_row_bf16(tcsr->aSwizzleMBS, uram, a + m * tcsr->aStride + k * 2, A, leftK);
                for(ulong n = 0; n < N; n += 1) {
                    // Fill B
                    fetch_row_bf16(tcsr->bSwizzleMBS, uram, b + n * tcsr->bStride + k * 2, B, leftK);
                    float sum = (acc || k != 0) ? *reinterpret_cast<float *>(uram + swizzle_addr(c + m * tcsr->cStride + n * 4, tcsr->cSwizzleMBS)) : 0.f;
                    // if(std::isnan(sum)) {
                    //     debugf("[ERROR] MAT: Encounter NAN!\n");
                    // }
                    for(ulong subK = 0; subK < 128; subK += 16) {
                        if(subK >= leftK) break;
                        sum = fp32_dot_product<16>(sum, A + subK, B + subK);
                    }
                    // if(std::isnan(sum) || std::fabs(sum) >= 1e20) {
                    //     debugf("[ERROR] MAT: Encounter NAN!\n");
                    // }
                    *reinterpret_cast<float *>(uram + swizzle_addr(c + m * tcsr->cStride + n * 4, tcsr->cSwizzleMBS)) = sum;
                }
            }
        }
        debugf("MAT: 0x%05lx = 0x%05lx @ 0x%05lx\n", c, a, b);
    }
}