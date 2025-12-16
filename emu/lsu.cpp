#include "common.hpp"
#include "lsu.hpp"
#include "swizzle.hpp"
#include <atomic>
#include <thread>

namespace RVTNG {
    void lsu_t::execute_dma_once(lsu_tcsr_t* tcsr, uint64_t src, uint64_t dst, bool src_dram, bool dst_dram) {
        // 真实 BlockPadding 同时取决于 blockPadding 及 transpose。
        uint64_t transpose_padding = 63 >> tcsr->blockTranspose;
        uint64_t transpose_size = 64 >> tcsr->blockTranspose;
        uint64_t elem_size = 1 << tcsr->blockTranspose;
        uint64_t real_padding = transpose_padding > tcsr->blockPadding ? transpose_padding : tcsr->blockPadding;
        for(uint64_t dim2 = 0 ; dim2 <= tcsr->dim2; dim2++) {
            for(uint64_t dim1 = 0; dim1 <= tcsr->dim1; dim1++) {
                for(uint64_t dim0 = 0; dim0 <= tcsr->dim0; dim0++) {
                    // 拷贝源数据到 lsu buffer
                    for(uint64_t block = 0; block <= real_padding; block++) {
                        if(block > tcsr->block) {
                            for(int transfer = 0; transfer < 64; transfer++) {
                                lsu_buffer[block * 64 + transfer] = tcsr->paddingValue[transfer % 8];
                            }
                        } else if(src_dram) {
                            {
                                uint64_t addr = src + block * tcsr->inBlockStride
                                  + dim0  * tcsr->inDim0Stride
                                  + dim1  * tcsr->inDim1Stride
                                  + dim2  * tcsr->inDim2Stride;
                                std::lock_guard<std::mutex> lock(*fdMsgMutex);
                                backend_dram_read(fdMsg, (char*)lsu_buffer + block * 64, 64, addr);
                            }
                            for(uint64_t transfer = tcsr->requestPadding + 1; transfer < 64; transfer++) {
                                lsu_buffer[block * 64 + transfer] = tcsr->paddingValue[transfer % 8];
                            }
                        } else {
                            for(uint64_t transfer = 0; transfer < 64; transfer += 8) {
                                if(transfer <= tcsr->requestPadding) {
                                uint64_t addr = swizzle_addr(src + transfer +
                                    block * tcsr->inBlockStride
                                  + dim0  * tcsr->inDim0Stride
                                  + dim1  * tcsr->inDim1Stride
                                  + dim2  * tcsr->inDim2Stride, tcsr->uramReadSwizzleMBS);
                                    *(uint64_t*)(lsu_buffer + block * 64 + transfer) = *(uint64_t*)(uram + addr);
                                } else {
                                    *(uint64_t*)(lsu_buffer + block * 64 + transfer) = *(uint64_t*)tcsr->paddingValue;
                                }
                            }
                        }
                    }
                    // 拷贝 lsu buffer 数据到目标终点，同时进行转置。
                    for(uint64_t block = 0; block <= real_padding; block += transpose_size) {
                        for(uint64_t transpose = 0; transpose < transpose_size; transpose++) {
                            // View lsu buffer as [transpose_size][transpose_size][elem_size]
                            // y = transfer / elem_size; x = transfer % elem_size;
                            // Write: [transpose][y][x]
                            // Read:  [y][transpose][x]
                            uint8_t transfer_buf[64];
                            for(uint64_t y = 0; y < transpose_size; y++) {
                                for(uint64_t x = 0; x < elem_size; x++) {
                                    transfer_buf[y * elem_size + x] = 
                                    lsu_buffer[(block + y) * 64 + transpose * elem_size + x];
                                }
                            }
                            if(dst_dram) {
                                uint64_t addr = dst + (block + transpose) * tcsr->outBlockStride
                                    + dim0  * tcsr->outDim0Stride
                                    + dim1  * tcsr->outDim1Stride
                                    + dim2  * tcsr->outDim2Stride;
                                std::lock_guard<std::mutex> lock(*fdMsgMutex);
                                backend_dram_write(fdMsg, (char*)transfer_buf, 64, addr);
                            } else {
                                for(int transfer = 0; transfer < 64; transfer += 8) {
                                    uint64_t addr = swizzle_addr(
                                dst + transfer + (block + transpose) * tcsr->outBlockStride
                                    + dim0  * tcsr->outDim0Stride
                                    + dim1  * tcsr->outDim1Stride
                                    + dim2  * tcsr->outDim2Stride, tcsr->uramWriteSwizzleMBS);
                                    *(uint64_t*)(uram + addr) = *(uint64_t*)(transfer_buf + transfer);
                                }
                            }
                        }
                    }
                }
            }
        }
        printf("LSU: 0x%08lx <- 0x%08lx\n", dst, src);
    }

    lsu_t::lsu_t(exec_unit_t* parrent, uint8_t* uram, int fdMsg, std::mutex* fdMsgMutex):fdMsgMutex(fdMsgMutex), fdMsg(fdMsg), uram(uram) {
        // TODO：分配资源
        parrent->register_issue_callback(this, &lsu_t::issue);
        this->register_commit_callback(parrent, &exec_unit_t::commit);
        this->register_scoreboard_callback(parrent, &exec_unit_t::scoreboard);
        lsu_buffer = new uint8_t[512 * 64];
        memset(this->tcsr, 0, sizeof(this->tcsr));
    }
    ulong lsu_t::issue(const rvt_pack_t* inst) {
        bool is_dma_tcsr = (inst->inst & 0xf600007f) == (0x04000000 | RVTNG_OPCODE);
        bool is_dma = is_dma_tcsr | ((inst->inst & 0xe000007f) == (0x20000000 | RVTNG_OPCODE));
        if(!is_dma) {
            // ignore
            return 0;
        }
        this->inst_queue.enqueue(*inst);
        printf("LSU: Enqueue LSU Instruction %x!\n",(uint32_t)inst->inst);
        return 1;
    }
    void lsu_t::execute(void) {
        while(1) {
            // Fetch
            auto inst = inst_queue.dequeue();
            // Decode
            bool is_dma_tcsr_write = (inst.inst & 0xfe004000) == 0x04000000;
            bool is_dma_tcsr_read  = (inst.inst & 0xfe004000) == 0x04004000;
            bool is_dma_sb_produce = (inst.inst & 0xfe004000) == 0x0c000000;
            bool is_dma_sb_bar     = (inst.inst & 0xfe004000) == 0x0c004000;
            if(is_dma_tcsr_write | is_dma_tcsr_read) {
                uint32_t tcsr_id = (inst.inst >> 15) & 0x3ff;
                printf("LSU: tcsr read write %d:\n", tcsr_id);
                if(tcsr_id > 8 * 32) {
                    printf("* Illgal id.\n");
                    continue;
                }
                uint64_t* _tcsr = (uint64_t*) &this->tcsr;
                if(is_dma_tcsr_write) {
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
            if(is_dma_sb_produce) {
                printf("LSU: scoreboard produce %lx:\n", inst.rd);
                scoreboard(inst.rd);
                continue;
            }
            if(is_dma_sb_bar) {
                rvt_commit_t cmt_pack;
                cmt_pack.timestamp = inst.timestamp;
                cmt_pack.rd = (inst.inst >> 7) & 0x1f;
                cmt_pack.data = inst.rd;
                printf("LSU: scoreboard bar %lx:\n", cmt_pack.rd);
                commit(&cmt_pack);
            }
            // DMA 指令执行
            {
                uint64_t group = (inst.inst >> 20) & 0x7;
                bool src_is_dram = (inst.inst >> 13) & 0x1;
                bool dst_is_dram = (inst.inst >> 12) & 0x1;
                uint64_t src = src_is_dram ? inst.rs1 : (inst.rs1 & 0xfffff);
                uint64_t dst = dst_is_dram ? inst.rd  : (inst.rd  & 0xfffff);
                execute_dma_once(&this->tcsr[group], src, dst, src_is_dram, dst_is_dram);
                std::atomic_thread_fence(std::memory_order_release);
                continue;
            }
        }
    }
}