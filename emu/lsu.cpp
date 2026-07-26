#include "utils.hpp"
#include "common.hpp"
#include "lsu.hpp"
#include "swizzle.hpp"
#include "rvt_decoder.hpp"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <thread>

namespace RVTNG {
    namespace {
        constexpr uint64_t kDramBase = 0x80000000ull;
        constexpr uint64_t kUramSize = 1024ull * 1024ull;
        constexpr uint64_t kLogicalRowBytes = 64;
        constexpr uint64_t kUramPortBytes = 32;

        void fill_padding(uint8_t* dst, const uint8_t* padding, uint64_t begin = 0) {
            for(uint64_t byte = begin; byte < kLogicalRowBytes; byte++) {
                dst[byte] = padding[byte % 8];
            }
        }

        void read_uram_row(uint8_t* dst, const uint8_t* uram, uint64_t addr, uint64_t swizzle) {
            for(uint64_t byte = 0; byte < kLogicalRowBytes;) {
                const uint64_t logical_addr = addr + byte;
                const uint64_t port_offset = logical_addr & (kUramPortBytes - 1);
                const uint64_t chunk = std::min(kUramPortBytes - port_offset,
                                                kLogicalRowBytes - byte);
                const uint64_t port_addr = logical_addr - port_offset;
                const uint64_t physical_addr =
                    swizzle_addr(port_addr, swizzle) & ~(kUramPortBytes - 1);
                std::memcpy(dst + byte, uram + physical_addr + port_offset, chunk);
                byte += chunk;
            }
        }

        void write_uram_row(uint8_t* uram, uint64_t addr, uint64_t swizzle,
                            const uint8_t* src, uint64_t size) {
            for(uint64_t byte = 0; byte < size;) {
                const uint64_t logical_addr = addr + byte;
                const uint64_t port_offset = logical_addr & (kUramPortBytes - 1);
                const uint64_t chunk = std::min(kUramPortBytes - port_offset, size - byte);
                const uint64_t port_addr = logical_addr - port_offset;
                const uint64_t physical_addr =
                    swizzle_addr(port_addr, swizzle) & ~(kUramPortBytes - 1);
                std::memcpy(uram + physical_addr + port_offset, src + byte, chunk);
                byte += chunk;
            }
        }

        bool load_table_entry(const uint8_t* uram, uint64_t table_base,
                              uint64_t index, uint64_t* value) {
            const uint64_t offset = table_base + index * sizeof(uint64_t);
            if(offset > kUramSize - sizeof(uint64_t)) {
                debugf("Fault! DMA address table exceeds URAM!\n");
                return false;
            }
            std::memcpy(value, uram + offset, sizeof(*value));
            return true;
        }
    }

    void lsu_t::execute_dma(lsu_tcsr_t* tcsr, uint64_t src, uint64_t dst,
                            bool src_dram, bool dst_dram, bool gather, bool scatter) {
        if((gather && (src & 31)) || (scatter && (dst & 31))) {
            debugf("Fault! DMA address table is not aligned to 32B!\n");
            return;
        }

        for(uint64_t index = 0; index <= tcsr->tableSize; index++) {
            uint64_t task_src = src + index * tcsr->tableStride;
            uint64_t task_dst = dst + index * tcsr->tableStride;
            if(gather && !load_table_entry(uram, src, index, &task_src)) {
                return;
            }
            if(scatter && !load_table_entry(uram, dst, index, &task_dst)) {
                return;
            }
            if(!execute_dma_once(tcsr, task_src, task_dst, src_dram, dst_dram)) {
                return;
            }
        }
    }

    bool lsu_t::execute_dma_once(lsu_tcsr_t* tcsr, uint64_t src, uint64_t dst, bool src_dram, bool dst_dram) {
        if(tcsr->blockTranspose > 6 || tcsr->requestReadSize > 63 ||
           tcsr->requestWriteSize > 63) {
            debugf("Fault! Invalid DMA control value!\n");
            return false;
        }

        const uint64_t transpose_size = kLogicalRowBytes >> tcsr->blockTranspose;
        const uint64_t elem_size = uint64_t{1} << tcsr->blockTranspose;
        const uint64_t real_rows = tcsr->block + 1;
        const uint64_t requested_rows = tcsr->blockPadding + 1;
        const uint64_t minimum_rows = std::max({real_rows, requested_rows, transpose_size});
        const uint64_t output_rows =
            (minimum_rows + transpose_size - 1) / transpose_size * transpose_size;
        const uint64_t read_size = tcsr->requestReadSize + 1;
        const uint64_t write_size = tcsr->requestWriteSize + 1;

        for(uint64_t dim2 = 0 ; dim2 <= tcsr->dim2; dim2++) {
            for(uint64_t dim1 = 0; dim1 <= tcsr->dim1; dim1++) {
                for(uint64_t dim0 = 0; dim0 <= tcsr->dim0; dim0++) {
                    // 拷贝源数据到 lsu buffer
                    for(uint64_t block = 0; block < output_rows; block++) {
                        uint8_t* row = lsu_buffer + block * kLogicalRowBytes;
                        if(block >= real_rows) {
                            fill_padding(row, tcsr->paddingValue);
                        } else {
                            const uint64_t addr = src + block * tcsr->inBlockStride
                              + dim0 * tcsr->inDim0Stride
                              + dim1 * tcsr->inDim1Stride
                              + dim2 * tcsr->inDim2Stride;
                            if(tcsr->blockTranspose < 6 && (addr & 63)) {
                                debugf("Fault! DMA source is not aligned to 64B for transpose!\n");
                                return false;
                            }
                            if(src_dram) {
                                std::memcpy(row, dram + addr - kDramBase, kLogicalRowBytes);
                            } else {
                                read_uram_row(row, uram, addr, tcsr->uramReadSwizzleMBS);
                            }
                            fill_padding(row, tcsr->paddingValue, read_size);
                        }
                    }
                    // 拷贝 lsu buffer 数据到目标终点，同时进行转置。
                    for(uint64_t block = 0; block < output_rows; block += transpose_size) {
                        for(uint64_t transpose = 0; transpose < transpose_size; transpose++) {
                            // View lsu buffer as [transpose_size][transpose_size][elem_size]
                            // y = transfer / elem_size; x = transfer % elem_size;
                            // Write: [transpose][y][x]
                            // Read:  [y][transpose][x]
                            uint8_t transfer_buf[kLogicalRowBytes];
                            for(uint64_t y = 0; y < transpose_size; y++) {
                                for(uint64_t x = 0; x < elem_size; x++) {
                                    transfer_buf[y * elem_size + x] =
                                        lsu_buffer[(block + y) * kLogicalRowBytes + transpose * elem_size + x];
                                }
                            }
                            const uint64_t addr = dst + (block + transpose) * tcsr->outBlockStride
                                + dim0 * tcsr->outDim0Stride
                                + dim1 * tcsr->outDim1Stride
                                + dim2 * tcsr->outDim2Stride;
                            if(tcsr->blockTranspose < 6 && (addr & 63)) {
                                debugf("Fault! DMA destination is not aligned to 64B for transpose!\n");
                                return false;
                            }
                            if(dst_dram) {
                                // std::lock_guard<std::mutex> lock(*fdMsgMutex);
                                // backend_dram_write(fdMsg, (char*)transfer_buf, write_size, addr);
                                std::memcpy(dram + addr - kDramBase, transfer_buf, write_size);
                            } else {
                                write_uram_row(uram, addr, tcsr->uramWriteSwizzleMBS,
                                               transfer_buf, write_size);
                            }
                        }
                    }
                }
            }
        }
        debugf("LSU: 0x%08lx <- 0x%08lx\n", dst, src);
        return true;
    }

    lsu_t::lsu_t(exec_unit_t* parrent, uint8_t* uram, uint8_t* dram):uram(uram), dram(dram) {
        // TODO：分配资源
        parrent->register_issue_callback(this, &lsu_t::issue);
        this->register_commit_callback(parrent, &exec_unit_t::commit);
        this->register_scoreboard_callback(parrent, &exec_unit_t::scoreboard);
        lsu_buffer = new uint8_t[512 * 64];
        memset(this->tcsr, 0, sizeof(this->tcsr));
    }
    lsu_t::~lsu_t() {
        delete[] lsu_buffer;
    }
    ulong lsu_t::issue(const rvt_pack_t* inst) {
        uint32_t unit = (inst->inst >> 23) & 0x1f;
        uint32_t tcsr_id = (inst->inst >> 13) & 0x3ff;
        bool is_dma_tcsr = rvt::is_base(inst->inst) && (unit == RVT_UNIT_LSU || (unit== RVT_UNIT_BASE && tcsr_id == RVT_UNIT_LSU));
        bool is_dma = rvt::is_lsu(inst->inst) | is_dma_tcsr;
        if(!is_dma) {
            // ignore
            return 0;
        }
        this->inst_queue.enqueue(*inst);
        // debugf("LSU: Enqueue LSU Instruction %x!\n",(uint32_t)inst->inst);
        return 1;
    }
    void lsu_t::execute(void) {
        while(1) {
            // Fetch
            auto inst = inst_queue.dequeue();
            // Fence
            MFENCE();
            // Decode
            uint32_t unit = (inst.inst >> 23) & 0x1f;
            uint32_t tcsr_id = (inst.inst >> 13) & 0x3ff;
            bool is_dma_tcsr_write = rvt_is_base(inst.inst) && unit == RVT_UNIT_LSU && ((inst.inst >> 12) & 0x1) == 0;
            bool is_dma_tcsr_read  = rvt_is_base(inst.inst) && unit == RVT_UNIT_LSU && ((inst.inst >> 12) & 0x1) == 1;
            bool is_dma_sb_produce = rvt_is_base(inst.inst) && unit == RVT_UNIT_BASE && ((inst.inst >> 12) & 0x1) == 0;
            bool is_dma_sb_bar     = rvt_is_base(inst.inst) && unit == RVT_UNIT_BASE && ((inst.inst >> 12) & 0x1) == 1;
            if(is_dma_tcsr_write || is_dma_tcsr_read) {
                debugf("LSU: tcsr read write %d:\n", tcsr_id);
                if(tcsr_id >= 8 * 32) {
                    debugf("* Illgal id.\n");
                    continue;
                }
                uint64_t* _tcsr = (uint64_t*) &this->tcsr;
                if(is_dma_tcsr_write) {
                    if((tcsr_id % 32) != 31) {
                        const unsigned width = this->tcsr_width[tcsr_id & 0x1f];
                        const uint64_t mask = width == 64
                            ? std::numeric_limits<uint64_t>::max()
                            : ((uint64_t{1} << width) - 1);
                        _tcsr[tcsr_id] = inst.rd & mask;
                    }
                    // this->tcsr[tcsr_id / 32].checksum += 1;
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
            if(is_dma_sb_produce) {
                debugf("LSU: scoreboard produce %lx:\n", inst.rd);
                scoreboard(inst.rd);
                continue;
            }
            if(is_dma_sb_bar) {
                rvt_commit_t cmt_pack;
                cmt_pack.timestamp = inst.timestamp;
                cmt_pack.rd = (inst.inst >> 7) & 0x1f;
                cmt_pack.data = inst.rd;
                debugf("LSU: scoreboard bar %lx:\n", cmt_pack.rd);
                commit(&cmt_pack);
                continue;
            }
            // DMA 指令执行
            {
                const uint64_t group = RVT_LSU_DMA_GROUP(inst.inst);
                const bool src_is_dram = RVT_LSU_DMA_SRC(inst.inst);
                const bool dst_is_dram = RVT_LSU_DMA_DST(inst.inst);
                const bool gather = RVT_LSU_DMA_GATHER(inst.inst);
                const bool scatter = RVT_LSU_DMA_SCATTER(inst.inst);
                const uint64_t src = (src_is_dram && !gather) ? inst.rs1 : (inst.rs1 & 0xfffff);
                const uint64_t dst = (dst_is_dram && !scatter) ? inst.rd : (inst.rd & 0xfffff);
                execute_dma(&this->tcsr[group], src, dst, src_is_dram, dst_is_dram,
                            gather, scatter);
                std::atomic_thread_fence(std::memory_order_release);
                continue;
            }
        }
    }
}
