#include "utils.hpp"
#include "common.hpp"
#include "lsu.hpp"
#include "swizzle.hpp"
#include "rvt_decoder.hpp"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <iterator>
#include <thread>

namespace RVTNG {
    namespace {
        constexpr uint64_t kUramSize = 1536ull * 1024ull;
        constexpr uint64_t kTensorUramBase = 0x00800000ull;
        constexpr uint64_t kTensorUramStride = 2ull * 1024ull * 1024ull;
        constexpr uint64_t kUramAddressMask = kTensorUramStride - 1;
        constexpr uint64_t kTensorUramBytes = 1536ull * 1024ull;
        constexpr uint64_t kLogicalRowBytes = 64;
        constexpr uint64_t kUramPortBytes = 32;

        void fill_padding(uint8_t* dst, const uint8_t* padding, uint64_t begin = 0) {
            for(uint64_t byte = begin; byte < kLogicalRowBytes; byte++) {
                dst[byte] = padding[byte % 8];
            }
        }

        void read_uram_row(uint8_t* dst, const uint8_t* uram, uint64_t addr,
                           uint64_t swizzle, uint64_t size) {
            for(uint64_t byte = 0; byte < size;) {
                const uint64_t logical_addr = addr + byte;
                const uint64_t port_offset = logical_addr & (kUramPortBytes - 1);
                const uint64_t chunk = std::min(kUramPortBytes - port_offset,
                                                size - byte);
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

    uint8_t* lsu_t::tensor_uram_address(uint64_t addr,
                                        uint64_t size) const {
        if(!cluster_urams || addr < kTensorUramBase) {
            return nullptr;
        }
        const uint64_t relative = addr - kTensorUramBase;
        const uint64_t core = relative / kTensorUramStride;
        const uint64_t offset = relative % kTensorUramStride;
        if(core >= cluster_core_count || offset > kTensorUramBytes ||
           size > kTensorUramBytes - offset) {
            return nullptr;
        }
        return cluster_urams[core] + offset;
    }

    void lsu_t::execute_dma(lsu_tcsr_t* tcsr, uint64_t src, uint64_t dst,
                            bool src_dram, bool dst_dram, bool gather, bool scatter) {
        if(gather && !check_uram_access(src, sizeof(uint64_t),
                                       interrupt_source_t::neodma_uram_table_range,
                                       interrupt_source_t::neodma_uram_table_align)) {
            return;
        }
        if(scatter && !check_uram_access(dst, sizeof(uint64_t),
                                        interrupt_source_t::neodma_uram_table_range,
                                        interrupt_source_t::neodma_uram_table_align)) {
            return;
        }
        if((gather && (src & 31)) || (scatter && (dst & 31))) {
            debugf("Fault! DMA address table is not aligned to 32B!\n");
            return;
        }

        for(uint64_t index = 0; index <= tcsr->tableSize; index++) {
            uint64_t task_src = src + index * tcsr->tableStride;
            uint64_t task_dst = dst + index * tcsr->tableStride;
            const uint64_t table_address = (gather ? src : dst) +
                index * sizeof(uint64_t);
            if((gather || scatter) &&
               !check_uram_access(table_address, sizeof(uint64_t),
                                  interrupt_source_t::neodma_uram_table_range,
                                  interrupt_source_t::neodma_uram_table_align,
                                  0)) {
                return;
            }
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
        const uint64_t uram_alignment =
            tcsr->blockTranspose < 6 ? kUramPortBytes : 0;

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
                                // D-side addresses are virtual whenever the
                                // MMU is enabled.  Translate first, then use
                                // the resulting PA to select a tensor URAM
                                // window or the shared DRAM image.  Looking up
                                // the VA directly would let a VA that happens
                                // to overlap the URAM aperture bypass page
                                // tables, permissions, and redirects.
                                uint64_t physical_address = 0;
                                if(!translate_dram_address(
                                       addr, mmu_access_type_t::read,
                                       read_size, &physical_address)) {
                                    debugf("Fault! Invalid D-side DMA source!\n");
                                    return false;
                                }
                                const uint8_t* source =
                                    tensor_uram_address(physical_address,
                                                        read_size);
                                if(!source) {
                                    source = dram +
                                        (physical_address - dram_base);
                                }
                                std::memcpy(row, source, read_size);
                            } else {
                                if(!check_uram_access(
                                       addr, read_size,
                                       interrupt_source_t::neodma_uram_read_range,
                                       interrupt_source_t::neodma_uram_read_align,
                                       uram_alignment,
                                       tcsr->uramReadSwizzleMBS)) {
                                    return false;
                                }
                                read_uram_row(row, uram, addr,
                                              tcsr->uramReadSwizzleMBS,
                                              read_size);
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
                                uint64_t physical_address = 0;
                                if(!translate_dram_address(
                                       addr, mmu_access_type_t::write,
                                       write_size, &physical_address)) {
                                    debugf("Fault! Invalid D-side DMA destination!\n");
                                    return false;
                                }
                                uint8_t* destination =
                                    tensor_uram_address(physical_address,
                                                        write_size);
                                if(!destination) {
                                    destination = dram +
                                        (physical_address - dram_base);
                                }
                                std::memcpy(destination, transfer_buf,
                                            write_size);
                            } else {
                                if(!check_uram_access(
                                       addr, write_size,
                                       interrupt_source_t::neodma_uram_write_range,
                                       interrupt_source_t::neodma_uram_write_align,
                                       uram_alignment,
                                       tcsr->uramWriteSwizzleMBS)) {
                                    return false;
                                }
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

    lsu_t::lsu_t(exec_unit_t* parrent, uint8_t* uram, uint8_t* dram,
                 uint8_t *const *cluster_urams,
                 size_t cluster_core_count,
                 tensor_context_t* context,
                 interrupt_controller_t* interrupts,
                 mmu_t* mmu,
                 size_t dram_size,
                 uint64_t dram_base,
                 uint8_t mmu_port)
        : uram(uram), dram(dram), dram_size(dram_size),
          dram_base(dram_base & MMU_PA_MASK), mmu(mmu), mmu_port(mmu_port),
          cluster_urams(cluster_urams),
          cluster_core_count(cluster_core_count), context(context),
          interrupts(interrupts),
          active_context(tensor_context_snapshot_t::defaults()) {
        // TODO：分配资源
        parrent->register_issue_callback(this, &lsu_t::issue);
        this->register_commit_callback(parrent, &exec_unit_t::commit);
        this->register_scoreboard_callback(parrent, &exec_unit_t::scoreboard);
        lsu_buffer = new uint8_t[512 * 64];
        memset(this->tcsr, 0, sizeof(this->tcsr));
    }

    bool lsu_t::translate_dram_address(uint64_t address,
                                       mmu_access_type_t access,
                                       size_t size,
                                       uint64_t* physical_address) {
        if(!physical_address) {
            return false;
        }

        uint64_t translated = address;
        if(mmu) {
            const mmu_translation_result_t result =
                mmu->translate(address, access, mmu_port);
            translated = result.pa;
            if(!result.ok()) {
                debugf("LSU: MMU fault=%u va=0x%lx redirect=0x%lx\n",
                       static_cast<unsigned>(result.fault), address,
                       translated);
            }
        }
        translated &= MMU_PA_MASK;
        // A translated D-side address may legally target a remote tensor
        // URAM window.  Such a PA is not part of the shared DRAM image, so
        // accept it before applying the DRAM range check below.
        if(tensor_uram_address(translated, size) != nullptr) {
            *physical_address = translated;
            return true;
        }
        if(!dram || translated < dram_base) {
            return false;
        }
        const uint64_t offset = translated - dram_base;
        if(dram_size != 0) {
            if(offset > dram_size || size > dram_size - offset) {
                return false;
            }
        } else if(size > std::numeric_limits<size_t>::max() - offset) {
            return false;
        }
        *physical_address = translated;
        return true;
    }
    lsu_t::~lsu_t() {
        delete[] lsu_buffer;
    }

    void lsu_t::enqueue_context(const rvt_pack_t& inst,
                                const tensor_context_snapshot_t& context_value) {
        inst_queue.enqueue(queued_tensor_inst_t{inst, context_value, true});
    }

    bool lsu_t::check_uram_access(
        uint64_t address, size_t size,
        interrupt_source_t range_source,
        interrupt_source_t align_source,
        size_t alignment,
        uint64_t swizzle) const {
        if(context == nullptr && interrupts == nullptr) {
            return true;
        }
        const uint64_t physical_address = swizzle_addr_raw(address, swizzle);
        if(alignment != 0 && (physical_address % alignment) != 0) {
            if(interrupts) {
                interrupts->raise(align_source,
                                  static_cast<uint32_t>(physical_address));
            }
            return false;
        }

        const uint64_t access_size = std::max<size_t>(size, 1);
        uint64_t consumed = 0;
        while(consumed < access_size) {
            if(address > std::numeric_limits<uint64_t>::max() - consumed) {
                if(interrupts) {
                    interrupts->raise(range_source,
                                      std::numeric_limits<uint32_t>::max());
                }
                return false;
            }
            const uint64_t logical_address = address + consumed;
            const uint64_t port_offset = logical_address & (kUramPortBytes - 1);
            const uint64_t logical_port = logical_address - port_offset;
            const uint64_t physical_port =
                swizzle_addr_raw(logical_port, swizzle) &
                ~(kUramPortBytes - 1);
            if(!active_context.uram_range.contains(physical_port,
                                                   kUramPortBytes)) {
                if(interrupts) {
                    interrupts->raise(range_source,
                                      static_cast<uint32_t>(physical_port));
                }
                return false;
            }
            consumed += std::min<uint64_t>(kUramPortBytes - port_offset,
                                           access_size - consumed);
        }
        return true;
    }

    ulong lsu_t::issue(const rvt_pack_t* inst) {
        uint32_t unit = (inst->inst >> 23) & 0x1f;
        uint32_t tcsr_id = (inst->inst >> 13) & 0x3ff;
        bool is_dma_tcsr = rvt::is_base(inst->inst) &&
            (unit == RVT_UNIT_LSU ||
             (unit == RVT_UNIT_BASE &&
              (tcsr_id & 0x1f) == RVT_UNIT_LSU));
        bool is_dma = rvt::is_lsu(inst->inst) | is_dma_tcsr;
        if(!is_dma) {
            // ignore
            return 0;
        }
        const auto snapshot = context ? context->snapshot()
                                      : tensor_context_snapshot_t::defaults();
        this->inst_queue.enqueue(queued_tensor_inst_t{*inst, snapshot, false});
        // debugf("LSU: Enqueue LSU Instruction %x!\n",(uint32_t)inst->inst);
        return 1;
    }
    void lsu_t::execute(void) {
        while(1) {
            // Fetch
            auto queued = inst_queue.dequeue();
            const auto& inst = queued.inst;
            active_context = queued.context;
            // Fence
            MFENCE();
            if(queued.context_write) {
                continue;
            }
            // Decode
            uint32_t unit = (inst.inst >> 23) & 0x1f;
            uint32_t tcsr_id = (inst.inst >> 13) & 0x3ff;
            bool is_dma_tcsr_write = rvt_is_base(inst.inst) && unit == RVT_UNIT_LSU && ((inst.inst >> 12) & 0x1) == 0;
            bool is_dma_tcsr_read  = rvt_is_base(inst.inst) && unit == RVT_UNIT_LSU && ((inst.inst >> 12) & 0x1) == 1;
            bool is_dma_sb_produce = rvt_is_base(inst.inst) &&
                unit == RVT_UNIT_BASE && tcsr_id == RVT_UNIT_LSU &&
                ((inst.inst >> 12) & 0x1) == 0;
            bool is_dma_sb_bar = rvt_is_base(inst.inst) &&
                unit == RVT_UNIT_BASE && tcsr_id == RVT_UNIT_LSU &&
                ((inst.inst >> 12) & 0x1) == 1;
            if(is_dma_tcsr_write || is_dma_tcsr_read) {
                debugf("LSU: tcsr read write %d:\n", tcsr_id);
                const uint8_t logical_group = tcsr_id / 32;
                const auto physical_group = active_context.physical_group(
                    tensor_unit_t::lsu, logical_group, std::size(tcsr));
                if(!physical_group) {
                    if(interrupts) {
                        interrupts->raise(
                            interrupt_source_t::neodma_tcsr_access_error,
                            static_cast<uint32_t>(inst.inst));
                    }
                    if(is_dma_tcsr_read) {
                        rvt_commit_t cmt_pack{};
                        cmt_pack.timestamp = inst.timestamp;
                        cmt_pack.rd = (inst.inst >> 7) & 0x1f;
                        cmt_pack.data = 0;
                        commit(&cmt_pack);
                    }
                    continue;
                }
                uint64_t* _tcsr = reinterpret_cast<uint64_t*>(
                    &this->tcsr[*physical_group]);
                const uint32_t sub_address = tcsr_id & 0x1f;
                if(is_dma_tcsr_write) {
                    if(sub_address != 31) {
                        const unsigned width = this->tcsr_width[sub_address];
                        const uint64_t mask = width == 64
                            ? std::numeric_limits<uint64_t>::max()
                            : ((uint64_t{1} << width) - 1);
                        _tcsr[sub_address] = inst.rd & mask;
                    }
                    // this->tcsr[tcsr_id / 32].checksum += 1;
                    debugf("* write %lx -> %lx.\n", inst.rd, _tcsr[sub_address]);
                } else {
                    rvt_commit_t cmt_pack;
                    cmt_pack.timestamp = inst.timestamp;
                    cmt_pack.rd = (inst.inst >> 7) & 0x1f;
                    cmt_pack.data = _tcsr[sub_address];
                    commit(&cmt_pack);
                    debugf("* read %lx -> %lx.\n", _tcsr[sub_address], cmt_pack.rd);
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
            if(!IS_RVT_LSU_DMA(inst.inst)) {
                if(interrupts) {
                    interrupts->raise(
                        interrupt_source_t::neodma_decode_error,
                        static_cast<uint32_t>(inst.inst));
                }
                continue;
            }
            // DMA 指令执行
            {
                const uint64_t group = RVT_LSU_DMA_GROUP(inst.inst);
                lsu_tcsr_t empty_tcsr{};
                const auto physical_group = active_context.physical_group(
                    tensor_unit_t::lsu, group, std::size(tcsr));
                if(!physical_group && interrupts) {
                    interrupts->raise(
                        interrupt_source_t::neodma_tcsr_access_error,
                        static_cast<uint32_t>(inst.inst));
                }
                const bool src_is_dram = RVT_LSU_DMA_SRC(inst.inst);
                const bool dst_is_dram = RVT_LSU_DMA_DST(inst.inst);
                const bool gather = RVT_LSU_DMA_GATHER(inst.inst);
                const bool scatter = RVT_LSU_DMA_SCATTER(inst.inst);
                const uint64_t src = (src_is_dram && !gather)
                    ? inst.rs1 : (inst.rs1 & kUramAddressMask);
                const uint64_t dst = (dst_is_dram && !scatter)
                    ? inst.rd : (inst.rd & kUramAddressMask);
                lsu_tcsr_t* selected_tcsr = physical_group
                    ? &this->tcsr[*physical_group] : &empty_tcsr;
                execute_dma(selected_tcsr, src, dst, src_is_dram, dst_is_dram,
                            gather, scatter);
                std::atomic_thread_fence(std::memory_order_release);
                continue;
            }
        }
    }
}
