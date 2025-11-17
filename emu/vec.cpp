#include "vec.hpp"
#include "swizzle.hpp"

namespace RVTNG {
    vec_t::vec_t(exec_unit_t* parrent, ulong lane_width) {
        // TODO：分配资源
        parrent->register_issue_callback(this, &vec_t::issue);
        // memset(this->tcsr, 0, sizeof(this->tcsr));
    }
    void vec_t::issue(const rvt_pack_t* inst) {
        // bool is_dma_tcsr = (inst->inst & 0xf6000000) == 0x04000000;
        // bool is_dma = is_dma_tcsr | ((inst->inst & 0xe0000000) == 0x20000000);
        // if(!is_dma) {
        //     // ignore
        //     return;
        // }
        // this->inst_queue.enqueue(*inst);
        // printf("LSU: Enqueue LSU Instruction %x!\n",(uint32_t)inst->inst);
    }
    void vec_t::execute(void) {
        while(1) {
            // Fetch
            auto inst = inst_queue.dequeue();
            // Decode
        }
    }
}