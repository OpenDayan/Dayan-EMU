/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-emu is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include "rvt64.hpp"

#include <iostream>
#include <stdexcept>

using namespace DAYAN;

namespace {

class validation_unit_t : public rvt64_t {
public:
    using rvt64_t::rvt64_t;
    using rvt64_t::validate_instruction;
};

dayan_pack_t macro(unsigned extension, unsigned src_type = 0,
                 unsigned operation = 0, unsigned mode = 0)
{
    dayan_pack_t inst{};
    inst.inst = 0x3f | (uint64_t{extension} << 25)
        | (uint64_t{src_type} << 32) | (uint64_t{operation} << 40)
        | (uint64_t{mode} << 46);
    inst.rs1 = 0x1000;
    inst.rs2 = 0x2000;
    inst.rd = 0x3004;
    return inst;
}

void accept(const char* name, const dayan_pack_t& inst, const rvt64_tcsr_t& control,
            bool valid = true,
            interrupt_source_t source = interrupt_source_t::neovec_uram_store_align)
{
    exec_unit_t parent;
    interrupt_controller_t interrupts;
    validation_unit_t unit(&parent, nullptr, nullptr, &interrupts);
    if(unit.validate_instruction(inst, control) != valid
       || interrupts.count() != (valid ? 0 : 1)
       || (!valid && (interrupts.read(1) >> 32) != static_cast<uint16_t>(source))) {
        throw std::runtime_error(name);
    }
}

void check_alignment()
{
    rvt64_tcsr_t control{};
    control.dim0 = 2;
    auto inst = macro(0);
    accept("FP32 store at byte offset 4", inst, control);
    inst.rd = 0x3002;
    accept("reject partial FP32 alignment", inst, control, false);
    inst = macro(0);
    inst.rs1 += 4;
    accept("keep source 0 beat alignment", inst, control, false,
           interrupt_source_t::neovec_uram_lda_align);
    inst = macro(0);
    inst.rs2 += 4;
    accept("keep source 1 beat alignment", inst, control, false,
           interrupt_source_t::neovec_uram_ldb_align);
    inst = macro(1, 0, 3);
    inst.rd = 0x3002;
    accept("FP16 store at byte offset 2", inst, control);
    inst.rd = 0x3001;
    accept("reject partial FP16 alignment", inst, control, false);
    inst = macro(0, 0, 0, 2);
    inst.rs1 = 0xbfa00000;
    accept("iv scalar is not a URAM address", inst, control);

    control.dim1 = control.dim2 = control.dim3 = 1;
    control.Src0Dim1Stride = control.Src0Dim2Stride = control.Src0Dim3Stride = 32;
    control.Src1Dim1Stride = control.Src1Dim2Stride = control.Src1Dim3Stride = 32;
    control.DstDim1Stride = control.DstDim2Stride = control.DstDim3Stride = 4;
    accept("element-aligned stores across every outer dimension", macro(0), control);
    using field_t = uint64_t rvt64_tcsr_t::*;
    const field_t source0[] = {&rvt64_tcsr_t::Src0Dim1Stride,
        &rvt64_tcsr_t::Src0Dim2Stride, &rvt64_tcsr_t::Src0Dim3Stride};
    const field_t source1[] = {&rvt64_tcsr_t::Src1Dim1Stride,
        &rvt64_tcsr_t::Src1Dim2Stride, &rvt64_tcsr_t::Src1Dim3Stride};
    const field_t destination[] = {&rvt64_tcsr_t::DstDim1Stride,
        &rvt64_tcsr_t::DstDim2Stride, &rvt64_tcsr_t::DstDim3Stride};
    for(unsigned dim = 0; dim < 3; ++dim) {
        auto bad = control;
        bad.*source0[dim] = 4;
        accept("reject source 0 stride alignment", macro(0), bad, false,
               interrupt_source_t::neovec_uram_lda_align);
        bad = control;
        bad.*source1[dim] = 4;
        accept("reject source 1 stride alignment", macro(0), bad, false,
               interrupt_source_t::neovec_uram_ldb_align);
        bad = control;
        bad.*destination[dim] = 2;
        accept("reject destination stride alignment", macro(0), bad, false);
        bad.dim1 = bad.dim2 = bad.dim3 = 0;
        accept("ignore strides of unused dimensions", macro(0), bad);
    }
}

void check_reduction_and_range()
{
    rvt64_tcsr_t control{};
    control.dim0 = 8;
    control.dim1 = control.dim2 = control.dim3 = 1;
    control.Src0Dim1Stride = 512;
    control.Src0Dim2Stride = 2048;
    control.Src0Dim3Stride = 4096;
    control.DstDim1Stride = 4;
    control.DstDim2Stride = 32;
    control.DstDim3Stride = 256;
    // Deliberately invalid as either load or index strides: reduction ignores Src1.
    control.Src1Dim1Stride = control.Src1Dim2Stride = control.Src1Dim3Stride = 0x1fffff;
    auto inst = macro(3, 0, 4, 1);
    inst.rs2 += 4;
    accept("reduction indices use destination strides", inst, control);
    inst.rs2 += 2;
    accept("index stores still require word alignment", inst, control, false);
    inst.rs2 = 0x17fffc;
    accept("index bounds include destination strides", inst, control, false,
           interrupt_source_t::neovec_uram_store_range);
    inst = macro(0);
    control = {};
    control.dim0 = 2;
    inst.rd = 0x17fffc;
    accept("offset stores cannot cross context bounds", inst, control, false,
           interrupt_source_t::neovec_uram_store_range);
    inst = macro(0);
    inst.rs1 = 0x180000;
    accept("load bounds remain enforced", inst, control, false,
           interrupt_source_t::neovec_uram_lda_range);
}

} // namespace

int main()
{
    check_alignment();
    check_reduction_and_range();
    std::cout << "macro_address_validation_test passed\n";
}
