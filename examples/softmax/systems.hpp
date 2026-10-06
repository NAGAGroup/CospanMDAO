#pragma once

// =============================================================================
// examples/softmax/systems.hpp — row-wise softmax, unfused and fused
// =============================================================================
//
// Both roots promote the same two names, so the driver is variant-agnostic:
//   input  x   [n, m]
//   output y   [n, m]
//
//   SoftmaxUnfused  rowmax, expsum, normalize: three launches through global memory (x is a tie_inputs alias of the
//                   two readers' x)
//   SoftmaxFused    ONE launch: fuse_components with a single segment<local_mem::shared> of LoadMax, ExpSumLocal and
//                   NormalizeWrite; the row stays in local memory

#include <naga/cospan/cospan.hpp>

#include "components.hpp"

namespace sm {

class SoftmaxUnfused : public cs::System {
    int n_, m_;

public:
    SoftmaxUnfused(int n, int m) : n_{n}, m_{m} {}
    void setup() override {
        add_subsystem(cs::subsystem_desc<RowMax>("rowmax", n_, m_, true));
        add_subsystem(cs::subsystem_desc<ExpSum>("expsum", n_, m_, true));
        add_subsystem(cs::subsystem_desc<Normalize>("normalize", n_, m_));
        tie_inputs<DType::f32, 2>({"rowmax.x", "expsum.x"}, "x", cs::shape(n_, m_));
        connect("rowmax.mx", "expsum.mx");
        connect("expsum.e", "normalize.e");
        connect("expsum.sum", "normalize.sum");
        promote<DType::f32, 2>("normalize.y", "y");
    }
};

class SoftmaxFused : public cs::System {
    int n_, m_;

public:
    SoftmaxFused(int n, int m) : n_{n}, m_{m} {}
    void setup() override {
        add_subsystem(cs::fuse_components(
            "sm", cs::segment<cs::local_mem::shared>(cs::subsystem_desc<LoadMax>("lm", n_, m_), cs::subsystem_desc<ExpSumLocal>("es", n_, m_),
                                                     cs::subsystem_desc<NormalizeWrite>("nw", n_, m_))));
        promote<DType::f32, 2>("sm.lm.x", "x");
        promote<DType::f32, 2>("sm.nw.y", "y");
    }
};

}  // namespace sm
