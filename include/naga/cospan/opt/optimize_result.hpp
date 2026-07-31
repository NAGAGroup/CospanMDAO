#pragma once

#include <cstddef>

namespace naga::cospan::opt {

struct OptimizeResult {
    bool   converged;
    size_t iterations;
    float  objective_value;
    float  grad_norm;
};

} // namespace naga::cospan::opt
