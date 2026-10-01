#pragma once
// RunNSystem (r01 §10.1, D137): a System that runs its subtree on the first n
// forwards and then returns early. Do-once is n = 1; there is no separate type.

#include "naga/cospan/system.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace naga::cospan {

/// Run the subtree on the first n forwards, then never (n >= 1; n == 0
/// throws std::invalid_argument). Do-once is n = 1: construction-time
/// operators (LBVH, LRPIM, nonlocal-operator assembly). Users derive from it,
/// pass n to its constructor and override setup() like any System. An early
/// return means its components neither allocate nor release (r01 §4.4): when
/// the subtree is transient, its roots are allocated and released in the runs,
/// and after the n-th run its transient roots stay released. Roots kept by a
/// consumer outside persist.
class RunNSystem : public System {
    std::size_t remaining_;

public:
    explicit RunNSystem(std::size_t n) : remaining_{n} {
        if (n == 0) throw std::invalid_argument("naga::cospan: RunNSystem needs n >= 1");
    }

protected:
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        if (remaining_ == 0) return nullptr;   // nothing submitted
        --remaining_;
        return System::run_compute_forward();  // the default loop
    }
};

} // namespace naga::cospan
