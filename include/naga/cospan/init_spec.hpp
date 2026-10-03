#pragma once

// =============================================================================
// naga/cospan/init_spec.hpp — the type-erased initializer
// =============================================================================
//
// An InitSpec is the initializer a variable's shared metadata carries
// (docs/design-review-2026-10-03.md, sections 1 and 6): a function that runs on
// one device's data of the variable and returns the EVENT of the work it enqueued,
// plus the shape it supplies. It is split from init.hpp so that variable.hpp can
// hold one without depending on the cs::init factories, which depend on it.

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/cospan/metadata.hpp"

namespace naga::cospan {

namespace detail {
class VarMeta;
struct DeviceDataBase;
struct Runtime;

/// What an initializer runs against: the variable (its shape and path), the device data it writes,
/// the run-time argument, and the events its work must come after.
struct InitCtx {
    VarMeta&                 var;
    DeviceDataBase&          data;
    Runtime&                 rt;
    std::vector<sycl::event> deps;
};
}  // namespace detail

class InitSpec {
public:
    using Fn = std::function<sycl::event(detail::InitCtx&)>;

    InitSpec() = default;
    explicit InitSpec(Fn f, std::optional<InitShape> shape = std::nullopt) : fn_{std::move(f)}, shape_{std::move(shape)} {}

    /// The shape this initializer supplies to a free input's unspecified dimensions, if it has one.
    const std::optional<InitShape>& shape() const noexcept { return shape_; }

    /// Runs the initializer and returns the event of its work.
    sycl::event apply(detail::InitCtx& c) const { return fn_ ? fn_(c) : sycl::event{}; }

    explicit operator bool() const noexcept { return static_cast<bool>(fn_); }

private:
    Fn                       fn_;
    std::optional<InitShape> shape_;
};

}  // namespace naga::cospan
