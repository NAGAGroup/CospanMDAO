#pragma once

// =============================================================================
// naga/cospan/init.hpp — initial values of free input roots (r01 §4.4, D138)
// =============================================================================
//
// An initial value is METADATA on a free input root (an input root with no
// source), declared during setup() with ComponentBase::init_input(name, init)
// or System::init_input(path, init). The highest level wins; the framework
// applies it when it allocates the root from nothing. There are no init hooks
// and no initializer helpers: outputs are never initialized (§6.0).
//
//   cs::init::fill(v)           every element = v (converted to the root's dtype)
//   cs::init::zero()            every element = 0
//   cs::init::from_host(vec)    element i = vec[i]; vec.size() = the root's flat size
//   cs::init::custom<DT,R>(fn)  fn(queue&, Initializer<DT,R>): the function enqueues
//                               its own kernel on the queue
//   cs::init::custom_host<DT,R>(fn)  fn(HostViewMut<DT,R>&): host-side, committed on return

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "naga/cospan/dtype.hpp"
#include "naga/cospan/host_view.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/wrappers.hpp"

namespace naga::cospan {

namespace detail {

/// Calls f(std::type_identity<T>{}) with T the C++ element type of `dt`.
template <class F>
void dispatch_dtype(DType dt, F&& f) {
    switch (value_dtype(dt)) {
        case DType::f32: f(std::type_identity<float>{}); break;
        case DType::f64: f(std::type_identity<double>{}); break;
        case DType::i32: f(std::type_identity<int32_t>{}); break;
        case DType::u32: f(std::type_identity<uint32_t>{}); break;
        case DType::i64: f(std::type_identity<int64_t>{}); break;
        case DType::u64: f(std::type_identity<uint64_t>{}); break;
        default: f(std::type_identity<uint8_t>{}); break;
    }
}

/// Depth of a subsystem in the tree: the root System is 0, its children 1, and so on.
/// The declaring subsystem's depth decides which initial value wins (smallest depth).
inline std::size_t depth_of(const Subsystem& s) {
    if (s.path().empty()) return 0;
    std::size_t d = 1;
    for (const char c : s.path())
        if (c == '.') ++d;
    return d;
}

/// Zero a root's primary storage (ordered in the stream).
inline void zero_primary(IOEntry& root, sk::queue& q) {
    if (root.flat_size() == 0) return;
    const Placement pl = root.storage_placements().front();
    q.memset_in_order(pl.device, root.storage_on(pl), 0, root.flat_size() * dtype_size(root.dtype()));
}

/// A free input root that its owner has just allocated from nothing: its init, else zeros;
/// then the other placements follow the primary.
inline void apply_root_init(IOEntry& root, sk::queue& q) {
    if (root.flat_size() == 0) return;
    if (const auto& s = root.init_spec())
        s->apply(root, q);
    else
        zero_primary(root, q);
    if (root.storage_placements().size() > 1) root.sync_other_placements(q);
}

}  // namespace detail

namespace init {

/// Every element = value (converted to the root's element type).
template <class T>
InitSpec fill(T value) {
    return InitSpec{[value](detail::IOEntry& root, sk::queue& q) {
        const Placement pl = root.storage_placements().front();
        void*           p  = root.storage_on(pl);
        detail::dispatch_dtype(root.dtype(), [&](auto tag) {
            using E   = typename decltype(tag)::type;
            const E v = static_cast<E>(value);
            q.fill_in_order(pl.device, p, v, root.flat_size());
        });
    }};
}

/// Every element = 0.
inline InitSpec zero() {
    return InitSpec{[](detail::IOEntry& root, sk::queue& q) { detail::zero_primary(root, q); }};
}

/// Element i = values[i]; the size must equal the root's flat size when it is applied.
template <class T>
InitSpec from_host(std::vector<T> values) {
    return InitSpec{[values = std::move(values)](detail::IOEntry& root, sk::queue& q) {
        if (values.size() != root.flat_size())
            throw std::invalid_argument("naga::cospan: init::from_host has " + std::to_string(values.size()) +
                                        " values but '" + root.full_path() + "' has " +
                                        std::to_string(root.flat_size()) + " elements");
        detail::dispatch_dtype(root.dtype(), [&](auto tag) {
            using E = typename decltype(tag)::type;
            std::vector<E> converted(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) converted[i] = static_cast<E>(values[i]);
            detail::storage_of<E>(root).write_primary(q, std::span<const E>{converted.data(), converted.size()});
        });
    }};
}

/// A device-side initial value: `fn` gets the queue and the writable twin of the root and
/// enqueues its own kernel (or copy) on the queue. DT and Rank are the root's, spelled at the call:
///   cs::init::custom<DType::f32, 1>([](sk::queue& q, cs::Initializer<DType::f32, 1> out) { ... })
template <DType DT, int Rank>
InitSpec custom(std::function<void(sk::queue&, Initializer<DT, Rank>)> fn) {
    return InitSpec{[fn = std::move(fn)](detail::IOEntry& root, sk::queue& q) {
        auto* typed = dynamic_cast<detail::TypedEntry<DT, Rank>*>(&root);
        if (typed == nullptr)
            throw std::invalid_argument("naga::cospan: init::custom for '" + root.full_path() +
                                        "' was declared with the wrong dtype or rank");
        fn(q, typed->make_initializer());
    }};
}

/// A host-side initial value: `fn` edits a host copy of the root; it is written back on return.
template <DType DT, int Rank>
InitSpec custom_host(std::function<void(HostViewMut<DT, Rank>&)> fn) {
    return InitSpec{[fn = std::move(fn)](detail::IOEntry& root, sk::queue& q) {
        auto view = detail::HostViewAccess::view_mut<DT, Rank>(root, q);
        fn(view);
        view.commit();
    }};
}

}  // namespace init
}  // namespace naga::cospan
