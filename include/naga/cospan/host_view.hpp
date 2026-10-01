#pragma once

// =============================================================================
// naga/cospan/host_view.hpp — host-side copies of an entry (r01 §2.1, D139)
// =============================================================================
//
// HostView is a read-only host copy of one entry, shaped like it, taken at
// construction. It reads the root's PRIMARY storage (the root's producer's
// device), which is authoritative; the other devices' copies follow it. A peer
// with a view presents its region only (r01 §8.1). A root that is not present
// (its owner has not run yet, or a transient root was released) throws; a
// present size-0 root gives an empty view with the right shape.
//
// There is NO host write API (D139): initial values go through init_input
// (cs::init::* factories). HostViewMut survives only as the argument type of
// cs::init::custom_host: the framework builds one over a root it is
// initializing, hands it to the user's function and commits it on return.

#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/cospan/metadata.hpp"

namespace naga::cospan {

class Model;
class ComponentBase;

namespace detail {

struct HostViewAccess;

template <class T>
StorageEntry<T>& storage_of(IOEntry& e) {
    auto* s = dynamic_cast<StorageEntry<T>*>(e.root());
    if (s == nullptr)
        throw std::invalid_argument("naga::cospan: '" + e.full_path() +
                                    "': element type does not match its root's storage");
    return *s;
}

/// Element offset from the root's storage of the f-th element (row-major over
/// the geometry's shape) of a region.
inline std::size_t element_offset(std::size_t f, const IOEntry::Geometry& g) {
    std::size_t off = g.offset;
    for (int d = 2; d >= 0; --d) {
        off += (f % g.shape[d]) * g.stride[d];
        f /= g.shape[d];
    }
    return off;
}

inline std::size_t element_count(const IOEntry::Geometry& g) {
    return g.shape[0] * g.shape[1] * g.shape[2];
}

inline void check_entry(const IOEntry& e, DType dt, int rank) {
    if (!dtype_compatible(e.dtype(), dt) || e.rank() != rank)
        throw std::invalid_argument("naga::cospan: host view of '" + e.full_path() +
                                    "' asked for the wrong dtype or rank");
}

template <class T>
struct Snapshot {
    std::vector<T>             data;
    std::array<std::size_t, 3> shape;
};

/// The region an entry presents, copied to the host.
template <class T>
Snapshot<T> take_snapshot(IOEntry& entry, sk::queue& q) {
    const IOEntry::Geometry g     = entry.geometry();
    const std::vector<T>    whole = storage_of<T>(entry).read_primary(q);
    std::vector<T>          out(element_count(g));
    for (std::size_t f = 0; f < out.size(); ++f) {
        const std::size_t off = element_offset(f, g);
        if (off >= whole.size())
            throw std::out_of_range("naga::cospan: '" + entry.full_path() +
                                    "' presents elements past its root's storage");
        out[f] = whole[off];
    }
    return {std::move(out), g.shape};
}

/// Write an entry's host copy back into its root's primary storage (through the region, if it
/// has one) and refresh the other devices. Only cs::init::custom_host calls this, through
/// HostViewMut::commit, on a root its owner has just allocated.
template <class T>
void write_back(IOEntry& entry, sk::queue& q, std::span<const T> host) {
    const IOEntry::Geometry g    = entry.geometry();
    StorageEntry<T>&        root = storage_of<T>(entry);
    if (host.size() != element_count(g))
        throw std::logic_error("naga::cospan: host copy of '" + entry.full_path() +
                               "' no longer matches its shape");
    if (!entry.view()) {
        if (host.size() != entry.root()->flat_size())
            throw std::logic_error("naga::cospan: '" + entry.full_path() +
                                   "' no longer matches its root's size");
        root.write_primary(q, host);
        return;
    }
    std::vector<T> whole = root.read_primary(q);
    for (std::size_t f = 0; f < host.size(); ++f) {
        const std::size_t off = element_offset(f, g);
        if (off >= whole.size())
            throw std::out_of_range("naga::cospan: '" + entry.full_path() +
                                    "' presents elements past its root's storage");
        whole[off] = host[f];
    }
    root.write_primary(q, whole);
}

template <int Rank>
constexpr std::size_t flat_of(const sycl::id<Rank>& idx, const sycl::range<Rank>& shape) {
    std::size_t flat = 0;
    for (int d = 0; d < Rank; ++d) flat = flat * shape[d] + idx[d];
    return flat;
}

}  // namespace detail

/// HostView<DT, Rank> — a read-only host copy of one entry, shaped like it.
/// Host-side only; not a kernel capture.
template <DType DT, int Rank>
class HostView {
public:
    using value_type = cpp_type_t<DT>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    const value_type& operator[](sycl::id<Rank> idx) const {
        return host_[detail::flat_of<Rank>(idx, shape_)];
    }
    const value_type& operator[](std::size_t flat) const { return host_[flat]; }
    sycl::range<Rank> shape() const { return shape_; }
    std::size_t       size() const { return host_.size(); }
    std::span<const value_type> span() const { return {host_.data(), host_.size()}; }

private:
    HostView(detail::IOEntry& entry, sk::queue& q) {
        detail::check_entry(entry, DT, Rank);
        auto snap = detail::take_snapshot<value_type>(entry, q);
        host_     = std::move(snap.data);
        shape_    = detail::make_range<Rank>(snap.shape);
    }

    std::vector<value_type> host_;
    sycl::range<Rank>       shape_ = detail::filled_range<Rank>(0);
    friend struct detail::HostViewAccess;
    friend class Model;
    friend class ComponentBase;
    friend class System;
};

/// HostViewMut<DT, Rank> — the writable host copy handed to a cs::init::custom_host function.
/// Built by the framework only (the constructor is private); the framework commits it when the
/// function returns. It is NOT a host write API: there is none (D139).
template <DType DT, int Rank>
class HostViewMut {
public:
    using value_type = cpp_type_t<DT>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    HostViewMut(const HostViewMut&)            = delete;
    HostViewMut& operator=(const HostViewMut&) = delete;
    HostViewMut(HostViewMut&&) noexcept        = default;

    value_type& operator[](sycl::id<Rank> idx) { return host_[detail::flat_of<Rank>(idx, shape_)]; }
    value_type& operator[](std::size_t flat) { return host_[flat]; }
    sycl::range<Rank> shape() const { return shape_; }
    std::size_t       size() const { return host_.size(); }
    std::span<value_type> span() { return {host_.data(), host_.size()}; }

    /// Write the copy into the root's storage (primary, then the other placements).
    void commit() {
        detail::write_back<value_type>(*entry_, *q_, std::span<const value_type>{host_.data(), host_.size()});
    }

private:
    HostViewMut(detail::IOEntry& entry, sk::queue& q) : entry_{&entry}, q_{&q} {
        detail::check_entry(entry, DT, Rank);
        auto snap = detail::take_snapshot<value_type>(entry, q);
        host_     = std::move(snap.data);
        shape_    = detail::make_range<Rank>(snap.shape);
    }

    detail::IOEntry*        entry_ = nullptr;
    sk::queue*              q_     = nullptr;
    std::vector<value_type> host_;
    sycl::range<Rank>       shape_ = detail::filled_range<Rank>(0);
    friend struct detail::HostViewAccess;
};

namespace detail {

/// INTERNAL: the friend of both views (Model builds HostViews; cs::init::custom_host builds the
/// HostViewMut of the root it initializes).
struct HostViewAccess {
    template <DType DT, int Rank>
    static HostView<DT, Rank> view(IOEntry& e, sk::queue& q) {
        return HostView<DT, Rank>{e, q};
    }
    template <DType DT, int Rank>
    static HostViewMut<DT, Rank> view_mut(IOEntry& e, sk::queue& q) {
        return HostViewMut<DT, Rank>{e, q};
    }
};

}  // namespace detail
}  // namespace naga::cospan
