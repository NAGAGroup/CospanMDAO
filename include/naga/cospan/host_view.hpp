#pragma once

// =============================================================================
// naga/cospan/host_view.hpp — host views of a variable
// =============================================================================
//
// A HOST VIEW is an instance of the variable on a host placement (docs/design-review-2026-10-03.md, section 5): like
// an input, output or alias it holds the variable's shared metadata and its host memory (a SlabKit container) by
// shared pointer, and the metadata holds that memory weakly.
//
//   - While any view exists, the host copy is one of the variable's live placements: EVERY FORWARD pushes one copy to
//     it, after the producer's launch, like any device copy. (Each live view costs one copy per forward.) When the
//     last view goes away, the host memory is freed; keep a view alive to avoid reallocating it.
//   - A view can be obtained only AFTER A RUN (Model::view_input and friends throw before the first forward). It starts
//     as a copy of a live placement and never initializes anything: a view made after Model::update_input shows the
//     old values until the next run.
//   - CALL sync() BEFORE READING: it waits on the host container's write events (its pushes and copies).
//     Reading without syncing, or while work is running, is the caller's responsibility.
//   - A view OUTLIVES THE MODEL: it keeps the metadata and its host memory, and the container's handle keeps the queue
//     alive. When the model goes, every device copy is freed (each waiting on its own events first).
//
// Host data reaches an input only through an initializer (cs::init::* factories; Model::update_input).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/cospan/dtype.hpp"
#include "naga/cospan/instance.hpp"
#include "naga/cospan/variable.hpp"
#include "naga/cospan/wrappers.hpp"

namespace naga::cospan {

class Model;

namespace detail {

/// The host data of a view instance, typed.
template <class T>
DeviceData<T>& host_store(const std::shared_ptr<Instance>& inst) {
    if (!inst || !inst->data)
        throw std::logic_error("naga::cospan: a host view has no data (its variable was freed)");
    return static_cast<DeviceData<T>&>(*inst->data);
}

/// Waits on what a read of the view must come after (section 5): the host container's write list, which holds the
/// pushes (each copy waited on the producer's writes) and the copy that made it.
template <class T>
void sync_view(const std::shared_ptr<Instance>& inst) {
    sycl::event::wait(host_store<T>(inst).read_deps());
}

}  // namespace detail

/// HostView<DT, Rank> — a host view of a DENSE variable, shaped like it. Host-side only; not a kernel capture.
template <DType DT, int Rank>
class HostView {
public:
    using value_type = cpp_type_t<value_dtype(DT)>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    /// Waits on the variable's write events and on this host copy's own events: the data is then the last run's.
    void sync() const { detail::sync_view<value_type>(inst_); }

    std::size_t        size() const { return store().size(); }
    const value_type*  data() const { return static_cast<const value_type*>(store().raw()); }
    std::span<const value_type> span() const { return {data(), size()}; }
    const value_type&  operator[](std::size_t flat) const { return data()[flat]; }

    /// The variable's shape. \throws std::invalid_argument for a ragged variable (use HostRagged).
    sycl::range<Rank> shape() const {
        const detail::ResolvedShape& s = inst_->meta->shape;
        if (!s.known || s.ragged())
            throw std::invalid_argument("naga::cospan: '" + inst_->meta->display_path() +
                                        "' is ragged (or has no shape yet) and has no rectangular host view; use view_ragged_input / "
                                        "view_ragged_output");
        return detail::make_range<Rank>(s.extents);
    }

    /// Row-major.
    const value_type& operator()(std::size_t i) const
        requires(Rank == 1)
    {
        return data()[i];
    }
    const value_type& operator()(std::size_t i, std::size_t j) const
        requires(Rank == 2)
    {
        return data()[i * shape()[1] + j];
    }
    const value_type& operator()(std::size_t i, std::size_t j, std::size_t k) const
        requires(Rank == 3)
    {
        const auto sh = shape();
        return data()[(i * sh[1] + j) * sh[2] + k];
    }

private:
    friend class Model;
    explicit HostView(std::shared_ptr<detail::Instance> inst) : inst_{std::move(inst)} {}

    detail::DeviceData<value_type>& store() const { return detail::host_store<value_type>(inst_); }

    std::shared_ptr<detail::Instance> inst_;
};

/// HostRagged<DT, Rank> — a host view of a ragged variable, Rank 2 only: the flat data (the blocks end to end), N blocks
/// and their offsets. Also taken of a DENSE variable (the blocks are its rows).
template <DType DT, int Rank>
class HostRagged {
public:
    static_assert(Rank == 2, "cospan: a ragged host view is rank 2 only");
    using value_type = cpp_type_t<value_dtype(DT)>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    void sync() const { detail::sync_view<value_type>(inst_); }

    std::size_t        size() const { return store().size(); }
    const value_type*  data() const { return static_cast<const value_type*>(store().raw()); }
    std::span<const value_type> span() const { return {data(), size()}; }
    const value_type&  operator[](std::size_t flat) const { return data()[flat]; }

    /// N, the number of blocks (dimension 0).
    std::size_t n() const { return inst_->meta->shape.extents[0]; }

    /// The N + 1 block offsets: block i is data()[offsets[i], offsets[i + 1]). From the host copy's own offsets array
    /// when it has one (a ragged shape), else the constant blocks of the dense shape.
    std::vector<std::uint64_t> block_offsets() const {
        const detail::ResolvedShape& s = inst_->meta->shape;
        if (const auto* rd = dynamic_cast<const detail::RaggedDeviceData<value_type>*>(inst_->data.get()))
            if (s.ragged() && rd->offs && rd->offs->size() == s.extents[0] + 1)
                return std::vector<std::uint64_t>(rd->offs->data(), rd->offs->data() + rd->offs->size());
        return s.block_offsets();
    }

    /// Block i, as a span over its elements (row-major within the block).
    std::span<const value_type> block(std::size_t i) const {
        const std::vector<std::uint64_t> offs = block_offsets();
        if (i + 1 >= offs.size()) throw std::out_of_range("naga::cospan: HostRagged::block past the last block");
        return {data() + offs[i], static_cast<std::size_t>(offs[i + 1] - offs[i])};
    }

private:
    friend class Model;
    explicit HostRagged(std::shared_ptr<detail::Instance> inst) : inst_{std::move(inst)} {}

    detail::DeviceData<value_type>& store() const { return detail::host_store<value_type>(inst_); }

    std::shared_ptr<detail::Instance> inst_;
};

}  // namespace naga::cospan
