#pragma once

// =============================================================================
// naga/cospan/metadata.hpp — I/O metadata and the declaration layer
//                            (r01 §3, §4, §8.1, §8.4)
// =============================================================================
//
// One detail::IOEntry per declared I/O: identity, shape, dimension records,
// the aliasing set (one ROOT, any number of PEERS), placement and, for roots,
// the per-device storage. Typed leaves (InputEntry, OutputEntry, AliasEntry)
// own the wrapper pointer; the storage layer is typed by element type only
// (StorageEntry<T>) because atomic and plain dtypes share storage and a peer's
// rank may differ from its root's.
//
// Declaration performs ZERO allocation. Allocation happens in allocate(), on
// roots only, through the route r01 §4.4 (as amended by D3) prescribes.

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/dtype.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/wrappers.hpp"

namespace naga::cospan {

namespace sk = ::naga::slabkit;

class System;
class ComponentBase;
class Model;

namespace detail {
class IOEntry;
struct EntryAccess;
}  // namespace detail

enum class IOKind : uint8_t { input, output };

// -----------------------------------------------------------------------------
// Dimension specification — DimSpec is a CONCEPT, not a type.
// -----------------------------------------------------------------------------

/// This dimension = dim `dim` of the upstream connection's shape. Inputs only.
struct FromSourceDim { uint8_t dim; };

/// This dimension = dim `dim` of sibling input `input`'s shape.
struct FromInputDim { std::string_view input; uint8_t dim; };

/// This dimension's extent = the VALUE of element `index` (flat) of sibling
/// input `input`, which SHALL have an integer value dtype.
struct FromInput { std::string_view input; std::size_t index; };

template <class T>
concept DimSpec = std::integral<T> || std::same_as<T, FromSourceDim> ||
                  std::same_as<T, FromInputDim> || std::same_as<T, FromInput>;

/// The whole-shape value-driven marker: on a rank-R declaration it expands to
/// FromInput{input,0}..{input,R-1}. Accepted alone by the declaration
/// overloads; expansion against the member's Rank happens in the declaration.
struct FromInputArray { std::string_view input; };

/// Sugar. The two shape helpers return arrays of specs (spread them with
/// std::apply); from_input_array returns the marker.
template <int Rank>
constexpr std::array<FromSourceDim, Rank> from_source_shape() {
    std::array<FromSourceDim, Rank> a{};
    for (int d = 0; d < Rank; ++d) a[d] = FromSourceDim{static_cast<uint8_t>(d)};
    return a;
}

template <int Rank>
constexpr std::array<FromInputDim, Rank> from_input_shape(std::string_view input) {
    std::array<FromInputDim, Rank> a{};
    for (int d = 0; d < Rank; ++d) a[d] = FromInputDim{input, static_cast<uint8_t>(d)};
    return a;
}

constexpr FromInputArray from_input_array(std::string_view input) { return {input}; }

// -----------------------------------------------------------------------------
// Views, slices and marks
// -----------------------------------------------------------------------------

/// A rectangular region of a root, in the ROOT's rank. offset/extents are
/// stored 1-padded to three dimensions (0 offset, extent 1 past `rank`).
struct SubarrayRef {
    int                        rank = 1;
    std::array<std::size_t, 3> offset{0, 0, 0};
    std::array<std::size_t, 3> extents{1, 1, 1};

    std::size_t flat_size() const { return extents[0] * extents[1] * extents[2]; }
};

/// subarray<2>({row, col}, {rows, cols}).
template <int R>
SubarrayRef subarray(const std::array<std::size_t, R>& offset,
                     const std::array<std::size_t, R>& extents) {
    static_assert(R >= 1 && R <= 3, "cospan: rank is 1..3");
    SubarrayRef s;
    s.rank = R;
    for (int d = 0; d < R; ++d) {
        s.offset[d]  = offset[d];
        s.extents[d] = extents[d];
    }
    return s;
}

/// What `connect` accepts beyond identity.
struct Slice { SubarrayRef src; };

/// Per-element or scalar bounds/scaling; host-side vectors sized to the marked
/// region's flat size (scalar forms broadcast).
struct Bounds { std::optional<std::vector<double>> lower, upper; };
struct Scaling { std::vector<double> ref{1.0}, ref0{0.0}; };

/// One mark on one root (Model::MarkRecord aliases this type).
struct MarkRecord {
    const detail::IOEntry*             root = nullptr;
    std::optional<SubarrayRef>         region;
    Bounds                             bounds;
    std::optional<std::vector<double>> equals;  // constraints only
    Scaling                            scaling;
};

// MemorySubstrate / Substrates live in subsystem.hpp (D7).

namespace detail {

/// Per-dimension storage form of a declared DimSpec.
struct DimRecord {
    enum class How : uint8_t { fixed, from_source_dim, from_input_dim, from_input };
    How         how = How::fixed;
    std::size_t extent = 0;  ///< fixed: declared; derived: last resolved (0 = unknown)
    std::string input;       ///< from_input_*: sibling input name (owned)
    uint8_t     dim = 0;     ///< from_source_dim / from_input_dim
    std::size_t value_index = 0;  ///< from_input
};

template <DimSpec S>
DimRecord make_dim_record(const S& s) {
    DimRecord r;
    if constexpr (std::integral<S>) {
        if constexpr (std::is_signed_v<S>) {
            if (s < 0) throw std::invalid_argument("naga::cospan: a fixed extent is negative");
        }
        r.how    = DimRecord::How::fixed;
        r.extent = static_cast<std::size_t>(s);
    } else if constexpr (std::same_as<S, FromSourceDim>) {
        r.how = DimRecord::How::from_source_dim;
        r.dim = s.dim;
    } else if constexpr (std::same_as<S, FromInputDim>) {
        r.how   = DimRecord::How::from_input_dim;
        r.input = std::string{s.input};
        r.dim   = s.dim;
    } else {
        r.how         = DimRecord::How::from_input;
        r.input       = std::string{s.input};
        r.value_index = s.index;
    }
    return r;
}

/// from_input_array against a member's Rank.
template <int Rank>
std::array<DimRecord, Rank> expand_from_input_array(const FromInputArray& m) {
    std::array<DimRecord, Rank> a{};
    for (int d = 0; d < Rank; ++d) {
        a[d].how         = DimRecord::How::from_input;
        a[d].input       = std::string{m.input};
        a[d].value_index = static_cast<std::size_t>(d);
    }
    return a;
}

/// Row-major strides of a 1-padded extents triple. For a rank-R entry the
/// first R strides are exactly rank-R row-major strides.
constexpr std::array<std::size_t, 3> contiguous_strides(const std::array<std::size_t, 3>& e) {
    return {e[1] * e[2], e[2], 1};
}

/// IOEntry — the metadata record behind one declared I/O. The virtual base is
/// the RTTI anchor: typed retrieval dynamic_casts it.
class IOEntry {
public:
    /// What a wrapper is pointed at: shape and stride (1-padded) and the
    /// element offset of the region from the root's storage.
    struct Geometry {
        std::array<std::size_t, 3> shape{1, 1, 1};
        std::array<std::size_t, 3> stride{1, 1, 1};
        std::size_t                offset = 0;
    };

    virtual ~IOEntry() = default;
    IOEntry(const IOEntry&)            = delete;
    IOEntry& operator=(const IOEntry&) = delete;

    // ---- identity & classification ---------------------------------------
    std::string_view name() const noexcept { return name_; }
    IOKind           kind() const noexcept { return io_kind_; }
    DType            dtype() const noexcept { return dtype_; }  ///< atomic-ness preserved
    int              rank() const noexcept { return rank_; }    ///< AUTHORITATIVE (§3.1)

    // ---- shape -------------------------------------------------------------
    sycl::range<3> extents() const { return sycl::range<3>{extents_[0], extents_[1], extents_[2]}; }
    /// Element step in the OWNING allocation (a sliced peer reports its root's).
    sycl::range<3> stride() const {
        const Geometry g = geometry();
        return sycl::range<3>{g.stride[0], g.stride[1], g.stride[2]};
    }
    std::size_t flat_size() const noexcept { return extents_[0] * extents_[1] * extents_[2]; }
    const DimRecord& dim(int d) const {
        if (d < 0 || d >= rank_) throw std::out_of_range("naga::cospan: dim index >= rank");
        return dims_[static_cast<std::size_t>(d)];
    }
    bool is_dynamic() const noexcept {
        for (const DimRecord& r : dims_)
            if (r.how != DimRecord::How::fixed) return true;
        return false;
    }

    /// The wrapper geometry of this entry: whole-root contiguous, or, for a
    /// peer with a view, the view's extents over the root's strides.
    Geometry geometry() const {
        Geometry g;
        if (view_) {
            const IOEntry* r = root_;
            if (view_->rank != r->rank_ || rank_ != r->rank_)
                throw std::invalid_argument(
                    "naga::cospan: '" + name_ + "' sees a subarray of '" + r->name_ +
                    "' and must have the root's rank (" + std::to_string(r->rank_) + ")");
            g.shape  = view_->extents;
            g.stride = contiguous_strides(r->extents_);
            for (int d = 0; d < 3; ++d) g.offset += view_->offset[d] * g.stride[d];
        } else {
            g.shape  = extents_;
            g.stride = contiguous_strides(extents_);
        }
        return g;
    }

    // ---- marks (semantics: §8): they ride on the root ----------------------
    const std::optional<MarkRecord>& design_var() const { return root_->design_var_; }
    const std::optional<MarkRecord>& constraint() const { return root_->constraint_; }
    const std::optional<MarkRecord>& objective() const { return root_->objective_; }

    // ---- the aliasing set (§4) -----------------------------------------------
    IOEntry*       root() noexcept { return root_; }
    const IOEntry* root() const noexcept { return root_; }
    bool           is_root() const noexcept { return root_ == this; }
    bool           owns_data() const noexcept { return is_root(); }
    /// Root only; empty otherwise. The root itself is not in its own list.
    std::span<IOEntry* const> peers() const noexcept {
        return is_root() ? std::span<IOEntry* const>{peers_} : std::span<IOEntry* const>{};
    }
    bool can_be_source() const noexcept { return can_be_source_; }
    bool can_be_target() const noexcept { return can_be_target_; }
    /// The subarray of the root this peer sees; whole-root when unset.
    std::optional<SubarrayRef> view() const { return view_; }

    // ---- placement and storage (§4.2, D3) ------------------------------------
    /// The owning subsystem's device.
    const sycl::device& device() const {
        if (!device_)
            throw std::logic_error("naga::cospan: '" + name_ + "' has no device yet");
        return *device_;
    }
    /// The kind of USM this root's storage is: device, or host for a host
    /// component's I/O (D3).
    sk::kind storage_kind() const noexcept { return storage_kind_; }
    bool     transient() const noexcept { return transient_; }

    /// THE ROOT HOLDS ALL STORAGE, PER DEVICE. \throws for a peer, or for a
    /// device the root has no allocation on. May be null for an empty extent.
    void* storage_on(const sycl::device& d) const {
        if (!is_root())
            throw std::logic_error("naga::cospan: storage_on is for roots; '" + name_ +
                                   "' is a peer of '" + root_->name_ + "'");
        return raw_storage_on(d);
    }
    /// Roots only; primary first.
    std::span<const sycl::device> storage_devices() const noexcept { return storage_devices_; }

    /// Arena route: a root, fully fixed, outside every transient subtree, in
    /// device-kind storage (arenas are device-kind).
    bool arena_eligible() const noexcept {
        return is_root() && !is_dynamic() && !transient_ && storage_kind_ == sk::kind::device;
    }
    /// The bytes ONE device's arena must hold for this root; 0 when the root
    /// takes another route. get_exec_info counts this once per device.
    std::size_t static_bytes() const noexcept {
        return arena_eligible() ? flat_size() * dtype_size(dtype_) : 0;
    }

    // ---- capabilities --------------------------------------------------------

    /// Automatic shape update: applies the DimRecords. FromSourceDim is
    /// resolved here; FromInputDim / FromInput extents are resolved by the
    /// owning Component first (set_resolved_extent) and applied here.
    virtual void update_shape() = 0;

    /// Resize. Growing past capacity reallocates through the root's allocator
    /// route; shrinking keeps capacity (usm_vector's contract).
    virtual void reshape(sycl::range<3> new_extents) = 0;

    /// INPUT PEERS ONLY, against root(): flat-size equality (the view's size
    /// for a slice, which must also lie inside the root). Throws.
    virtual void validate_connection() const = 0;

    /// ROOTS ONLY (no-op on peers): one usm_vector per device in
    /// storage_devices(), each from that device's substrate.
    virtual void allocate(sk::queue& q, Substrates& substrates) = 0;

    /// Write {data pointer, shape, stride} into the registered member wrapper.
    virtual void point_wrapper() = 0;

    /// The root's producer copies primary -> every other device's storage.
    virtual void sync_other_devices(sk::queue& q) = 0;

    /// Whether the ROOT of this entry's set has been allocated.
    virtual bool has_storage() const = 0;

protected:
    IOEntry(std::string name, IOKind k, DType dt, int rank, std::vector<DimRecord> dims,
            bool can_source, bool can_target)
        : name_{std::move(name)}
        , io_kind_{k}
        , dtype_{dt}
        , rank_{rank}
        , dims_{std::move(dims)}
        , can_be_source_{can_source}
        , can_be_target_{can_target} {
        if (rank_ < 1 || rank_ > 3)
            throw std::invalid_argument("naga::cospan: '" + name_ + "': rank must be 1..3");
        if (static_cast<int>(dims_.size()) != rank_)
            throw std::invalid_argument("naga::cospan: '" + name_ +
                                        "': the number of dimension specs must equal the rank");
        for (int d = 0; d < rank_; ++d) {
            const DimRecord& r = dims_[static_cast<std::size_t>(d)];
            if (r.how == DimRecord::How::from_source_dim && k == IOKind::output)
                throw std::invalid_argument("naga::cospan: '" + name_ +
                                            "': FromSourceDim is ill-formed for an output");
            if ((r.how == DimRecord::How::from_source_dim ||
                 r.how == DimRecord::How::from_input_dim) &&
                r.dim > 2)
                throw std::invalid_argument("naga::cospan: '" + name_ +
                                            "': a dimension index is out of range");
            extents_[static_cast<std::size_t>(d)] = r.extent;  // derived dims start at 0
        }
    }

    virtual void* raw_storage_on(const sycl::device& d) const = 0;

    /// The only writer of wrapper fields (friendship, r01 §2.1).
    template <class W, class P>
    static void write_wrapper(W& w, const Geometry& g, P* data) {
        w.shape_  = make_range<W::rank>(g.shape);
        w.stride_ = make_range<W::rank>(g.stride);
        w.data_   = data;
    }

    /// Registered in get_exec_info (System/Component); appends if absent, so
    /// the first registration is the primary.
    void register_device(const sycl::device& d) {
        if (!is_root()) throw std::logic_error("naga::cospan: only roots register devices");
        for (const sycl::device& e : storage_devices_)
            if (e == d) return;
        storage_devices_.push_back(d);
    }

    std::array<std::size_t, 3> extents_{1, 1, 1};  ///< 1-padded
    std::vector<DimRecord>     dims_;
    std::vector<sycl::device>  storage_devices_;   ///< roots; primary first
    std::optional<sycl::device> device_;
    sk::kind                   storage_kind_ = sk::kind::device;

private:
    /// The four structural operations and placement are System's (r01 §4.2);
    /// EntryAccess is the internal/test-only stand-in until System exists.
    friend class naga::cospan::System;
    friend class naga::cospan::ComponentBase;
    friend struct EntryAccess;

    /// merge(into, from): `from` and its peers become peers of `into`.
    static void merge(IOEntry& into, IOEntry& from) {
        if (!into.is_root() || !from.is_root())
            throw std::logic_error("naga::cospan: merge takes two set roots");
        if (&into == &from) return;
        into.peers_.reserve(into.peers_.size() + from.peers_.size() + 1);
        for (IOEntry* p : from.peers_) {
            p->root_ = &into;
            into.peers_.push_back(p);
        }
        from.peers_.clear();
        from.root_ = &into;
        into.peers_.push_back(&from);
    }

    std::string name_;
    IOKind      io_kind_;
    DType       dtype_;
    int         rank_;

    IOEntry*              root_ = this;
    std::vector<IOEntry*> peers_;  // meaningful iff root_ == this
    bool                  can_be_source_;
    bool                  can_be_target_;
    std::optional<SubarrayRef> view_;
    bool                  transient_ = false;

    std::optional<MarkRecord> design_var_, constraint_, objective_;
};

// -----------------------------------------------------------------------------
// Storage layer: everything that depends on the element type only
// -----------------------------------------------------------------------------

template <class T>
class StorageEntry : public IOEntry {
public:
    void update_shape() override {
        std::array<std::size_t, 3> next = extents_;
        for (int d = 0; d < rank(); ++d) {
            DimRecord& r = dims_[static_cast<std::size_t>(d)];
            switch (r.how) {
                case DimRecord::How::fixed: break;
                case DimRecord::How::from_source_dim: r.extent = source_extent(r.dim); break;
                case DimRecord::How::from_input_dim:
                case DimRecord::How::from_input: break;  // resolved by the Component
            }
            next[static_cast<std::size_t>(d)] = r.extent;
        }
        if (next != extents_) reshape(sycl::range<3>{next[0], next[1], next[2]});
    }

    void reshape(sycl::range<3> r) override {
        for (int d = rank(); d < 3; ++d)
            if (r[d] != 1)
                throw std::invalid_argument("naga::cospan: '" + std::string{name()} +
                                            "': extents past the rank must be 1");
        extents_ = {r[0], r[1], r[2]};
        if (is_root())
            for (auto& s : slots_) s->data.resize(flat_size());
    }

    void validate_connection() const override {
        if (kind() != IOKind::input || is_root()) return;
        const IOEntry& r        = *root();
        const Geometry g        = geometry();  // throws on a rank mismatch under a view
        std::size_t    expected = r.flat_size();
        if (const auto v = view()) {
            expected = v->flat_size();
            for (int d = 0; d < 3; ++d)
                if (v->offset[d] + v->extents[d] > r.extents()[d])
                    throw std::invalid_argument("naga::cospan: the subarray '" +
                                                std::string{name()} + "' sees is not inside '" +
                                                std::string{r.name()} + "'");
        }
        (void)g;
        if (flat_size() != expected)
            throw std::invalid_argument(
                "naga::cospan: flat size of input '" + std::string{name()} + "' (" +
                std::to_string(flat_size()) + ") differs from what it is connected to, '" +
                std::string{r.name()} + "' (" + std::to_string(expected) + ")");
    }

    void allocate(sk::queue& q, Substrates& subs) override {
        if (!is_root()) return;
        if (storage_devices_.empty()) register_device(device_ ? *device_ : q.get_device());
        for (const sycl::device& d : storage_devices_) {
            if (find_slot(d) != nullptr) continue;
            slots_.push_back(std::make_unique<Slot>(d, sk::usm_vector<T>(flat_size(), route(q, subs, d))));
        }
    }

    bool has_storage() const override {
        const auto* r = dynamic_cast<const StorageEntry<T>*>(root());
        return r != nullptr && !r->slots_.empty();
    }

    void sync_other_devices(sk::queue&) override {
        if (!is_root()) throw std::logic_error("naga::cospan: sync_other_devices is for roots");
        if (slots_.size() < 2) return;
        Slot& primary = primary_slot();
        for (auto& s : slots_)
            if (s.get() != &primary) s->data = primary.data;  // routed, ordered copy
    }

    /// A host copy of the WHOLE root, from the primary storage. Device kind:
    /// usm_vector::host_view (an owned copy, ordered after the stream's prior
    /// work). Host kind: queue.wait() then get_access().
    std::vector<T> read_primary(sk::queue& q) const {
        Slot& s = primary_slot();
        if (s.data.get_kind() == sk::kind::device) {
            auto hv = s.data.host_view();
            return std::vector<T>(hv.begin(), hv.end());
        }
        q.wait();
        auto acc = s.data.get_access();
        return std::vector<T>(acc.begin(), acc.end());
    }

    /// Replace the primary storage's contents from the host, then refresh the
    /// copies on the other devices.
    void write_primary(sk::queue& q, std::span<const T> src) {
        primary_slot().data.from_host(src);
        sync_other_devices(q);
    }

protected:
    StorageEntry(std::string name, IOKind k, DType dt, int rank, std::vector<DimRecord> dims,
                 bool can_source, bool can_target)
        : IOEntry{std::move(name), k, dt, rank, std::move(dims), can_source, can_target} {}

    void* raw_storage_on(const sycl::device& d) const override {
        Slot* s = find_slot(d);
        if (s == nullptr)
            throw std::logic_error("naga::cospan: '" + std::string{name()} +
                                   "' has no storage on that device");
        return s->data.data();
    }

private:
    struct Slot {
        Slot(sycl::device d, sk::usm_vector<T> v) : dev{std::move(d)}, data{std::move(v)} {}
        sycl::device      dev;
        sk::usm_vector<T> data;
    };

    Slot* find_slot(const sycl::device& d) const {
        for (const auto& s : slots_)
            if (s->dev == d) return s.get();
        return nullptr;
    }

    Slot& primary_slot() const {
        if (!is_root() || storage_devices_.empty())
            throw std::logic_error("naga::cospan: '" + std::string{name()} + "' has no storage");
        Slot* s = find_slot(storage_devices_.front());
        if (s == nullptr)
            throw std::logic_error("naga::cospan: '" + std::string{name()} + "' is not allocated");
        return *s;
    }

    std::size_t source_extent(uint8_t d) const {
        if (is_root())
            throw std::logic_error("naga::cospan: source-derived dimension on the unconnected input '" +
                                   std::string{name()} + "'");
        if (const auto v = view()) {
            if (d >= v->rank) throw std::out_of_range("naga::cospan: FromSourceDim past the source's rank");
            return v->extents[d];
        }
        if (d >= root()->rank())
            throw std::out_of_range("naga::cospan: FromSourceDim past the source's rank");
        return root()->extents()[d];
    }

    /// r01 §4.4 as amended by D3: host-kind roots take the OS route on their
    /// device; a fully fixed root outside a transient subtree takes the arena
    /// (never falling back: an arena is a budget); everything else the pool if
    /// the device has one, else the OS.
    sk::allocator<T> route(sk::queue& q, Substrates& subs, const sycl::device& d) const {
        if (storage_kind() != sk::kind::device) return sk::allocator<T>{q, storage_kind(), d};
        const auto it = subs.find(d);
        if (arena_eligible()) {
            if (it == subs.end() || !it->second.arena)
                throw std::logic_error("naga::cospan: '" + std::string{name()} +
                                       "' takes the arena route but its device has no arena");
            return sk::allocator<T>{q, *it->second.arena};
        }
        if (it != subs.end() && it->second.pool) return sk::allocator<T>{q, *it->second.pool};
        return sk::allocator<T>{q, sk::kind::device, d};
    }

    std::vector<std::unique_ptr<Slot>> slots_;  // parallel to storage_devices_ once allocated
};

/// The rank/dtype-typed layer: the wrapper pointer helpers.
template <DType DT, int Rank>
class TypedEntry : public StorageEntry<cpp_type_t<DT>> {
public:
    using value_type = cpp_type_t<DT>;

    /// The writable twin for init(): the same geometry as the member wrapper,
    /// over the root's storage on this entry's device.
    Initializer<DT, Rank> make_initializer() const {
        Initializer<DT, Rank> init;
        point(init);
        return init;
    }

protected:
    TypedEntry(std::string name, IOKind k, const std::array<DimRecord, Rank>& dims,
               bool can_source, bool can_target)
        : StorageEntry<value_type>{std::move(name), k, DT, Rank,
                                   std::vector<DimRecord>(dims.begin(), dims.end()), can_source,
                                   can_target} {}

    template <class W>
    void point(W& w) const {
        const IOEntry::Geometry g = this->geometry();
        auto* base = static_cast<value_type*>(this->root()->storage_on(this->device()));
        IOEntry::write_wrapper(w, g, base != nullptr ? base + g.offset : nullptr);
    }
};

/// InputEntry holds Input<DT,Rank>* to the member variable. Inputs: can be a
/// target, cannot be a source.
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
class InputEntry final : public TypedEntry<DT, Rank> {
public:
    InputEntry(std::string name, const std::array<DimRecord, Rank>& dims, Input<DT, Rank>* member)
        : TypedEntry<DT, Rank>{std::move(name), IOKind::input, dims, false, true}
        , member_{member} {}

    void point_wrapper() override {
        if (member_ != nullptr) this->point(*member_);
    }

    /// The registered member wrapper.
    Input<DT, Rank>& member() const { return *member_; }

private:
    Input<DT, Rank>* member_;
};

/// OutputEntry holds Output<DT,Rank>*. Outputs: can be a source, cannot be a
/// target.
template <DType DT, int Rank>
class OutputEntry final : public TypedEntry<DT, Rank> {
public:
    OutputEntry(std::string name, const std::array<DimRecord, Rank>& dims, Output<DT, Rank>* member)
        : TypedEntry<DT, Rank>{std::move(name), IOKind::output, dims, true, false}
        , member_{member} {}

    void point_wrapper() override {
        if (member_ != nullptr) this->point(*member_);
    }

    /// The registered member wrapper.
    Output<DT, Rank>& member() const { return *member_; }

private:
    Output<DT, Rank>* member_;
};

/// A System-level alias (promotion or tie alias): no member wrapper. Its
/// flags are given (a promotion snapshots its root's, a tie alias is a fresh
/// root); as a root it owns storage like any other.
template <DType DT, int Rank>
class AliasEntry final : public TypedEntry<DT, Rank> {
public:
    AliasEntry(std::string name, IOKind k, const std::array<DimRecord, Rank>& dims,
               bool can_source, bool can_target)
        : TypedEntry<DT, Rank>{std::move(name), k, dims, can_source, can_target} {}

    void point_wrapper() override {}
};

/// INTERNAL / TEST-ONLY. The structural and placement mutators are System's
/// and the Component's (friendship); until they exist this is the one
/// sanctioned way to drive them from outside (smoke tests). Deleted from the
/// tests when System lands.
struct EntryAccess {
    static void merge(IOEntry& into, IOEntry& from) { IOEntry::merge(into, from); }
    static void register_device(IOEntry& e, const sycl::device& d) { e.register_device(d); }
    static void set_device(IOEntry& e, const sycl::device& d) { e.device_ = d; }
    static void set_storage_kind(IOEntry& e, sk::kind k) { e.storage_kind_ = k; }
    static void set_transient(IOEntry& e, bool t) { e.transient_ = t; }
    static void set_view(IOEntry& e, std::optional<SubarrayRef> v) { e.view_ = std::move(v); }
    /// What the owning Component does for FromInputDim / FromInput dims.
    static void set_resolved_extent(IOEntry& e, int d, std::size_t extent) {
        if (d < 0 || d >= e.rank_) throw std::out_of_range("naga::cospan: dim index >= rank");
        e.dims_[static_cast<std::size_t>(d)].extent = extent;
    }
};

}  // namespace detail
}  // namespace naga::cospan
