#pragma once

// =============================================================================
// naga/cospan/variable.hpp — a variable's shared metadata and its per-device data
// =============================================================================
//
// One VarMeta per variable, held by shared_ptr by EVERY instance that belongs to it
// (docs/design-review-2026-10-03.md, section 1). It carries what the variable IS: dtype, rank, the shape
// definition, the resolved shape, the initializer, the stale flag, the event lists, and weak pointers
// to the instances that share it and to each device's data. It holds no memory and keeps nothing alive:
//
//   - the metadata lives while any instance holding it exists;
//   - a device's data (a DeviceData, a SlabKit container) lives while any instance on that device holds
//     it, and the metadata's entry for it is weak;
//   - "initialized" means some device has live data: there is no flag for it, only the stale flag of
//     section 5 (a variable that is not materialized is always stale).
//
// A ragged variable is a different metadata type (RaggedVar) on the same idea: its device data also holds
// the device copies of the block offsets and the per-index extents.
//
// RULES ENFORCED HERE. materialize (below) is the one function that brings data into being. It never
// runs outside a Runtime (runtime.hpp), allocates on every device that has an instance, and initializes
// or copies, as section 3 says. Merging variables, checking ranks and deciding which initializer wins
// are the business of the connect, tie and promote calls (a later phase): VarMeta is passive.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/slabkit/slabkit.hpp"

#include "naga/cospan/dispatch.hpp"
#include "naga/cospan/dtype.hpp"
#include "naga/cospan/init_spec.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/runtime.hpp"

namespace naga::cospan {

namespace sk = ::naga::slabkit;

namespace detail {

// -----------------------------------------------------------------------------
// Device data
// -----------------------------------------------------------------------------

/// Where one copy of a variable's data lives: a device and a USM kind. A host component's port is
/// (its device, host); its producer's is (the same device, device). Two keys on one device are two storages.
struct DataKey {
    sycl::device dev;
    sk::kind     kind = sk::kind::device;
    friend bool operator==(const DataKey& a, const DataKey& b) { return a.dev == b.dev && a.kind == b.kind; }
};

/// One device's data of a variable, type-erased. Held by shared_ptr by every instance on that device; the
/// VarMeta keeps a weak pointer.
struct DeviceDataBase {
    DataKey key;

    explicit DeviceDataBase(DataKey k) : key{std::move(k)} {}
    virtual ~DeviceDataBase() = default;

    /// What a submission that touches this data must come after (the container's own events).
    virtual std::vector<sycl::event> events() const = 0;
    /// Hands the container an event of outside work over its data, so a later mutation or the free waits for it.
    virtual void register_event(const sycl::event& e) = 0;
    virtual void*       raw()        = 0;
    virtual std::size_t size() const = 0;
    /// Makes this data a copy of `src` (the same element type, another placement), after `deps`: the container's
    /// assign, so it resizes to the source. Returns the event of the copy. A ragged variable's index arrays follow.
    virtual sycl::event copy_from(DeviceDataBase& src, const std::vector<sycl::event>& deps) = 0;
};

template <class T>
struct DeviceData : DeviceDataBase {
    sk::usm_vector<T> vec;

    DeviceData(DataKey k, sk::usm_vector<T> v) : DeviceDataBase{std::move(k)}, vec{std::move(v)} {}

    std::vector<sycl::event> events() const override { return vec.events(); }
    void                     register_event(const sycl::event& e) override { vec.register_event(e); }
    void*                    raw() override { return vec.data(); }
    std::size_t              size() const override { return vec.size(); }
    sycl::event              copy_from(DeviceDataBase& src, const std::vector<sycl::event>& deps) override {
        return vec.assign(static_cast<DeviceData<T>&>(src).vec, std::span<const sycl::event>{deps});
    }
};

/// A ragged variable's device data: the elements, the N + 1 block offsets and, for each per-index dimension
/// d >= 1, the extents of dimension d (indexed d - 1). The index arrays are containers of their own, so their
/// events count as the data's.
template <class T>
struct RaggedDeviceData : DeviceData<T> {
    std::optional<sk::usm_vector<std::uint64_t>>                offs;
    std::array<std::optional<sk::usm_vector<std::uint64_t>>, 2> per;

    using DeviceData<T>::DeviceData;

    std::vector<sycl::event> events() const override {
        std::vector<sycl::event> out = DeviceData<T>::events();
        const auto               add = [&](const std::optional<sk::usm_vector<std::uint64_t>>& v) {
            if (!v) return;
            const std::vector<sycl::event> e = v->events();
            out.insert(out.end(), e.begin(), e.end());
        };
        add(offs);
        for (const auto& p : per) add(p);
        return out;
    }
    void register_event(const sycl::event& e) override {
        DeviceData<T>::register_event(e);
        if (offs) offs->register_event(e);  // the kernel reads the index arrays too
        for (auto& p : per)
            if (p) p->register_event(e);
    }
    /// The elements, then the index arrays that both sides have (each after the source array's own events). The
    /// returned event is the elements' copy; the index copies are tracked by their containers, which `events()` reports.
    sycl::event copy_from(DeviceDataBase& src, const std::vector<sycl::event>& deps) override {
        const sycl::event e = DeviceData<T>::copy_from(src, deps);
        auto&             s = static_cast<RaggedDeviceData<T>&>(src);
        const auto        assign = [](std::optional<sk::usm_vector<std::uint64_t>>& to,
                               const std::optional<sk::usm_vector<std::uint64_t>>& from) {
            if (!to || !from) return;
            const std::vector<sycl::event> ev = from->events();
            to->assign(*from, std::span<const sycl::event>{ev});
        };
        assign(offs, s.offs);
        for (std::size_t k = 0; k < per.size(); ++k) assign(per[k], s.per[k]);
        return e;
    }
};

// -----------------------------------------------------------------------------
// Device columns and the resolved shape
// -----------------------------------------------------------------------------

/// Integer value dtype (not f32 / f64, plain or atomic): what a size may be read from.
constexpr bool is_integer_value_dtype(DType dt) noexcept {
    const DType v = value_dtype(dt);
    return v != DType::f32 && v != DType::f64;
}

/// A driving column's per-index extents and their exclusive scan, on the DRIVER's device (docs/design-review-2026-10-03.md,
/// section 6): built once per write of the driving variable, by one device task, and read by the consumers' index arrays
/// (and by nothing on the host but the total and a negative-value flag). The vectors are `mutable` so that a holder of a
/// const pointer can register the event of its own read on them.
struct DeviceColumn {
    DataKey                                          key;      ///< where the column lives (the driver's device)
    std::size_t                                      n = 0;    ///< the number of indices
    mutable std::optional<sk::usm_vector<std::uint64_t>> extents;     ///< n entries
    mutable std::optional<sk::usm_vector<std::uint64_t>> offs;        ///< n + 1 entries: the exclusive scan
    std::optional<sk::usm_vector<std::uint64_t>>     host_total;  ///< host kind, 2 elements: [total, negative flag]
    std::uint64_t                                    total   = 0;     ///< read from host_total once the task is done
    std::uint64_t                                    version = ~0ull; ///< the driver's write_count() when it was built
    sycl::event                                      ev;              ///< the task that built it
};

// -----------------------------------------------------------------------------
// The resolved shape
// -----------------------------------------------------------------------------

/// The shape a variable has NOW, as numbers. A per-index dimension reports extent 0 and is sourced from the host
/// (`per_index[d]`) OR from the device (`dev_per[d]`, a DeviceColumn), never both. The device path covers ONE
/// per-index dimension (the common CSR case); with two, normalize() brings them to the host.
struct ResolvedShape {
    bool                                    known = false;
    std::array<std::size_t, 3>              extents{1, 1, 1};  ///< 1-padded past the rank
    std::array<std::vector<std::size_t>, 3> per_index{};
    std::array<bool, 3>                     per{};
    std::array<std::shared_ptr<const DeviceColumn>, 3> dev_per{};      ///< device-sourced per-index dimensions
    std::array<std::uint64_t, 3>            dev_version{};            ///< the column's version when this shape was built
    std::uint64_t                           total = 0;                ///< a ragged shape's number of elements

    bool ragged() const noexcept { return per[1] || per[2]; }
    bool device_sourced() const noexcept { return dev_per[1] || dev_per[2]; }

    /// The sum of the block sizes, from the host arrays (no device-sourced dimension).
    std::size_t host_sum() const {
        std::size_t sum = 0;
        for (std::size_t i = 0; i < extents[0]; ++i) {
            std::size_t block = 1;
            for (std::size_t d = 1; d < 3; ++d) block *= per[d] ? per_index[d][i] : extents[d];
            sum += block;
        }
        return sum;
    }

    /// The number of elements: the product of the extents, or, for a ragged shape, `total` when a dimension is
    /// device-sourced, else the sum of the block sizes (the same arithmetic as InitShape::elements()). 0 while the
    /// shape is not known.
    std::size_t elements() const {
        if (!known) return 0;
        if (!ragged()) return extents[0] * extents[1] * extents[2];
        if (device_sourced()) return static_cast<std::size_t>(total);
        return host_sum();
    }

    /// Sets `total` of a host-only ragged shape (a dense or device-sourced one is left alone).
    void settle_host_total() {
        if (!device_sourced()) total = ragged() ? static_cast<std::uint64_t>(host_sum()) : 0;
    }

    /// The N + 1 block offsets of a ragged shape: offsets[i + 1] = offsets[i] + the size of block i. Host arrays only.
    std::vector<std::uint64_t> block_offsets() const {
        if (device_sourced()) throw std::logic_error("naga::cospan: its block offsets live on the device (a value-driven ragged shape)");
        std::vector<std::uint64_t> offs(extents[0] + 1, 0);
        for (std::size_t i = 0; i < extents[0]; ++i) {
            std::uint64_t block = 1;
            for (std::size_t d = 1; d < 3; ++d) block *= per[d] ? per_index[d][i] : extents[d];
            offs[i + 1] = offs[i] + block;
        }
        return offs;
    }

    /// Dimension `d`'s per-index extents on the host: its host array, or the device column copied down (waiting on it).
    std::vector<std::size_t> per_index_on_host(std::size_t d, Runtime& rt) const {
        if (!dev_per[d]) return per_index[d];
        const DeviceColumn&      col = *dev_per[d];
        std::vector<std::uint64_t> h(col.n);
        std::vector<sycl::event>   deps{col.ev};
        const std::vector<sycl::event> own = col.extents->events();
        deps.insert(deps.end(), own.begin(), own.end());
        if (col.n != 0) rt.q.copy(col.key.dev, deps, static_cast<const std::uint64_t*>(col.extents->data()), h.data(), col.n).wait();
        return std::vector<std::size_t>(h.begin(), h.end());
    }

    /// The fallback for TWO per-index dimensions: when one is device-sourced, every device-sourced dimension becomes a host
    /// array (the device path covers one per-index dimension, the common CSR case; a later phase may extend it), and the
    /// total is recomputed from the host arrays.
    void normalize(Runtime& rt) {
        if (!device_sourced() || !(per[1] && per[2])) return;
        for (std::size_t d = 1; d < 3; ++d) {
            if (!dev_per[d]) continue;
            per_index[d] = per_index_on_host(d, rt);
            dev_per[d].reset();
            dev_version[d] = 0;
        }
        settle_host_total();
    }

    friend bool operator==(const ResolvedShape& a, const ResolvedShape& b) {
        return a.known == b.known && a.extents == b.extents && a.per_index == b.per_index && a.per == b.per &&
               a.dev_per == b.dev_per && a.dev_version == b.dev_version && a.total == b.total;
    }
};

/// Finishes a ragged shape that was built dimension by dimension: per-index dimensions get extent 0; with two per-index
/// dimensions a device-sourced one is brought to the host (normalize); and the total is set: the column's total times
/// the scalar dimensions when exactly one dimension is device-sourced, else the sum of the host arrays. `who` names the
/// variable in messages. One place, used by every path that builds a value-driven shape.
inline void finish_ragged_total(ResolvedShape& s, Runtime& rt, const std::string& who) {
    for (std::size_t u = 0; u < 3; ++u)
        if (s.per[u]) s.extents[u] = 0;
    if (!s.ragged()) {
        s.total = 0;
        return;
    }
    s.normalize(rt);
    if (!s.device_sourced()) {
        s.settle_host_total();
        return;
    }
    const std::size_t dd = s.dev_per[1] ? 1 : 2;  // exactly one per-index dimension, device-sourced
    const DeviceColumn& col = *s.dev_per[dd];
    if (col.n != s.extents[0])
        throw std::invalid_argument("naga::cospan: '" + who + "': per-index dimension " + std::to_string(dd) + " has " +
                                    std::to_string(col.n) + " extents but dimension 0 has " + std::to_string(s.extents[0]));
    std::uint64_t prod = 1;
    for (std::size_t d = 1; d < 3; ++d)
        if (d != dd) prod *= s.extents[d];
    s.dev_version[dd] = col.version;
    s.total           = col.total * prod;
}

// -----------------------------------------------------------------------------
// Event lists
// -----------------------------------------------------------------------------

inline bool event_complete(const sycl::event& e) {
    return e.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
}

/// The events of the writes and reads of a variable, in two generations (docs/design-review-2026-10-03.md,
/// section 3). A WRITER waits on the last generation's writers and readers and never on another writer of its
/// own forward, so tied outputs (atomic or disjoint) run concurrently. A READER waits on this forward's writers if
/// there are any, else on the last generation's: a reader that runs before any writer this forward (a cycle) reads
/// the previous value. The generation rolls over when a writer asks in a new forward.
///
/// Container operations (resize, cross-device copy, ragged index uploads) put their events on the CONTAINERS, not
/// here; a submission also depends on the events of its own device's containers.
class EventLists {
public:
    /// The events a write in `forward` must come after. Rolls the generation first when `forward` is new.
    std::vector<sycl::event> write_deps(std::uint64_t forward) {
        roll(forward);
        return joined(prev_writes_, prev_reads_);
    }
    /// What write_deps would return, without rolling.
    std::vector<sycl::event> peek_write_deps(std::uint64_t forward) const {
        return forward != gen_forward_ ? joined(cur_writes_, cur_reads_) : joined(prev_writes_, prev_reads_);
    }
    /// The events a read must come after.
    std::vector<sycl::event> read_deps() const { return cur_has_writer_ ? cur_writes_ : prev_writes_; }

    void record_write(sycl::event e) {
        ++write_count_;
        cur_writes_.push_back(std::move(e));
        cur_has_writer_ = true;
        prune(cur_writes_);
    }
    void record_read(sycl::event e) {
        cur_reads_.push_back(std::move(e));
        prune(cur_reads_);
    }

    /// How many writes were ever recorded: a consumer's cached copy of a value is valid while this has not changed.
    std::uint64_t write_count() const { return write_count_; }

    /// Every write still tracked (this forward's and the last's), for a host sync.
    std::vector<sycl::event> all_writes() const { return joined(cur_writes_, prev_writes_); }

private:
    static std::vector<sycl::event> joined(const std::vector<sycl::event>& a, const std::vector<sycl::event>& b) {
        std::vector<sycl::event> out;
        out.reserve(a.size() + b.size());
        out.insert(out.end(), a.begin(), a.end());
        out.insert(out.end(), b.begin(), b.end());
        return out;
    }
    static void prune(std::vector<sycl::event>& v) { std::erase_if(v, event_complete); }

    void roll(std::uint64_t forward) {
        if (forward == gen_forward_) return;
        prev_writes_ = std::move(cur_writes_);
        prev_reads_  = std::move(cur_reads_);
        cur_writes_.clear();
        cur_reads_.clear();
        cur_has_writer_ = false;
        gen_forward_    = forward;
        prune(prev_writes_);
        prune(prev_reads_);
    }

    std::vector<sycl::event> cur_writes_, cur_reads_, prev_writes_, prev_reads_;
    bool                     cur_has_writer_ = false;
    std::uint64_t            write_count_    = 0;
    std::uint64_t            gen_forward_    = ~std::uint64_t{0};
};

// -----------------------------------------------------------------------------
// The shared metadata
// -----------------------------------------------------------------------------

class Instance;  // instance.hpp

// What VarMeta needs to know of an instance, defined in instance.hpp (which includes this header).
inline DataKey             key_of(const Instance& i);
inline void                give_data(Instance& i, std::shared_ptr<DeviceDataBase> d);
inline bool                is_output_port(const Instance& i);
inline std::string         path_of(const Instance& i);
inline const sycl::device& device_of(const Instance& i);

class VarMeta;
inline std::shared_ptr<VarMeta> meta_of(const Instance& i);  // the instance's variable
inline void check_init_shape(const VarMeta& v, const ResolvedShape& r, const InitShape& s);

class VarMeta : public std::enable_shared_from_this<VarMeta> {
public:
    VarMeta(DType dt, int r, std::vector<DimRecord> def) : dtype{dt}, rank{r}, definition{std::move(def)} {}

    VarMeta(const VarMeta&)            = delete;
    VarMeta& operator=(const VarMeta&) = delete;
    virtual ~VarMeta()                 = default;

    DType dtype;  ///< atomic-ness preserved: the DType of the first declaration
    int   rank;

    /// The shape definition: one record per dimension. Which definition wins when variables merge is decided by
    /// connect, tie and promote (a later phase).
    std::vector<DimRecord>    definition;
    std::optional<AliasShape> alias_shape;  ///< a tie's shape rule
    ResolvedShape             shape;

    /// The initializer, and what the highest-level-wins rule needs to decide (in the calls that declare it,
    /// a later phase): the depth of the declaring subsystem and its path.
    std::optional<InitSpec> init;
    std::size_t             init_depth = 0;
    std::string             init_by;
    /// Where the initialization runs (set by tie_inputs): else the device with the most instances.
    std::optional<sycl::device> init_device;

    /// See is_stale(): this flag alone says nothing for a variable that is not materialized.
    bool stale_ = true;

    /// Connected inside one fused component; a later phase may keep such an intermediate in local memory (section 4).
    bool internal = false;

    /// The forward in which refresh_alias_shape last ran (once per forward); ~0 = never.
    std::uint64_t alias_forward_ = ~0ull;

    // ---- shape-driving state (section 6) ---------------------------------------------------------
    // Set when a consumer reads this variable's VALUES to size something (driving_value). The producer then does the
    // shape work in its postcompute: one small copy of the driving elements to a host buffer, once per variable,
    // however many consumers it sizes.
    bool                 shape_driving_ = false;
    std::set<std::size_t> value_indices_;                 ///< the elements consumers asked for
    std::uint64_t        shape_write_count_ = ~0ull;      ///< events.write_count() when the copy was queued
    sycl::event          shape_event_;                    ///< the copy of the driving elements
    bool                 shape_valid_ = false;            ///< the host buffer holds the values of the last write

    /// Per-index extents (section 6, ragged): for each requested column of this variable, the extents and their scan on
    /// the DEVICE, built once per write of this variable (do_column_work), so the consumers' offsets never touch the host.
    std::map<std::size_t, std::shared_ptr<DeviceColumn>> columns_;
    std::set<std::size_t>                                columns_requested_;

    /// What driving_values returns: the column (shared with this variable's cache) and the version it has now.
    struct PerIndexSource {
        std::shared_ptr<const DeviceColumn> col;
        std::uint64_t                       version = 0;
    };

    /// The instances that share this metadata (weak; used by merges and to answer where the variable lives).
    std::vector<std::weak_ptr<Instance>> instances;
    /// Each device's data, weak: the data lives while an instance on that device holds it.
    std::vector<std::pair<DataKey, std::weak_ptr<DeviceDataBase>>> data;
    EventLists                                                      events;

    // ---- instances ---------------------------------------------------------------------------------
    void add_instance(std::weak_ptr<Instance> w) { instances.push_back(std::move(w)); }

    /// The live instances; expired entries are dropped from the list.
    std::vector<std::shared_ptr<Instance>> live_instances() {
        std::vector<std::shared_ptr<Instance>> out;
        std::erase_if(instances, [&](const std::weak_ptr<Instance>& w) {
            std::shared_ptr<Instance> s = w.lock();
            if (!s) return true;
            out.push_back(std::move(s));
            return false;
        });
        return out;
    }

    /// The path of the first live instance (for messages), else "<variable>".
    std::string display_path() const {
        for (const auto& w : instances)
            if (const std::shared_ptr<Instance> s = w.lock()) return path_of(*s);
        return "<variable>";
    }

    /// An instance that writes this variable exists: an output port.
    bool has_producer() const {
        for (const auto& w : instances)
            if (const std::shared_ptr<Instance> s = w.lock())
                if (is_output_port(*s)) return true;
        return false;
    }

    std::size_t instance_count_on(const sycl::device& dev) const {
        std::size_t n = 0;
        for (const auto& w : instances)
            if (const std::shared_ptr<Instance> s = w.lock())
                if (device_of(*s) == dev) ++n;
        return n;
    }

    /// The device with the most live instances of this variable; ties go to the earliest in the list.
    /// Counted from the instance list, so it also works before materialization. It is the device host views
    /// will use for their copy (section 5).
    sycl::device busiest_device() const {
        std::vector<std::pair<sycl::device, std::size_t>> counts;
        for (const auto& w : instances) {
            const std::shared_ptr<Instance> s = w.lock();
            if (!s) continue;
            const sycl::device& d  = device_of(*s);
            bool                hit = false;
            for (auto& c : counts)
                if (c.first == d) {
                    ++c.second;
                    hit = true;
                    break;
                }
            if (!hit) counts.emplace_back(d, 1);
        }
        if (counts.empty()) return default_device();
        const std::pair<sycl::device, std::size_t>* best = &counts.front();
        for (const auto& c : counts)
            if (c.second > best->second) best = &c;
        return best->first;
    }

    sycl::device pick_init_device() const { return init_device ? *init_device : busiest_device(); }

    // ---- state -------------------------------------------------------------------------------------
    /// Some device has live data.
    bool is_materialized() const {
        for (const auto& p : data)
            if (!p.second.expired()) return true;
        return false;
    }
    /// A variable that is not materialized is always stale; a materialized one is stale after mark_stale()
    /// until its initializer has run again (docs/design-review-2026-10-03.md, section 5).
    bool is_stale() const { return !is_materialized() || stale_; }
    void mark_stale() { stale_ = true; }

    /// How a dimension was declared, for messages.
    std::string declared_rule(int d) const {
        const DimRecord& r = definition[static_cast<std::size_t>(d)];
        switch (r.how) {
            case DimRecord::How::fixed: return "fixed";
            case DimRecord::How::unspecified: return "unspecified";
            case DimRecord::How::from_input_dim: return "FromInputDim{'" + r.input + "', " + std::to_string(r.dim) + "}";
            case DimRecord::How::from_input: return "FromInput{'" + r.input + "', " + std::to_string(r.value_index) + "}";
            case DimRecord::How::per_index_host: return "PerIndex";
            case DimRecord::How::per_index_from_input:
                return "FromInputValues{'" + r.input + "', " + std::to_string(r.value_index) + "}";
        }
        return "?";
    }

    /// A variable nothing produces and whose shape is not known takes it from its definition (fixed dimensions) and
    /// its initializer's shape (unspecified ones, per-index dimensions honored), and the initializer's shape is
    /// checked against the result. \throws std::logic_error when it cannot be resolved here.
    void resolve_free_shape() {
        const InitShape* is = (init && init->shape()) ? &*init->shape() : nullptr;
        ResolvedShape    s;
        s.known = true;
        for (int d = 0; d < rank; ++d) {
            const auto       u = static_cast<std::size_t>(d);
            const DimRecord& r = definition[u];
            switch (r.how) {
                case DimRecord::How::fixed: s.extents[u] = r.extent; break;
                case DimRecord::How::per_index_host:
                    s.per[u]       = true;
                    s.per_index[u] = r.per_index;
                    s.extents[u]   = 0;
                    break;
                case DimRecord::How::unspecified:
                    if (is == nullptr || is->rank != rank) throw shape_unknown();
                    if (d >= 1 && is->per[u]) {
                        s.per[u]       = true;
                        s.per_index[u] = is->per_index[u];
                        s.extents[u]   = 0;
                    } else {
                        s.extents[u] = is->extents[u];
                    }
                    break;
                default:  // derived from a sibling input: the component's precompute resolves it and calls set_shape
                    throw shape_unknown();
            }
        }
        for (int d = 1; d < rank; ++d) {
            const auto u = static_cast<std::size_t>(d);
            if (s.per[u] && s.per_index[u].size() != s.extents[0])
                throw std::invalid_argument("naga::cospan: '" + display_path() + "': per-index dimension " + std::to_string(d) +
                                            " has " + std::to_string(s.per_index[u].size()) + " extents but dimension 0 has " +
                                            std::to_string(s.extents[0]));
        }
        if (is != nullptr) check_init_shape(*this, s, *is);
        s.settle_host_total();
        shape = std::move(s);
    }

    /// "'<path>': its shape is not known yet".
    std::logic_error shape_unknown() const {
        return std::logic_error("naga::cospan: '" + display_path() + "': its shape is not known yet");
    }

    /// The data on `key`, if it is live.
    virtual std::shared_ptr<DeviceDataBase> data_on(const DataKey& key) const {
        for (const auto& p : data)
            if (p.first == key) return p.second.lock();
        return {};
    }

    virtual bool is_ragged() const noexcept { return false; }

    /// The driving-value accessor (section 6): the VALUE of element `index` (flat) of this variable, which sizes a
    /// consumer's dimension. The one accessor a consumer reads driving values through. It marks the variable
    /// shape-driving; a variable that is stale (or not materialized: a free input) is materialized first, which runs its
    /// initializer; the producer usually queued the small copy right after its launch, so the wait on it is usually
    /// already over (section 6). A negative value is an error.
    std::size_t driving_value(Runtime& rt, std::size_t index) {
        shape_driving_ = true;
        value_indices_.insert(index);
        if (is_stale()) materialize(rt);
        if (!shape_valid_ || shape_write_count_ != events.write_count() || index >= shape_host_size()) do_shape_work(rt);
        shape_event_.wait();
        return read_shape_host(index);
    }
    /// The same for per-index extents: the values of column `column` (rank 1 [N]: column 0; rank 2 [N, k]) as N extents
    /// and their exclusive scan, computed ON THE DEVICE by the producer's shape work (section 6); only the total is
    /// copied to the host, and the host waits on that small copy alone. A negative value is an error.
    PerIndexSource driving_values(Runtime& rt, std::size_t column) {
        shape_driving_ = true;
        columns_requested_.insert(column);
        if (is_stale()) materialize(rt);
        const auto it = columns_.find(column);
        if (it == columns_.end() || it->second->version != events.write_count()) do_column_work(rt, column);
        DeviceColumn& col = *columns_.at(column);
        col.ev.wait();
        col.total = (*col.host_total).data()[0];
        if ((*col.host_total).data()[1] != 0)
            throw std::invalid_argument("naga::cospan: '" + display_path() + "': column " + std::to_string(column) +
                                        " holds a negative value; a size cannot be negative");
        return PerIndexSource{columns_.at(column), col.version};
    }

    /// "2x(per-index)x4": a shape as text, for messages.
    static std::string alias_shape_str(const std::array<std::size_t, 3>& ext, const std::array<bool, 3>& per, int rank) {
        std::string out;
        for (int d = 0; d < rank; ++d) {
            if (d > 0) out += "x";
            out += per[static_cast<std::size_t>(d)] ? std::string{"(per-index)"} : std::to_string(ext[static_cast<std::size_t>(d)]);
        }
        return out;
    }

    /// A tie alias's shape, evaluated by its rule NOW and given to the variable (section 2 and section 6): a no-op for a
    /// variable without an alias rule, and once per forward. The rules read other variables, so the shape is reactive:
    /// it can change between forwards. `unspecified` is not evaluated here: the connection or the initializer supplies
    /// it (resolve_free_shape). A source that does not exist any more, or whose shape is not known, is an error.
    void refresh_alias_shape(Runtime& rt) {
        if (!alias_shape || alias_forward_ == rt.forward) return;
        const AliasShape& rule = *alias_shape;
        if (rule.kind == AliasShape::Kind::unspecified) return;
        ResolvedShape s;
        s.known = true;
        switch (rule.kind) {
            case AliasShape::Kind::fixed:
                for (int d = 0; d < rank; ++d) s.extents[static_cast<std::size_t>(d)] = rule.extents[static_cast<std::size_t>(d)];
                break;
            case AliasShape::Kind::unspecified: return;
            case AliasShape::Kind::from_output_shape:
            case AliasShape::Kind::from_input_shape: {
                const std::shared_ptr<VarMeta> up = alias_source(rt, rule.src, /*need_shape=*/true);
                for (int d = 0; d < rank; ++d) copy_alias_dim(s, static_cast<std::size_t>(d), up->shape, static_cast<std::size_t>(d), rule.src);
                break;
            }
            case AliasShape::Kind::from_output_values: {
                const std::shared_ptr<VarMeta> up = alias_source(rt, rule.src, /*need_shape=*/false);
                for (int d = 0; d < rank; ++d)
                    s.extents[static_cast<std::size_t>(d)] = up->driving_value(rt, static_cast<std::size_t>(d));
                break;
            }
            case AliasShape::Kind::dims:
                for (int d = 0; d < rank; ++d) {
                    const auto      u  = static_cast<std::size_t>(d);
                    const AliasDim& sp = rule.dim_specs[u];
                    switch (sp.how) {
                        case AliasDim::How::fixed: s.extents[u] = sp.extent; break;
                        case AliasDim::How::per_index_host:
                            s.per[u]       = true;
                            s.per_index[u] = sp.values;
                            break;
                        // an input source is read through its variable, as an output is
                        case AliasDim::How::output_value:
                            s.extents[u] = alias_source(rt, sp.src, false)->driving_value(rt, sp.index);
                            break;
                        case AliasDim::How::output_values: {
                            const PerIndexSource src = alias_source(rt, sp.src, false)->driving_values(rt, sp.index);
                            s.per[u]         = true;
                            s.dev_per[u]     = src.col;
                            s.dev_version[u] = src.version;
                            break;
                        }
                        case AliasDim::How::output_dim: {
                            const std::shared_ptr<VarMeta> up = alias_source(rt, sp.src, true);
                            copy_alias_dim(s, u, up->shape, sp.index, sp.src);
                            break;
                        }
                    }
                }
                break;
        }
        // every host-sourced per-index array has one entry per index of dimension 0 (a device column's is checked below)
        for (int d = 1; d < rank; ++d) {
            const auto u = static_cast<std::size_t>(d);
            if (s.per[u] && !s.dev_per[u] && s.per_index[u].size() != s.extents[0])
                throw std::invalid_argument("naga::cospan: tie alias '" + display_path() + "': per-index dimension " + std::to_string(d) +
                                            " has " + std::to_string(s.per_index[u].size()) + " extents but dimension 0 has " +
                                            std::to_string(s.extents[0]) + " (the shape is " + alias_shape_str(s.extents, s.per, rank) + ")");
        }
        // per-index dimensions have no scalar extent; the total (a device column's times the scalar dimensions, else the
        // host sum); two per-index dimensions are brought to the host
        finish_ragged_total(s, rt, display_path());
        set_shape(rt, s);
        alias_forward_ = rt.forward;
    }

    /// Brings the variable's data into being on every device that has an instance, then initializes it or
    /// copies it, as section 3 says. A no-op when everything is live and not stale.
    virtual void materialize(Runtime& rt) = 0;

    /// Gives the variable a new resolved shape: resizes every live data in place (contents are kept, and the
    /// variable never becomes stale). A no-op when the shape is unchanged.
    virtual void set_shape(Runtime& rt, const ResolvedShape& s) = 0;

    /// Makes a placement that is MISSING live (a host view's, a new instance's) by copying from a live one. It NEVER
    /// initializes, even when the variable is stale (section 5: a view made after update_input shows the old values
    /// until the next run). \throws std::logic_error when nothing holds data.
    virtual void attach_missing(Runtime& rt) = 0;

    /// The shape work (section 6): copies the driving elements to a small host buffer, event-ordered. A no-op unless the
    /// variable is shape-driving and materialized.
    virtual void do_shape_work(Runtime& rt) = 0;

    /// Builds (or rebuilds) the device column of `column`: its per-index extents and their scan, by one device task.
    virtual void do_column_work(Runtime& rt, std::size_t column) = 0;

    /// The host buffer of driving elements: its size, and element `index` converted to a size.
    virtual std::size_t shape_host_size() const = 0;
    virtual std::size_t read_shape_host(std::size_t index) const = 0;

private:
    /// The variable an alias rule reads, through the instance it names (weak: a deleted child's expires). With
    /// `need_shape` its shape must be known: a free source that is not materialized yet is materialized first (that is
    /// where its initializer's shape is applied).
    std::shared_ptr<VarMeta> alias_source(Runtime& rt, const AliasSrc& src, bool need_shape) {
        const std::shared_ptr<Instance> up = src.inst.lock();
        if (!up)
            throw std::logic_error("naga::cospan: '" + display_path() + "': its shape comes from '" + src.path +
                                   "', which does not exist right now");
        std::shared_ptr<VarMeta> m = meta_of(*up);
        if (need_shape) {
            if (!m->shape.known && !m->has_producer()) m->materialize(rt);
            if (!m->shape.known)
                throw std::logic_error("naga::cospan: '" + display_path() + "': its shape comes from '" + src.path +
                                       "', whose shape is not known yet");
        }
        return m;
    }

    /// Slot `to` of `out` takes dimension `d` of an upstream shape (per-index too, if that dim is).
    void copy_alias_dim(ResolvedShape& out, std::size_t to, const ResolvedShape& up, std::size_t d, const AliasSrc& src) const {
        if (up.per[d]) {
            if (to == 0)
                throw std::invalid_argument("naga::cospan: a per-index dimension of '" + src.path +
                                            "' cannot become dimension 0 of the tie alias '" + display_path() + "'");
            out.per[to]         = true;
            out.per_index[to]   = up.per_index[d];
            out.dev_per[to]     = up.dev_per[d];  // a device-sourced dimension stays on the device
            out.dev_version[to] = up.dev_version[d];
            out.extents[to]     = 0;
        } else {
            out.extents[to] = up.extents[d];
        }
    }
};

inline std::string extents_str(const std::vector<std::size_t>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + std::to_string(v[i]);
    return out + "]";
}

/// The shape an init carries against the variable's resolved shape: the rank and EVERY dimension (fixed,
/// inherited, value-driven, per-index element by element). A scalar extent c equals a per-index array of c's.
/// Bare data (from_host(vec)) describes only a rank-1 variable; its size is checked when it is written.
inline void check_init_shape(const VarMeta& v, const ResolvedShape& r, const InitShape& s) {
    if (s.flat && v.rank != 1) return;
    const std::string name = v.display_path();
    if (s.rank != v.rank)
        throw std::invalid_argument("naga::cospan: init for '" + name + "' has rank " + std::to_string(s.rank) + " but '" + name +
                                    "' resolved rank " + std::to_string(v.rank));
    const std::size_t n = r.extents[0];
    for (int d = 0; d < v.rank; ++d) {
        const auto u        = static_cast<std::size_t>(d);
        const bool init_per = d >= 1 && s.per[u];
        const bool var_per  = d >= 1 && r.per[u];
        std::vector<std::size_t> init_v, var_v;
        if (d == 0) {
            init_v = {s.extents[0]};
            var_v  = {n};
        } else {
            init_v = init_per ? s.per_index[u] : std::vector<std::size_t>(s.extents[0], s.extents[u]);
            var_v  = var_per ? r.per_index[u] : std::vector<std::size_t>(n, r.extents[u]);
        }
        if (init_v == var_v) continue;
        const std::string init_str = init_per ? extents_str(init_v) : std::to_string(s.extents[u]);
        const std::string var_str  = var_per ? extents_str(var_v) : std::to_string(r.extents[u]);
        throw std::invalid_argument("naga::cospan: init for '" + name + "' has dimension " + std::to_string(d) + " = " + init_str +
                                    " but '" + name + "' resolved dimension " + std::to_string(d) + " (" + v.declared_rule(d) +
                                    ") = " + var_str);
    }
}

// -----------------------------------------------------------------------------
// The typed metadata
// -----------------------------------------------------------------------------

/// A dense variable of element type T (rank at run time, as the storage layer is: atomic and plain dtypes share
/// storage).
template <class T>
class Var : public VarMeta {
public:
    using Data = DeviceData<T>;

    Var(DType dt, int r, std::vector<DimRecord> def) : VarMeta{dt, r, std::move(def)} {}

    void materialize(Runtime& rt) override {
        const std::vector<std::shared_ptr<Instance>> insts = live_instances();
        if (insts.empty()) return;
        std::erase_if(data, [](const auto& p) { return p.second.expired(); });

        // The keys that have an instance, and those among them without live data.
        const std::vector<DataKey> needed  = keys_of(insts);
        const std::vector<DataKey> missing = missing_of(needed);

        const bool stale = is_stale();
        if (!stale && missing.empty()) return;  // (c) nothing to do

        if (!shape.known) {
            if (has_producer()) throw shape_unknown();
            resolve_free_shape();
        } else if (stale && !has_producer()) {
            refresh_free_shape(rt);  // update_input may have supplied a new shape
        }

        // A live data to copy from, if the variable is live somewhere already.
        std::shared_ptr<Data> donor = live_donor();

        // Allocate on every missing key. The new data are held here until every instance has taken its own.
        const std::vector<std::shared_ptr<Data>> fresh = allocate_missing(rt, insts, missing);

        // (a) Live elsewhere and not stale: a placement that comes back is a COPY of a live one, never recomputed.
        if (!stale) {
            if (donor)
                for (const auto& d : fresh) {
                    const std::vector<sycl::event> dev = donor->events();
                    const sycl::event e = d->vec.assign(donor->vec, std::span<const sycl::event>{dev});
                    donor->register_event(e);  // a later writer of the donor's copy waits for the read
                    copy_index(*d, *donor);
                }
            return;
        }

        // (b) Stale (or never materialized): initialize on the init device, then copy to the other placements.
        for (const auto& d : fresh) fill_index(rt, *d, d->key);
        const sycl::device idev = pick_init_device();
        DataKey            ikey = needed.front();
        bool               found = false;
        for (const DataKey& k : needed)
            if (k.dev == idev && k.kind == sk::kind::device) {
                ikey  = k;
                found = true;
                break;
            }
        if (!found)
            for (const DataKey& k : needed)
                if (k.dev == idev) {
                    ikey  = k;
                    found = true;
                    break;
                }
        const std::shared_ptr<Data> idata = std::static_pointer_cast<Data>(data_on(ikey));

        if (!has_producer()) {  // an output is never initialized: its producer writes it
            shape_valid_ = false;  // an initializer is about to run: the copy of the driving elements is out of date
            std::vector<sycl::event> deps = events.write_deps(rt.forward);
            {
                const std::vector<sycl::event> own = idata->events();
                deps.insert(deps.end(), own.begin(), own.end());
            }
            sycl::event e;
            if (init) {
                if (init->shape()) check_init_shape(*this, shape, *init->shape());
                InitCtx ctx{*this, *idata, rt, deps};
                e = init->apply(ctx);
            } else {
                e = idata->vec.fill(T{}, std::span<const sycl::event>{deps});
            }
            events.record_write(e);
            idata->register_event(e);
            const std::array<sycl::event, 1> after{e};
            for (const DataKey& k : needed) {
                if (k == ikey) continue;
                const std::shared_ptr<Data> other = std::static_pointer_cast<Data>(data_on(k));
                other->vec.assign(idata->vec, std::span<const sycl::event>{after});
            }
        }
        stale_ = false;
    }

    void attach_missing(Runtime& rt) override {
        if (!is_materialized())
            throw std::logic_error("naga::cospan: '" + display_path() +
                                   "' has no data: nothing that uses it has run, or its data was freed");
        std::erase_if(data, [](const auto& p) { return p.second.expired(); });
        const std::vector<std::shared_ptr<Instance>> insts   = live_instances();
        const std::vector<DataKey>                   missing = missing_of(keys_of(insts));
        if (missing.empty()) return;
        const std::shared_ptr<Data> donor = live_donor();
        for (const auto& d : allocate_missing(rt, insts, missing)) {
            const std::vector<sycl::event> dev = donor->events();
            const sycl::event              e   = d->copy_from(*donor, dev);  // a ragged variable's index arrays follow
            donor->register_event(e);                                         // a later writer of the donor's copy waits for the read
        }
    }

    void set_shape(Runtime& rt, const ResolvedShape& s) override {
        ResolvedShape next = s;
        next.known         = true;
        if (shape == next) return;
        const std::vector<sycl::event> base = events.peek_write_deps(rt.forward);
        shape                               = std::move(next);
        for (auto& entry : data) {
            const std::shared_ptr<DeviceDataBase> b = entry.second.lock();
            if (!b) continue;
            const std::shared_ptr<Data> d = std::static_pointer_cast<Data>(b);
            std::vector<sycl::event>    deps = base;
            const std::vector<sycl::event> own = d->events();
            deps.insert(deps.end(), own.begin(), own.end());
            d->vec.resize(shape.elements(), std::span<const sycl::event>{deps});  // the prefix is kept
            fill_index(rt, *d, d->key);
        }
    }

    /// The producer's shape work (section 6): the small host copy of the driving elements asked for by value, and the
    /// device column of every column asked for by per-index values.
    void do_shape_work(Runtime& rt) override {
        if (!shape_driving_ || !is_materialized()) return;
        if (!value_indices_.empty()) do_value_work(rt);
        for (const std::size_t c : columns_requested_) do_column_work(rt, c);
    }

    /// The driving elements, copied to the host buffer.
    void do_value_work(Runtime& rt) {
        const std::size_t n = *value_indices_.rbegin() + 1;
        if (n > shape.elements())
            throw std::invalid_argument("naga::cospan: '" + display_path() + "' drives a size from element " +
                                        std::to_string(*value_indices_.rbegin()) + " but has only " +
                                        std::to_string(shape.elements()) + " elements");
        // The source: a live device-kind data if there is one, else any live data.
        std::shared_ptr<Data> src;
        for (const auto& p : data) {
            const std::shared_ptr<DeviceDataBase> b = p.second.lock();
            if (!b) continue;
            if (!src || (b->key.kind == sk::kind::device && src->key.kind != sk::kind::device)) src = std::static_pointer_cast<Data>(b);
        }
        // The host buffer: reallocated only when it must grow.
        if (!shape_host_ || shape_host_->size() < n) shape_host_.emplace(n, rt.allocator_for<T>(src->key.dev, sk::kind::host));
        std::vector<sycl::event> deps = events.read_deps();
        for (const std::vector<sycl::event>& v : {src->events(), shape_host_->events()}) deps.insert(deps.end(), v.begin(), v.end());
        const sycl::event e = rt.q.copy(src->key.dev, deps, static_cast<const T*>(src->raw()), shape_host_->data(), n);
        src->register_event(e);
        shape_host_->register_event(e);
        events.record_read(e);  // a later writer waits until the copy has read
        shape_event_       = e;
        shape_write_count_ = events.write_count();
        shape_valid_       = true;
    }

    std::size_t shape_host_size() const override { return shape_host_ ? shape_host_->size() : 0; }

    std::size_t read_shape_host(std::size_t index) const override {
        if (!shape_host_ || index >= shape_host_->size())
            throw std::logic_error("naga::cospan: '" + display_path() + "': element " + std::to_string(index) +
                                   " was not copied to the host");
        const T v = shape_host_->data()[index];
        if constexpr (std::is_signed_v<T>) {
            if (v < 0)
                throw std::invalid_argument("naga::cospan: '" + display_path() + "': element " + std::to_string(index) + " is " +
                                            std::to_string(v) + "; a size cannot be negative");
        }
        return static_cast<std::size_t>(v);
    }

    /// ONE serial device task for now (section 6: "possibly optimized later"): for i in [0, n): v = data[i * k + column];
    /// a negative value sets the flag (and counts as 0); extents[i] = v; offs[i + 1] = offs[i] + extents[i]; finally the
    /// total and the flag go to the host buffer. The producer queues this in its postcompute (do_shape_work), once per
    /// write; a consumer that asks first does it itself. The column held in columns_ keeps the containers alive.
    void do_column_work(Runtime& rt, std::size_t column) override {
        if (!is_materialized()) return;
        if (!is_integer_value_dtype(dtype))
            throw std::invalid_argument("naga::cospan: '" + display_path() + "' drives per-index extents by value and must have an integer dtype");
        if (rank > 2)
            throw std::invalid_argument("naga::cospan: '" + display_path() +
                                        "' drives per-index extents by value and must be rank 1 [N] or rank 2 [N,k]");
        const std::size_t n = shape.extents[0];
        const std::size_t k = rank == 2 ? shape.extents[1] : 1;
        if (column >= k)
            throw std::invalid_argument("naga::cospan: '" + display_path() + "' drives per-index extents from column " +
                                        std::to_string(column) + " but has only " + std::to_string(k) + " column(s)");
        // The source: a live device-kind data if there is one, else any live data.
        std::shared_ptr<Data> src;
        for (const auto& p : data) {
            const std::shared_ptr<DeviceDataBase> b = p.second.lock();
            if (!b) continue;
            if (!src || (b->key.kind == sk::kind::device && src->key.kind != sk::kind::device)) src = std::static_pointer_cast<Data>(b);
        }
        std::shared_ptr<DeviceColumn>& slot = columns_[column];
        if (!slot) slot = std::make_shared<DeviceColumn>();
        DeviceColumn&      col = *slot;
        const sycl::device dev = src->key.dev;
        // The vectors are reused while they live on the same device; their size follows n (a resize within capacity).
        if (!(col.key.dev == dev) || !col.extents || !col.offs || !col.host_total) {
            col.key        = DataKey{dev, sk::kind::device};
            col.extents.emplace(n, rt.allocator_for<std::uint64_t>(dev, sk::kind::device));
            col.offs.emplace(n + 1, rt.allocator_for<std::uint64_t>(dev, sk::kind::device));
            col.host_total.emplace(2, rt.allocator_for<std::uint64_t>(dev, sk::kind::host));
        } else {
            if (col.extents->size() != n) col.extents->resize(n, std::span<const sycl::event>{col.extents->events()});
            if (col.offs->size() != n + 1) col.offs->resize(n + 1, std::span<const sycl::event>{col.offs->events()});
        }
        col.n = n;

        std::vector<sycl::event> deps = events.read_deps();
        for (const std::vector<sycl::event>& v : {src->events(), col.extents->events(), col.offs->events(), col.host_total->events()})
            deps.insert(deps.end(), v.begin(), v.end());

        const T*             in  = static_cast<const T*>(src->raw());
        std::uint64_t*       ex  = col.extents->data();
        std::uint64_t*       of  = col.offs->data();
        std::uint64_t*       tot = col.host_total->data();
        const std::size_t    nn = n, kk = k, cc = column;
        const sycl::event e = rt.uq.single_task(dev, deps, [=]() {
            std::uint64_t flag = 0;
            of[0]              = 0;
            for (std::size_t i = 0; i < nn; ++i) {
                const T       v = in[i * kk + cc];
                std::uint64_t x = 0;
                if constexpr (std::is_signed_v<T>) {
                    if (v < T{0}) flag = 1;
                    else x = static_cast<std::uint64_t>(v);
                } else {
                    x = static_cast<std::uint64_t>(v);
                }
                ex[i]     = x;
                of[i + 1] = of[i] + x;
            }
            tot[0] = of[nn];
            tot[1] = flag;
        });
        src->register_event(e);
        col.extents->register_event(e);
        col.offs->register_event(e);
        col.host_total->register_event(e);
        events.record_read(e);  // a later writer waits until the task has read
        col.ev      = e;
        col.version = events.write_count();
    }

protected:
    /// Allocates this variable's data on `key` (its elements; a ragged variable adds its index arrays).
    virtual std::shared_ptr<Data> make_data(Runtime& rt, const DataKey& key) {
        return std::make_shared<Data>(key, sk::usm_vector<T>(shape.elements(), rt.allocator_for<T>(key.dev, key.kind)));
    }
    /// A ragged variable's index arrays follow the host shape. Nothing for a dense one.
    virtual void fill_index(Runtime&, Data&, const DataKey&) {}
    /// A ragged variable's index arrays copied from a donor's.
    virtual void copy_index(Data&, Data&) {}

private:
    /// The distinct keys of the instances' placements.
    static std::vector<DataKey> keys_of(const std::vector<std::shared_ptr<Instance>>& insts) {
        std::vector<DataKey> keys;
        for (const auto& i : insts) {
            const DataKey k = key_of(*i);
            if (std::find(keys.begin(), keys.end(), k) == keys.end()) keys.push_back(k);
        }
        return keys;
    }
    /// Those among `keys` without live data.
    std::vector<DataKey> missing_of(const std::vector<DataKey>& keys) const {
        std::vector<DataKey> missing;
        for (const DataKey& k : keys)
            if (!data_on(k)) missing.push_back(k);
        return missing;
    }
    /// The first live data, if any.
    std::shared_ptr<Data> live_donor() const {
        for (const auto& p : data)
            if (std::shared_ptr<DeviceDataBase> b = p.second.lock()) return std::static_pointer_cast<Data>(b);
        return nullptr;
    }
    /// Allocates the data of every missing key and hands each to the instances on that key (the shared step of
    /// materialize and attach_missing). The metadata keeps them weakly: the caller holds the returned pointers.
    std::vector<std::shared_ptr<Data>> allocate_missing(Runtime& rt, const std::vector<std::shared_ptr<Instance>>& insts,
                                                        const std::vector<DataKey>& missing) {
        std::vector<std::shared_ptr<Data>> fresh;
        for (const DataKey& k : missing) {
            std::shared_ptr<Data> d = make_data(rt, k);
            data.emplace_back(k, std::weak_ptr<DeviceDataBase>{d});
            fresh.push_back(d);
            for (const auto& i : insts)
                if (key_of(*i) == k) give_data(*i, d);
        }
        return fresh;
    }

    /// A free variable whose initializer carries a shape, and whose definition has an unspecified dimension: the shape
    /// follows the initializer (update_input may have replaced it). Only when every dimension is fixed, unspecified or
    /// per-index from the host: a derived dimension is the component's to resolve.
    void refresh_free_shape(Runtime& rt) {
        if (!init || !init->shape()) return;
        bool has_unspecified = false;
        for (const DimRecord& r : definition) {
            if (r.how == DimRecord::How::unspecified) has_unspecified = true;
            else if (r.how != DimRecord::How::fixed && r.how != DimRecord::How::per_index_host) return;
        }
        if (!has_unspecified) return;
        const ResolvedShape keep = shape;
        resolve_free_shape();
        const ResolvedShape next = shape;
        shape                    = keep;
        if (!(next == keep)) set_shape(rt, next);
    }

    std::optional<sk::usm_vector<T>> shape_host_;  ///< the driving elements, on the host (section 6)
};

/// A ragged variable: the same, with the block offsets and per-index extents on the device beside the elements
/// (a different metadata type on the same idea; section 1). A ragged variable whose shape turns out dense keeps
/// empty index arrays and offsets of null, as a dense port expects.
template <class T>
class RaggedVar final : public Var<T> {
public:
    using Data = typename Var<T>::Data;

    RaggedVar(DType dt, int r, std::vector<DimRecord> def) : Var<T>{dt, r, std::move(def)} {}

    bool is_ragged() const noexcept override { return true; }

protected:
    std::shared_ptr<Data> make_data(Runtime& rt, const DataKey& key) override {
        auto d = std::make_shared<RaggedDeviceData<T>>(
            key, sk::usm_vector<T>(this->shape.elements(), rt.allocator_for<T>(key.dev, key.kind)));
        const auto alloc = rt.allocator_for<std::uint64_t>(key.dev, key.kind);
        d->offs.emplace(alloc);
        for (int dd = 1; dd < this->rank; ++dd)
            if (this->shape.per[static_cast<std::size_t>(dd)]) d->per[static_cast<std::size_t>(dd - 1)].emplace(alloc);
        return d;
    }

    /// The offsets and the per-index arrays, from the resolved shape: a host-sourced dimension from its host array
    /// (from_host: the same rules as the dense path); a device-sourced one from its DeviceColumn, assigned across
    /// (SlabKit routes it between devices) after the column's event (section 6). Nothing here touches the host for a
    /// value-driven shape.
    void fill_index(Runtime& rt, Data& base, const DataKey& key) override {
        const ResolvedShape& s = this->shape;
        if (!s.ragged()) return;
        auto&      d     = static_cast<RaggedDeviceData<T>&>(base);
        const auto alloc = rt.allocator_for<std::uint64_t>(key.dev, key.kind);
        if (!d.offs) d.offs.emplace(alloc);
        for (int dd = 1; dd < this->rank; ++dd) {
            const auto u = static_cast<std::size_t>(dd);
            if (!s.per[u]) continue;
            auto& slot = d.per[u - 1];
            if (!slot) slot.emplace(alloc);
            if (s.dev_per[u]) {
                const DeviceColumn&             col = *s.dev_per[u];
                const std::array<sycl::event, 1> dep{col.ev};
                const sycl::event e = slot->assign(*col.extents, std::span<const sycl::event>{dep});
                col.extents->register_event(e);  // a later rebuild of the column waits for this read
            } else {
                const std::vector<std::uint64_t> conv(s.per_index[u].begin(), s.per_index[u].end());
                slot->from_host(std::span<const std::uint64_t>{conv});
            }
        }
        if (!s.device_sourced()) {
            const std::vector<std::uint64_t> offs = s.block_offsets();
            d.offs->from_host(std::span<const std::uint64_t>{offs});
            return;
        }
        // Exactly one device-sourced per-index dimension (finish_ragged_total guarantees it): its scan, times the scalar
        // dimensions. A product of 1 is the column's scan itself; otherwise it is scaled in place on this key's device.
        const std::size_t   dd  = s.dev_per[1] ? 1 : 2;
        const DeviceColumn& col = *s.dev_per[dd];
        std::uint64_t       prod = 1;
        for (std::size_t x = 1; x < 3; ++x)
            if (x != dd) prod *= s.extents[x];
        const std::array<sycl::event, 1> dep{col.ev};
        const sycl::event                a = d.offs->assign(*col.offs, std::span<const sycl::event>{dep});  // resizes to n + 1
        col.offs->register_event(a);
        if (prod == 1) return;
        std::vector<sycl::event> deps{a};
        const std::vector<sycl::event> own = d.offs->events();
        deps.insert(deps.end(), own.begin(), own.end());
        std::uint64_t* o = d.offs->data();
        const sycl::event scaled = rt.uq.parallel_for(key.dev, deps, sycl::range<1>{col.n + 1}, [=](sycl::id<1> i) { o[i] = o[i] * prod; });
        d.offs->register_event(scaled);
    }

    void copy_index(Data& dst, Data& src) override {
        auto& D = static_cast<RaggedDeviceData<T>&>(dst);
        auto& S = static_cast<RaggedDeviceData<T>&>(src);
        const auto assign = [](std::optional<sk::usm_vector<std::uint64_t>>& to,
                               const std::optional<sk::usm_vector<std::uint64_t>>& from) {
            if (!to || !from) return;
            const std::vector<sycl::event> ev = from->events();
            to->assign(*from, std::span<const sycl::event>{ev});
        };
        assign(D.offs, S.offs);
        for (std::size_t k = 0; k < D.per.size(); ++k) assign(D.per[k], S.per[k]);
    }
};

/// A new shared metadata of the right typed kind: dtype dispatched at run time (atomic dtypes share their value
/// type's storage), `ragged` choosing the ragged type.
inline std::shared_ptr<VarMeta> make_var_meta(DType dt, int rank, std::vector<DimRecord> def, bool ragged) {
    std::shared_ptr<VarMeta> out;
    dispatch_dtype(dt, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (ragged)
            out = std::make_shared<RaggedVar<T>>(dt, rank, std::move(def));
        else
            out = std::make_shared<Var<T>>(dt, rank, std::move(def));
    });
    return out;
}

}  // namespace detail
}  // namespace naga::cospan
