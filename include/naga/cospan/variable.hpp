#pragma once

// =============================================================================
// naga/cospan/variable.hpp — a variable's shared metadata and its per-device data
// =============================================================================
//
// One VarMeta per variable, held by shared_ptr by EVERY instance that belongs to it
// (docs/design-review-2026-10-03.md, section 1). It carries what the variable IS: dtype, rank, the shape
// definition, the resolved shape, the initializer, the stale and materialized flags, and weak pointers
// to the instances that share it and to each placement's data. It holds no memory and keeps nothing alive. It holds
// NO EVENTS: every container keeps its own read and write lists (SlabKit's usm_vector), and a task orders itself
// through the containers it touches.
//
//   - the metadata lives while any instance holding it exists;
//   - a placement's data (a DeviceData, which owns a SlabKit container) is created EMPTY when an instance (a
//     holder) is declared on that placement, lives while any holder of it exists, and the metadata's entry for it is
//     weak. Entries are never removed: a key whose data is gone gets a new empty one when a holder asks again;
//   - `materialized` is set when the first materialization (section 3) has sized every live container; until then
//     the variable is always stale. After it, a holder checks only its own container (an empty one is sized and
//     copied from the first live container with data).
//
// A ragged variable (rank 2 only) is a different metadata type (RaggedVar) on the same idea: its device data also
// holds the device copies of the block offsets and the counts.
//
// THE SHAPE (design review, section 10). The DEFINITION (one DimRecord per dimension) is fixed once setup ends; the
// ACTUAL shape (ResolvedShape) changes at run time, and ONLY through VarMeta::set_shape / set_dims, which check every
// change against the definition (who may set which dimension, fixed values, agreement within a forward).
//
// RULES ENFORCED HERE. materialize (below) is the one function that brings data into being. It never
// runs outside a Runtime (runtime.hpp), allocates on every device that has an instance, and initializes
// or copies, as section 3 says. Merging variables, checking ranks and deciding which initializer wins
// are the business of the connect, tie and promote calls (wiring.hpp): VarMeta is passive.

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

/// One placement's data of a variable, type-erased: the container (a SlabKit usm_vector) and, through it, the
/// events of everything that read or wrote it. Held by shared_ptr by every holder (instance) on that placement; the
/// VarMeta keeps a weak pointer.
///
/// The data exists from the declaration of its first holder, EMPTY and UNBOUND: no Runtime is available at setup, so
/// the container itself (which needs an allocator) is made by bind(), at the first materialization that touches it.
/// An unbound data is size 0, has no events and holds no memory.
///
/// EVENTS (SlabKit's rules): a READER of the data (a kernel that reads it, a copy out of it) waits on read_deps() (the
/// writes) and registers a read; a WRITER (a kernel that writes it) waits on write_deps() (the reads) and registers a
/// write. Writes never wait on writes. The container's own operations (resize, a copy in, a free) wait on both lists.
struct DeviceDataBase {
    DataKey key;

    explicit DeviceDataBase(DataKey k) : key{std::move(k)} {}
    virtual ~DeviceDataBase() = default;

    /// What an outside READER must wait on: the write list(s), collapsed.
    virtual std::vector<sycl::event> read_deps() const = 0;
    /// What an outside WRITER must wait on: the read list(s), collapsed.
    virtual std::vector<sycl::event> write_deps() const = 0;
    /// Hands the container the events of outside work that read (wrote) its data.
    virtual void register_read(std::span<const sycl::event> evs)  = 0;
    virtual void register_write(std::span<const sycl::event> evs) = 0;
    void         register_read(const sycl::event& e) { register_read(std::span<const sycl::event>{&e, 1}); }
    void         register_write(const sycl::event& e) { register_write(std::span<const sycl::event>{&e, 1}); }

    /// Makes the container (size 0, no allocation) from the Runtime's allocator for this key. Idempotent.
    virtual void bind(Runtime& rt) = 0;
    virtual bool bound() const     = 0;
    /// Null while unbound or empty.
    virtual void*       raw()        = 0;
    virtual std::size_t size() const = 0;
    /// The allocated bytes of the element container (its capacity), for the pool accounting.
    virtual std::size_t capacity_bytes() const = 0;
    /// Makes this data a copy of `src` (the same element type, another placement): the container's assign, so it
    /// waits on the source's write list and on both of its own, resizes to the source, and registers a read on the
    /// source and a write on this. `deps` are extra dependencies. Returns the event of the copy. A ragged variable's
    /// index arrays follow. Both must be bound.
    virtual sycl::event copy_from(DeviceDataBase& src, std::span<const sycl::event> deps) = 0;
    /// A ragged variable's data (it carries index arrays).
    virtual bool is_ragged_data() const noexcept { return false; }
};

template <class T>
struct DeviceData : DeviceDataBase {
    std::optional<sk::usm_vector<T>> vec;  ///< engaged by bind()

    explicit DeviceData(DataKey k) : DeviceDataBase{std::move(k)} {}

    /// The container. \throws std::logic_error while unbound.
    sk::usm_vector<T>& v() {
        if (!vec) throw std::logic_error("naga::cospan: a data was used before it was bound to a Runtime");
        return *vec;
    }
    const sk::usm_vector<T>& v() const {
        if (!vec) throw std::logic_error("naga::cospan: a data was used before it was bound to a Runtime");
        return *vec;
    }

    std::vector<sycl::event> read_deps() const override { return vec ? vec->read_deps() : std::vector<sycl::event>{}; }
    std::vector<sycl::event> write_deps() const override { return vec ? vec->write_deps() : std::vector<sycl::event>{}; }
    void register_read(std::span<const sycl::event> evs) override {
        if (vec) vec->register_read(evs);
    }
    void register_write(std::span<const sycl::event> evs) override {
        if (vec) vec->register_write(evs);
    }
    using DeviceDataBase::register_read;
    using DeviceDataBase::register_write;

    void bind(Runtime& rt) override {
        if (!vec) vec.emplace(rt.allocator_for<T>(key.dev, key.kind));
    }
    bool        bound() const override { return vec.has_value(); }
    void*       raw() override { return vec ? vec->data() : nullptr; }
    std::size_t size() const override { return vec ? vec->size() : 0; }
    std::size_t capacity_bytes() const override { return vec ? vec->capacity() * sizeof(T) : 0; }
    sycl::event copy_from(DeviceDataBase& src, std::span<const sycl::event> deps) override {
        return v().assign(static_cast<DeviceData<T>&>(src).v(), deps);
    }
};

/// A ragged variable's device data (rank 2): the elements, the N + 1 block offsets and the counts (the extents of the
/// per-index dimension). The index arrays are containers of their own and COUNT AS THE DATA: a kernel that reads or
/// writes the elements READS the index arrays (so it registers a read on them and waits on their writes); reshaping or
/// setting them (fill_index, a copy) is a write of theirs, which waits on both their lists.
template <class T>
struct RaggedDeviceData : DeviceData<T> {
    std::optional<sk::usm_vector<std::uint64_t>> offs;
    std::optional<sk::usm_vector<std::uint64_t>> per;

    using DeviceData<T>::DeviceData;

    static void append(std::vector<sycl::event>& out, const std::optional<sk::usm_vector<std::uint64_t>>& v) {
        if (!v) return;
        const std::vector<sycl::event> e = v->read_deps();  // the index arrays' writes
        out.insert(out.end(), e.begin(), e.end());
    }
    /// The elements' writes and the index arrays' writes (a kernel reads the index arrays too).
    std::vector<sycl::event> read_deps() const override {
        std::vector<sycl::event> out = DeviceData<T>::read_deps();
        append(out, offs);
        append(out, per);
        return out;
    }
    /// The elements' reads, and the index arrays' writes (a writing kernel reads them).
    std::vector<sycl::event> write_deps() const override {
        std::vector<sycl::event> out = DeviceData<T>::write_deps();
        append(out, offs);
        append(out, per);
        return out;
    }
    void register_read(std::span<const sycl::event> evs) override {
        DeviceData<T>::register_read(evs);
        if (offs) offs->register_read(evs);
        if (per) per->register_read(evs);
    }
    /// A write of the elements; the kernel that did it also read the index arrays.
    void register_write(std::span<const sycl::event> evs) override {
        DeviceData<T>::register_write(evs);
        if (offs) offs->register_read(evs);
        if (per) per->register_read(evs);
    }
    using DeviceDataBase::register_read;
    using DeviceDataBase::register_write;

    void bind(Runtime& rt) override {
        DeviceData<T>::bind(rt);
        const auto alloc = rt.allocator_for<std::uint64_t>(this->key.dev, this->key.kind);
        if (!offs) offs.emplace(alloc);
        if (!per) per.emplace(alloc);
    }
    bool is_ragged_data() const noexcept override { return true; }

    /// The elements, then the index arrays that both sides have (each assign waits on the source array's writes).
    /// The returned event is the elements' copy.
    sycl::event copy_from(DeviceDataBase& src, std::span<const sycl::event> deps) override {
        const sycl::event e = DeviceData<T>::copy_from(src, deps);
        auto&             s = static_cast<RaggedDeviceData<T>&>(src);
        const auto        assign = [](std::optional<sk::usm_vector<std::uint64_t>>& to,
                               const std::optional<sk::usm_vector<std::uint64_t>>& from) {
            if (!to || !from) return;
            to->assign(*from, std::span<const sycl::event>{});
        };
        assign(offs, s.offs);
        assign(per, s.per);
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

/// The per-index dimension of a ragged shape, on the device (docs/design-review-2026-10-03.md, section 10): the extents
/// (counts) and their exclusive scan (the n + 1 block offsets), built by a ragged shape hook (shape_hook.hpp:
/// counts_from / offsets_from) with a parallel scan, and read by the consumers' index arrays. The host sees only the total
/// and a negative / invalid flag. The vectors are `mutable` so that a holder of a const pointer can register the event of
/// its own read on them. The hook reuses one column across forwards and bumps `version` at each rebuild.
struct DeviceColumn {
    DataKey                                          key;      ///< where the column lives (the source's device)
    std::size_t                                      n = 0;    ///< the number of indices
    mutable std::optional<sk::usm_vector<std::uint64_t>> extents;     ///< n entries
    mutable std::optional<sk::usm_vector<std::uint64_t>> offs;        ///< n + 1 entries: the exclusive scan
    mutable std::optional<sk::usm_vector<std::uint64_t>> neg_flag;    ///< 1 entry on the device: set (atomic OR) by a negative / invalid value
    std::optional<sk::usm_vector<std::uint64_t>>     host_total;  ///< host kind, 2 elements: [total, negative / invalid flag]
    std::uint64_t                                    total   = 0;     ///< read from host_total once the task is done
    std::uint64_t                                    version = 0;     ///< bumped at each rebuild
    sycl::event                                      ev;              ///< the task that built it
};

inline std::string extents_str(const std::vector<std::size_t>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + std::to_string(v[i]);
    return out + "]";
}

// -----------------------------------------------------------------------------
// The resolved shape
// -----------------------------------------------------------------------------

/// The shape a variable has NOW, as numbers. RAGGED IS RANK 2 ONLY (section 10): dimension 1 is per-index (`per`), it
/// reports extent 0, and its extents are sourced from the host (`per_index`) OR from the device (`dev_per`, a
/// DeviceColumn), never both. `total` is the number of elements of a ragged shape (the sum of the counts). While some
/// dimension is not set yet, `known` is false.
struct ResolvedShape {
    bool                                known = false;
    std::array<std::size_t, 3>          extents{1, 1, 1};  ///< 1-padded past the rank
    bool                                per = false;       ///< dimension 1 is per-index
    std::vector<std::size_t>            per_index{};       ///< the host counts, one per index of dimension 0
    std::shared_ptr<const DeviceColumn> dev_per{};         ///< the device counts and offsets (instead of per_index)
    std::uint64_t                       dev_version = 0;   ///< the column's version when this shape was built
    std::uint64_t                       total = 0;         ///< a ragged shape's number of elements

    bool ragged() const noexcept { return per; }
    bool device_sourced() const noexcept { return per && dev_per != nullptr; }

    /// The sum of the host counts.
    std::size_t host_sum() const {
        std::size_t sum = 0;
        for (const std::size_t c : per_index) sum += c;
        return sum;
    }

    /// The number of elements: the product of the extents, or, for a ragged shape, `total`. 0 while the shape is not
    /// known.
    std::size_t elements() const {
        if (!known) return 0;
        if (!per) return extents[0] * extents[1] * extents[2];
        return static_cast<std::size_t>(total);
    }

    /// Sets `total` of a ragged shape (the device column's, else the host sum); a dense shape's is 0.
    void settle_total() { total = !per ? 0 : (dev_per ? dev_per->total : static_cast<std::uint64_t>(host_sum())); }

    /// The N + 1 block offsets: offsets[i + 1] = offsets[i] + the size of block i (the count of a ragged shape, else one
    /// row of the rectangle). Host arrays only.
    std::vector<std::uint64_t> block_offsets() const {
        if (device_sourced()) throw std::logic_error("naga::cospan: its block offsets live on the device (a hooked ragged shape)");
        std::vector<std::uint64_t> offs(extents[0] + 1, 0);
        for (std::size_t i = 0; i < extents[0]; ++i)
            offs[i + 1] = offs[i] + static_cast<std::uint64_t>((per ? per_index[i] : extents[1]) * extents[2]);
        return offs;
    }

    /// The per-index extents on the host: the host array, or the device column copied down (waiting on it).
    std::vector<std::size_t> per_index_on_host(Runtime& rt) const {
        if (!dev_per) return per_index;
        const DeviceColumn&        col = *dev_per;
        std::vector<std::uint64_t> h(col.n);
        std::vector<sycl::event>   deps{col.ev};
        const std::vector<sycl::event> own = col.extents->read_deps();  // the extents' writes
        deps.insert(deps.end(), own.begin(), own.end());
        if (col.n != 0) {
            sycl::event e =
                rt.q.copy(col.key.dev, deps, static_cast<const std::uint64_t*>(col.extents->data()), h.data(), col.n);
            col.extents->register_read(e);  // a later rebuild of the column waits for this read
            e.wait();
        }
        return std::vector<std::size_t>(h.begin(), h.end());
    }

    friend bool operator==(const ResolvedShape& a, const ResolvedShape& b) {
        return a.known == b.known && a.extents == b.extents && a.per == b.per && a.per_index == b.per_index &&
               a.dev_per == b.dev_per && a.dev_version == b.dev_version && a.total == b.total;
    }
};

// -----------------------------------------------------------------------------
// The shared metadata
// -----------------------------------------------------------------------------

class Instance;  // instance.hpp

// What VarMeta needs to know of an instance, defined in instance.hpp (which includes this header).
inline DataKey             key_of(const Instance& i);
inline std::shared_ptr<DeviceDataBase> data_of(const Instance& i);  // the container the holder holds
inline void                refresh_container(Instance& i);           // fetch the holder's container for its key
inline IOKind              kind_of(const Instance& i);
inline bool                is_output_port(const Instance& i);
inline bool                is_tie_output(const Instance& i);
inline std::string         path_of(const Instance& i);
inline const sycl::device& device_of(const Instance& i);

class VarMeta;
inline std::shared_ptr<VarMeta> meta_of(const Instance& i);  // the instance's variable
inline void check_init_shape(const VarMeta& v, const ResolvedShape& r, const InitShape& s);

/// Who sets a dimension (design review, section 10). A producer sets its outputs (a component's pre_forward), an initializer
/// (or the definition itself) sets a free input, a shape hook sets the dimensions it owns (`hook` is its owner id).
/// `who` names the setter in errors.
struct SetterId {
    enum class Kind : uint8_t { producer, initializer, hook };
    Kind          kind = Kind::producer;
    std::uint32_t hook = 0;
    std::string   who;
};

/// Which dimensions a set touches.
using DimMask = std::array<bool, 3>;

class VarMeta : public std::enable_shared_from_this<VarMeta> {
public:
    VarMeta(DType dt, int r, std::vector<DimRecord> def) : dtype{dt}, rank{r}, definition{std::move(def)} {
        for (std::size_t u = 0; u < 3; ++u) dim_known_[u] = static_cast<int>(u) >= rank;  // past the rank: 1, always known
    }

    VarMeta(const VarMeta&)            = delete;
    VarMeta& operator=(const VarMeta&) = delete;
    virtual ~VarMeta()                 = default;

    DType dtype;  ///< atomic-ness preserved: the DType of the first declaration
    int   rank;

    /// The shape DEFINITION: one record per dimension, fixed once setup ends (section 10). Connect, tie and promote merge
    /// the definitions of what they join (wiring.hpp, merge_definitions); a shape hook marks the dimensions it owns.
    std::vector<DimRecord> definition;
    /// A ragged variable's declared per-index counts (cs::counts / cs::offsets): they define dimension 0 (their length)
    /// and dimension 1, whose record is then `fixed`.
    std::optional<std::vector<std::size_t>> def_counts;
    /// The ACTUAL shape, which changes at run time but only through set_shape / set_dims, each change checked against the
    /// definition.
    ResolvedShape shape;

    /// The initializer: the last declaration wins (section 2), so it is simply replaced by each declaration.
    std::optional<InitSpec> init;
    /// Where the initialization runs (set by tie_inputs): else the device with the most instances.
    std::optional<sycl::device> init_device;

    /// See is_stale(): this flag alone says nothing for a variable that is not materialized.
    bool stale_ = true;

    /// The first materialization has sized every live container (variable.hpp, Var::materialize). Never cleared.
    bool materialized = false;

    /// Set by tie_outputs: the members' own postcompute pushes to the variable's other placements are off, and the System
    /// that owns the tie alias pushes through the alias instead (System::run_compute).
    bool push_suppressed = false;

    /// The variable closes a cycle (connect found a consumer registered before its producer). Its initializer is kept and
    /// runs at the first materialization even though the variable is connected; with none, that is an error naming
    /// `cycle_note`. Sticky: merges carry it.
    bool        cycle = false;
    std::string cycle_note;

    /// Tie marks, one per kind (design review, 11.2): a tie call refuses a member whose variable already has the mark of
    /// its kind. Set by the tie call on the alias's variable; merges carry them (a promotion of a tie shares the variable).
    bool tied_input  = false;
    bool tied_output = false;

    /// Connected inside one fused component; a later phase may keep such an intermediate in local memory (section 4).
    bool internal = false;

    /// The instances that share this metadata (weak; used by merges and to answer where the variable lives).
    std::vector<std::weak_ptr<Instance>> instances;
    /// Each placement's data, weak: it lives while a holder on that placement holds it. Entries are never removed; one
    /// whose data is gone is given a new empty data when a holder asks for it (container_for).
    std::vector<std::pair<DataKey, std::weak_ptr<DeviceDataBase>>> data;

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

    /// A tie_outputs alias is one of this variable's instances: the shape belongs to that tie's System (a shape hook), and
    /// the pushes to the other placements are the tie's System's.
    bool is_tied_output() const {
        for (const auto& w : instances)
            if (const std::shared_ptr<Instance> s = w.lock())
                if (is_tie_output(*s)) return true;
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

    // ---- containers ----------------------------------------------------------------------------------
    /// A new EMPTY, unbound data of this variable's type for `key`.
    virtual std::shared_ptr<DeviceDataBase> new_container(const DataKey& key) const = 0;

    /// The live data of `key`, or a new empty one (recorded weakly here: the caller, a holder, keeps it alive).
    std::shared_ptr<DeviceDataBase> container_for(const DataKey& key) {
        for (auto& p : data)
            if (p.first == key) {
                if (std::shared_ptr<DeviceDataBase> live = p.second.lock()) return live;
                std::shared_ptr<DeviceDataBase> fresh = new_container(key);
                p.second                              = fresh;
                return fresh;
            }
        std::shared_ptr<DeviceDataBase> fresh = new_container(key);
        data.emplace_back(key, fresh);
        return fresh;
    }

    /// A merge: this variable survives and takes the live data of `other`'s keys it has none for (of the same kind:
    /// ragged or not). The holders of `other` are rebound next and find them.
    void adopt_containers(const VarMeta& other) {
        for (const auto& p : other.data) {
            const std::shared_ptr<DeviceDataBase> theirs = p.second.lock();
            if (!theirs || data_on(p.first) || theirs->is_ragged_data() != is_ragged()) continue;
            bool placed = false;
            for (auto& mine : data)
                if (mine.first == p.first) {
                    mine.second = theirs;
                    placed      = true;
                    break;
                }
            if (!placed) data.emplace_back(p.first, theirs);
        }
    }

    // ---- state -------------------------------------------------------------------------------------
    /// The first materialization has happened.
    bool is_materialized() const { return materialized; }
    /// A variable that is not materialized is always stale; a materialized one is stale after mark_stale()
    /// until its initializer has run again (docs/design-review-2026-10-03.md, section 5).
    bool is_stale() const { return !is_materialized() || stale_; }
    void mark_stale() { stale_ = true; }

    // ---- the shape: definition, actual shape, and who may change it (section 10) ----------------------
    /// How a dimension was declared, for messages.
    std::string declared_rule(int d) const {
        const auto       u = static_cast<std::size_t>(d);
        const DimRecord& r = definition[u];
        switch (r.how) {
            case DimRecord::How::fixed:
                return (u == 1 && def_counts) ? "fixed counts " + extents_str(*def_counts) : "fixed";
            case DimRecord::How::unspecified: return "unspecified";
            case DimRecord::How::hook: return "set by a shape hook";
        }
        return "?";
    }

    /// Every dimension has a value.
    bool all_dims_known() const { return dim_known_[0] && dim_known_[1] && dim_known_[2]; }

    /// "'<path>': its shape is not known yet", naming the dimensions that are not and who should set each.
    std::logic_error shape_unknown() const {
        std::string unknown;
        for (int d = 0; d < rank; ++d) {
            const auto u = static_cast<std::size_t>(d);
            if (dim_known_[u]) continue;
            std::string who;
            if (definition[u].how == DimRecord::How::hook)
                who = "a shape hook, which has not run";
            else if (has_producer())
                who = "its producer, in pre_forward (set_shape)";
            else
                who = "an initializer with a shape, a connection, or a tie";
            unknown += std::string{unknown.empty() ? "" : "; "} + "dimension " + std::to_string(d) + " (" + declared_rule(d) +
                       "): to be set by " + who;
        }
        return std::logic_error("naga::cospan: '" + display_path() + "': its shape is not known yet" +
                                (unknown.empty() ? std::string{} : ": " + unknown));
    }

    /// Gives the variable a whole new shape (every dimension), checked against the definition: see set_dims.
    void set_shape(Runtime& rt, const ResolvedShape& s, const SetterId& by) {
        DimMask all{};
        for (int d = 0; d < rank; ++d) all[static_cast<std::size_t>(d)] = true;
        set_dims(rt, s, all, by);
    }

    /// Sets the dimensions in `mask` from `s`. EVERY set is checked against the definition (section 10): a fixed dimension
    /// must equal its declared extent (or counts); a dimension owned by a shape hook is set by that hook only, and a hook
    /// sets only the dimensions it owns (and fixed ones, with their own value); an unspecified one is set by a producer or
    /// an initializer; and two sets of one dimension in one forward must agree. Nothing is changed when a check throws.
    /// Once every dimension has a value the shape is known; when it changed, every live data is resized in place (contents
    /// are kept, and the variable never becomes stale).
    void set_dims(Runtime& rt, const ResolvedShape& s, const DimMask& mask, const SetterId& by) {
        if (s.per && rank != 2)
            shape_fail("only a rank-2 variable can be ragged (" + by.who + " gave per-index extents for a rank " + std::to_string(rank) +
                       " variable)");
        for (int d = 0; d < rank; ++d) {
            const auto u = static_cast<std::size_t>(d);
            if (!mask[u]) continue;
            check_setter(u, s, by, rt);
            check_same_forward(u, s, by, rt);
        }
        // the per-index counts have one entry per index of dimension 0
        if (rank == 2 && ((mask[0] && mask[1] && s.per) || (mask[1] && s.per && dim_known_[0]) || (mask[0] && !mask[1] && shape.per && dim_known_[1]))) {
            const std::size_t n    = mask[0] ? s.extents[0] : shape.extents[0];
            const bool        take = mask[1];
            const std::size_t have = take ? (s.dev_per ? s.dev_per->n : s.per_index.size())
                                          : (shape.dev_per ? shape.dev_per->n : shape.per_index.size());
            if (have != n)
                shape_fail("the per-index dimension has " + std::to_string(have) + " extents but dimension 0 is " + std::to_string(n) +
                           " (" + by.who + ")");
        }

        const ResolvedShape before = shape;
        for (int d = 0; d < rank; ++d) {
            const auto u = static_cast<std::size_t>(d);
            if (!mask[u]) continue;
            if (u == 1 && rank == 2) {
                shape.per         = s.per;
                shape.per_index   = s.per ? s.per_index : std::vector<std::size_t>{};
                shape.dev_per     = s.per ? s.dev_per : nullptr;
                shape.dev_version = s.per ? s.dev_version : 0;
                shape.extents[1]  = s.per ? 0 : s.extents[1];
            } else {
                shape.extents[u] = s.extents[u];
            }
            dim_known_[u]  = true;
            set_forward_[u] = rt.forward;
            set_log_[u].emplace(by.who, s);
        }
        shape.known = all_dims_known();
        if (shape.known) shape.settle_total();
        if (shape.known && !(shape == before)) apply_shape(rt);
    }

    /// The dimensions the definition fixes take their declared value (a hooked tie's, an output nothing set_shape'd).
    void fill_fixed_dims(Runtime& rt) {
        DimMask       mask{};
        ResolvedShape s    = shape;
        bool          any  = false;
        const bool    cnts = rank == 2 && def_counts.has_value();
        for (int d = 0; d < rank; ++d) {
            const auto u = static_cast<std::size_t>(d);
            if (dim_known_[u] || definition[u].how != DimRecord::How::fixed) continue;
            mask[u] = true;
            any     = true;
            if (cnts) {  // the counts define dimension 0 (their length) and dimension 1
                mask[0] = mask[1] = true;
                s.per             = true;
                s.per_index       = *def_counts;
                s.extents[0]      = def_counts->size();
                s.extents[1]      = 0;
            } else {
                s.extents[u] = definition[u].extent;
            }
        }
        if (any) set_dims(rt, s, mask, SetterId{SetterId::Kind::initializer, 0, "the definition of '" + display_path() + "'"});
    }

    /// A variable nothing produces takes its shape from its definition (fixed dimensions) and its initializer's shape
    /// (unspecified ones; ragged counts honored), the initializer's shape being checked against the definition. Dimensions
    /// owned by a shape hook are left to the hook. Safe to call again after update_input replaced the initializer.
    void resolve_free_shape(Runtime& rt) {
        fill_fixed_dims(rt);
        if (!init || !init->shape()) return;
        const InitShape& is = *init->shape();
        check_init_against_definition(is);
        if (is.rank != rank) return;
        DimMask       mask{};
        ResolvedShape s   = shape;
        bool          any = false;
        for (int d = 0; d < rank; ++d) {
            const auto u = static_cast<std::size_t>(d);
            if (definition[u].how != DimRecord::How::unspecified) continue;
            mask[u] = true;
            any     = true;
            if (u == 1 && rank == 2 && is.per) {
                s.per       = true;
                s.per_index = is.per_index;
                s.dev_per   = nullptr;
                s.extents[1] = 0;
            } else {
                if (u == 1 && rank == 2) s.per = false;
                s.extents[u] = is.extents[u];
            }
        }
        if (any) set_dims(rt, s, mask, SetterId{SetterId::Kind::initializer, 0, "the initializer of '" + display_path() + "'"});
    }

    /// "4x[2,0,1,1]": a shape as text for messages (a per-index dimension prints its extents, read down from the device
    /// here: use on error paths only).
    std::string shape_text(const ResolvedShape& r, Runtime& rt) const {
        std::string out;
        for (int d = 0; d < rank; ++d) {
            const auto u = static_cast<std::size_t>(d);
            if (d > 0) out += "x";
            out += (u == 1 && rank == 2 && r.per) ? extents_str(r.per_index_on_host(rt)) : std::to_string(r.extents[u]);
        }
        return out;
    }

    /// The data on `key`, if it is live.
    virtual std::shared_ptr<DeviceDataBase> data_on(const DataKey& key) const {
        for (const auto& p : data)
            if (p.first == key) return p.second.lock();
        return {};
    }

    virtual bool is_ragged() const noexcept { return false; }

    /// MATERIALIZATION (section 3). The FIRST one of a variable is global: it sizes EVERY live container to the shape
    /// and, unless `trigger` (the holder that asked) is an output, runs the initializer on one placement (the tie's
    /// init_device, else the trigger's, else the busiest device) and copies the result to the other containers. After
    /// it a holder checks only ITS OWN container: an empty one is sized and copied from the first live container that
    /// holds data. A stale variable (update_input) is initialized again in place. `trigger` may be null (a shape hook
    /// reading a variable that is not materialized yet).
    virtual void materialize(Runtime& rt, const Instance* trigger = nullptr) = 0;

    /// The shape changed (and is known): resizes every live, bound container in place (contents are kept, and the
    /// variable never becomes stale) and rebuilds a ragged variable's index arrays. Called by set_dims. Before the
    /// first materialization it does nothing (materialize sizes everything).
    virtual void apply_shape(Runtime& rt) = 0;

    /// Makes `holder`'s own container live (a host view's, a new instance's) by copying from a live one. It NEVER
    /// initializes, even when the variable is stale (section 5: a view made after update_input shows the old values
    /// until the next run). \throws std::logic_error when the variable is not materialized.
    virtual void attach_holder(Runtime& rt, const Instance& holder) = 0;

private:
    [[noreturn]] void shape_fail(const std::string& why) const {
        throw std::invalid_argument("naga::cospan: shape of '" + display_path() + "': " + why);
    }

    /// One dimension of `r` as text, for messages.
    std::string dim_text(const ResolvedShape& r, std::size_t u, Runtime& rt) const {
        return (u == 1 && rank == 2 && r.per) ? extents_str(r.per_index_on_host(rt)) : std::to_string(r.extents[u]);
    }

    /// Dimension u of two shapes is the same value (the same device column and version is the same extents).
    bool dim_equal(std::size_t u, const ResolvedShape& a, const ResolvedShape& b, Runtime& rt) const {
        if (u == 1 && rank == 2) {
            if (a.per != b.per) return false;
            if (a.per) {
                if (a.dev_per && a.dev_per == b.dev_per && a.dev_version == b.dev_version) return true;
                return a.per_index_on_host(rt) == b.per_index_on_host(rt);
            }
        }
        return a.extents[u] == b.extents[u];
    }

    /// Dimension u of `s` against the definition and the setter's right to touch it.
    void check_setter(std::size_t u, const ResolvedShape& s, const SetterId& by, Runtime& rt) const {
        const DimRecord&  r   = definition[u];
        const std::string who = by.who.empty() ? std::string{"<unknown>"} : by.who;
        const std::string dim = "dimension " + std::to_string(u);
        if (r.how == DimRecord::How::hook) {
            if (by.kind != SetterId::Kind::hook || by.hook != r.hook_owner)
                shape_fail(dim + " is set by a shape hook; " + who + " cannot set it");
        } else if (by.kind == SetterId::Kind::hook && r.how != DimRecord::How::fixed) {
            shape_fail(dim + " is " + declared_rule(static_cast<int>(u)) + ", not owned by " + who + ": a hook sets only the dimensions it owns");
        }
        if (r.how != DimRecord::How::fixed) return;
        if (u == 1 && def_counts) {
            if (!s.per)
                shape_fail(dim + " is declared as the per-index counts " + extents_str(*def_counts) + " but " + who + " gave the extent " +
                           std::to_string(s.extents[u]));
            const std::vector<std::size_t> have = s.per_index_on_host(rt);
            if (have != *def_counts)
                shape_fail(dim + " is declared as the per-index counts " + extents_str(*def_counts) + " but " + who + " gave " + extents_str(have));
        } else {
            if (u == 1 && rank == 2 && s.per)
                shape_fail(dim + " is declared as " + std::to_string(r.extent) + " but " + who + " gave per-index extents " +
                           extents_str(s.per_index_on_host(rt)));
            if (s.extents[u] != r.extent)
                shape_fail(dim + " is declared as " + std::to_string(r.extent) + " but " + who + " gave " + std::to_string(s.extents[u]));
        }
    }

    /// A second set of dimension u in one forward must agree with the first.
    void check_same_forward(std::size_t u, const ResolvedShape& s, const SetterId& by, Runtime& rt) const {
        if (set_forward_[u] != rt.forward || !set_log_[u]) return;
        const auto& prev = *set_log_[u];
        if (dim_equal(u, prev.second, s, rt)) return;
        shape_fail("dimension " + std::to_string(u) + " was set to " + dim_text(prev.second, u, rt) + " by " + prev.first + " and to " +
                   dim_text(s, u, rt) + " by " + by.who + " in the same forward; two sets of one dimension must agree");
    }

    /// The initializer's shape against the fixed dimensions of the definition (and the rank).
    void check_init_against_definition(const InitShape& is) const {
        if (is.flat && rank != 1) return;
        const std::string name = display_path();
        if (is.rank != rank)
            throw std::invalid_argument("naga::cospan: init for '" + name + "' has rank " + std::to_string(is.rank) + " but '" + name +
                                        "' resolved rank " + std::to_string(rank));
        for (int d = 0; d < rank; ++d) {
            const auto       u = static_cast<std::size_t>(d);
            const DimRecord& r = definition[u];
            if (r.how != DimRecord::How::fixed) continue;
            std::vector<std::size_t> init_v, def_v;
            bool                     as_counts = false;
            if (u == 1 && def_counts) {
                as_counts = true;
                init_v    = is.per ? is.per_index : std::vector<std::size_t>(is.extents[0], is.extents[1]);
                def_v     = *def_counts;
            } else {
                as_counts = u == 1 && is.per;
                init_v    = as_counts ? is.per_index : std::vector<std::size_t>{is.extents[u]};
                def_v     = {r.extent};
            }
            if (init_v == def_v) continue;
            throw std::invalid_argument("naga::cospan: init for '" + name + "' has dimension " + std::to_string(d) + " = " +
                                        (as_counts ? extents_str(init_v) : std::to_string(init_v.front())) + " but '" + name +
                                        "' resolved dimension " + std::to_string(d) + " (" + declared_rule(d) + ") = " +
                                        (as_counts ? extents_str(def_v) : std::to_string(r.extent)));
        }
    }

    std::array<bool, 3>          dim_known_{};                          ///< which dimensions have a value
    std::array<std::uint64_t, 3> set_forward_{~0ull, ~0ull, ~0ull};     ///< the forward of each dimension's last set
    std::array<std::optional<std::pair<std::string, ResolvedShape>>, 3> set_log_{};  ///< who set it, and the shape given
};

/// The shape an init carries against the variable's resolved shape: the rank and EVERY dimension (fixed, inherited, hooked,
/// per-index element by element). A scalar extent c equals a per-index array of c's. A device-sourced per-index dimension is
/// compared by the number of indices and the total only (its extents are not read down). Bare data (from_host(vec)) describes
/// only a rank-1 variable; its size is checked when it is written.
inline void check_init_shape(const VarMeta& v, const ResolvedShape& r, const InitShape& s) {
    if (s.flat && v.rank != 1) return;
    const std::string name = v.display_path();
    if (s.rank != v.rank)
        throw std::invalid_argument("naga::cospan: init for '" + name + "' has rank " + std::to_string(s.rank) + " but '" + name +
                                    "' resolved rank " + std::to_string(v.rank));
    const std::size_t n = r.extents[0];
    for (int d = 0; d < v.rank; ++d) {
        const auto u        = static_cast<std::size_t>(d);
        const bool init_per = d == 1 && s.per;
        const bool var_per  = d == 1 && r.per;
        if (var_per && r.dev_per) {
            if (s.extents[0] != n || s.elements() != static_cast<std::size_t>(r.total))
                throw std::invalid_argument("naga::cospan: init for '" + name + "' has " + std::to_string(s.extents[0]) + " indices and " +
                                            std::to_string(s.elements()) + " elements but '" + name + "' resolved " + std::to_string(n) +
                                            " indices and " + std::to_string(r.total) + " elements (dimension 1, " + v.declared_rule(d) + ")");
            continue;
        }
        std::vector<std::size_t> init_v, var_v;
        if (d == 0) {
            init_v = {s.extents[0]};
            var_v  = {n};
        } else {
            init_v = init_per ? s.per_index : std::vector<std::size_t>(s.extents[0], s.extents[u]);
            var_v  = var_per ? r.per_index : std::vector<std::size_t>(n, r.extents[u]);
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

    std::shared_ptr<DeviceDataBase> new_container(const DataKey& key) const override { return std::make_shared<Data>(key); }

    void materialize(Runtime& rt, const Instance* trigger) override {
        const std::vector<std::shared_ptr<Instance>> insts = live_instances();
        if (insts.empty()) return;
        for (const auto& i : insts) refresh_container(*i);  // every holder has the container of its own key

        if (!materialized) {
            first_materialization(rt, trigger);
            return;
        }
        // Materialized: a stale variable (update_input) is initialized again in place; then the holder that asked checks
        // only its own container. (A cycle variable has a producer and is initialized by its initializer all the same.)
        if (stale_ && (!has_producer() || cycle)) reinitialize(rt, trigger);
        if (trigger != nullptr) populate(rt, std::static_pointer_cast<Data>(data_of(*trigger)));
    }

    void attach_holder(Runtime& rt, const Instance& holder) override {
        if (!is_materialized())
            throw std::logic_error("naga::cospan: '" + display_path() +
                                   "' has no data: nothing that uses it has run, or its data was freed");
        populate(rt, std::static_pointer_cast<Data>(data_of(holder)));
    }

    /// The shape changed and is known (set_dims): every live, bound container is resized in place (the prefix is kept)
    /// and a ragged variable's index arrays follow. Nothing before the first materialization, which sizes everything.
    void apply_shape(Runtime& rt) override {
        if (!materialized) return;
        for (const auto& d : live_containers(/*bound_only=*/true)) size_container(rt, *d);
    }

protected:
    /// A ragged variable's index arrays follow the host shape. Nothing for a dense one.
    virtual void fill_index(Runtime&, Data&, const DataKey&) {}

private:
    /// THE FIRST MATERIALIZATION (global): the shape, then every live container sized to it, then the initializer.
    void first_materialization(Runtime& rt, const Instance* trigger) {
        // The shape: the dimensions the definition fixes, and for a variable nothing produces, its initializer's shape
        // (section 10). Anything still unknown is an error naming the dimension and who should set it.
        // A cycle variable is read before its producer has run: it has no initializer only by mistake, and the error
        // names the cycle (design review 11.2: checked here, not at setup, because update_input may supply it).
        if (cycle && !init)
            throw std::logic_error("naga::cospan: '" + display_path() + "' closes a cycle (" + cycle_note +
                                   ") and has no initializer: its first read needs one (init_input, or update_input before the "
                                   "first forward)");
        const bool from_init = !has_producer() || cycle;  // an initializer supplies the values, and the shape it carries
        if (!shape.known) {
            fill_fixed_dims(rt);
            if (from_init) resolve_free_shape(rt);
            if (!shape.known) throw shape_unknown();
        } else if (from_init) {
            resolve_free_shape(rt);  // an initializer's shape, or update_input's
        }

        // Size every live container (bound here: this is where the Runtime first reaches them).
        const std::vector<std::shared_ptr<Data>> live = live_containers(/*bound_only=*/false);
        for (const auto& d : live) size_container(rt, *d);

        // Unless the holder that asked is an output (its producer writes the data), initialize and copy to the others.
        // A variable with no initializer is all zeros already (a container is sized by value-initialization).
        // A cycle variable always initializes: whichever holder asks first, the consumer reads it before the producer writes.
        const bool output_asked = trigger != nullptr && kind_of(*trigger) == IOKind::output;
        if ((!output_asked || cycle) && init) initialize(rt, live, trigger);
        materialized = true;
        stale_       = false;
    }

    /// A stale, materialized variable with no producer (update_input): the (possibly new) shape, then the initializer
    /// again, in place, on the live containers that hold data.
    void reinitialize(Runtime& rt, const Instance* trigger) {
        resolve_free_shape(rt);  // update_input may have supplied a new shape (set_dims resizes the live containers)
        const std::vector<std::shared_ptr<Data>> live = live_containers(/*bound_only=*/true);
        if (!live.empty()) initialize(rt, live, trigger);
        stale_ = false;
    }

    /// Runs the initializer (or zero-fills) on one of `live`, then copies the result to the others. The initializer is a
    /// WRITER of its container: it waits on both its lists, and what it enqueued is registered as a write and, so that a
    /// later kernel writer waits for it too, as a read.
    void initialize(Runtime& rt, const std::vector<std::shared_ptr<Data>>& live, const Instance* trigger) {
        const std::shared_ptr<Data> d = pick_init_container(live, trigger);
        if (!d) return;
        std::vector<sycl::event> deps = d->read_deps();
        {
            const std::vector<sycl::event> w = d->write_deps();
            deps.insert(deps.end(), w.begin(), w.end());
        }
        sycl::event e;
        if (init) {
            if (init->shape()) check_init_shape(*this, shape, *init->shape());
            InitCtx ctx{*this, *d, rt, deps};
            e = init->apply(ctx);
        } else {
            e = d->v().fill(T{}, std::span<const sycl::event>{deps});
        }
        d->register_write(e);
        d->register_read(e);
        for (const auto& o : live) {
            if (o == d) continue;
            const sycl::event c = o->copy_from(*d, std::span<const sycl::event>{});  // waits on d's writes, incl. e
            o->register_read(c);
        }
    }

    /// Where the initializer runs: the tie's init_device, else the holder that asked, else the busiest device (a
    /// device-kind container preferred on it).
    std::shared_ptr<Data> pick_init_container(const std::vector<std::shared_ptr<Data>>& live, const Instance* trigger) const {
        const auto on = [&](const sycl::device& dev) -> std::shared_ptr<Data> {
            std::shared_ptr<Data> any;
            for (const auto& d : live) {
                if (!(d->key.dev == dev)) continue;
                if (d->key.kind == sk::kind::device) return d;
                if (!any) any = d;
            }
            return any;
        };
        if (init_device)
            if (std::shared_ptr<Data> d = on(*init_device)) return d;
        if (trigger != nullptr) {
            const std::shared_ptr<Data> t = std::static_pointer_cast<Data>(data_of(*trigger));
            if (t && std::find(live.begin(), live.end(), t) != live.end()) return t;
        }
        if (std::shared_ptr<Data> d = on(busiest_device())) return d;
        return live.empty() ? nullptr : live.front();
    }

    /// `own` is the holder's container. An empty one is bound, sized and copied from the first live container that
    /// holds data (or, with none, just sized: a variable nothing wrote yet is zeros).
    void populate(Runtime& rt, const std::shared_ptr<Data>& own) {
        if (!own) return;
        own->bind(rt);
        // A ragged variable with no elements still has its index arrays (N + 1 zero offsets): size it, do not return.
        if (own->size() != 0 || (shape.elements() == 0 && !is_ragged())) return;
        for (const auto& p : data) {
            const std::shared_ptr<DeviceDataBase> b = p.second.lock();
            if (!b || b == own || !b->bound() || b->size() == 0) continue;
            const sycl::event e = own->copy_from(*b, std::span<const sycl::event>{});  // sizes to the source; a ragged one's index follows
            own->register_read(e);  // a later kernel writer of this container waits for the copy
            return;
        }
        size_container(rt, *own);
    }

    /// Binds `d` and sizes it to the shape (the prefix is kept; new elements are zero), then its index arrays. The
    /// resize is the container's own write; it is also registered as a read, so that a kernel that writes the container
    /// (a writer waits on the read list) comes after it.
    void size_container(Runtime& rt, Data& d) {
        d.bind(rt);
        const std::size_t n = shape.elements();
        if (d.size() != n) d.register_read(d.v().resize(n));
        fill_index(rt, d, d.key);
    }

    /// The live data (every key that has one), optionally only those already bound to the Runtime.
    std::vector<std::shared_ptr<Data>> live_containers(bool bound_only) const {
        std::vector<std::shared_ptr<Data>> out;
        for (const auto& p : data)
            if (std::shared_ptr<DeviceDataBase> b = p.second.lock())
                if (!bound_only || b->bound()) out.push_back(std::static_pointer_cast<Data>(b));
        return out;
    }
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

    std::shared_ptr<DeviceDataBase> new_container(const DataKey& key) const override {
        return std::make_shared<RaggedDeviceData<T>>(key);
    }

protected:

    /// The offsets and the counts, from the resolved shape (rank 2): host counts from the host array (from_host: the same
    /// rules as the dense path); a device-sourced dimension from its DeviceColumn, assigned across (SlabKit routes it
    /// between devices) after the column's event (section 10). Nothing here touches the host for a hooked ragged shape.
    void fill_index(Runtime& rt, Data& base, const DataKey&) override {
        const ResolvedShape& s = this->shape;
        if (!s.ragged()) return;
        auto& d = static_cast<RaggedDeviceData<T>&>(base);
        d.bind(rt);  // the index arrays
        if (s.dev_per) {
            // assign waits on the column's writes (build_column registered its task as one) and registers the read, so a
            // later rebuild of the column waits for it.
            const DeviceColumn&              col = *s.dev_per;
            const std::array<sycl::event, 1> dep{col.ev};
            d.per->assign(*col.extents, std::span<const sycl::event>{dep});
            d.offs->assign(*col.offs, std::span<const sycl::event>{dep});  // resizes to n + 1
            return;
        }
        const std::vector<std::uint64_t> conv(s.per_index.begin(), s.per_index.end());
        d.per->from_host(std::span<const std::uint64_t>{conv});
        const std::vector<std::uint64_t> offs = s.block_offsets();
        d.offs->from_host(std::span<const std::uint64_t>{offs});
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
