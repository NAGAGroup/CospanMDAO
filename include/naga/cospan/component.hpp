#pragma once

// =============================================================================
// naga/cospan/component.hpp — ComponentBase and Component<KB> (r01 §6)
// =============================================================================
//
// ComponentBase is the user surface: hooks, declarations, retrieval,
// initializers, and every runner body that touches no KB-typed state. It is
// the friend of detail::IOEntry (friendship is not inherited), so those bodies
// live here. Component<KB> adds the typed layer: launch-config staging,
// compile() and the compute-forward runner.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/cospan/alias_eval.hpp"
#include "naga/cospan/dtype.hpp"
#include "naga/cospan/host_view.hpp"
#include "naga/cospan/init.hpp"
#include "naga/cospan/kernel.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/reactive_shape.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/wrappers.hpp"

namespace naga::cospan {

namespace detail {

/// RAII flag: set for a scope, cleared on every exit.
struct FlagScope {
    explicit FlagScope(bool& f) : f_{f} { f_ = true; }
    ~FlagScope() { f_ = false; }
    FlagScope(const FlagScope&)            = delete;
    FlagScope& operator=(const FlagScope&) = delete;
    bool& f_;
};

}  // namespace detail

class ComponentBase : public Subsystem {
public:
    ComponentBase(const ComponentBase&)            = delete;
    ComponentBase& operator=(const ComponentBase&) = delete;
    ~ComponentBase() override = default;

    // ---- HOOKS (user-implemented) ------------------------------------------
    /// Declare I/O. Runs once, during the declare pass. Zero allocation.
    virtual void setup() = 0;
    /// There is no init hook: a free input's initial value is declared in setup() with
    /// init_input, and outputs are never initialized (r01 §6.0, §4.4).
    /// Per-forward manual shape control, after automatic derivation.
    virtual void pre_forward() {}

    // ---- declaration API (legal inside setup() only) ------------------------
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_input(Input<DT, Rank>& m, std::string name, Specs... specs) {
        add_input<DT, Rank>(m, std::move(name),
                            std::array<detail::DimRecord, Rank>{detail::make_dim_record(specs)...});
    }
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_output(Output<DT, Rank>& m, std::string name, Specs... specs) {
        add_output<DT, Rank>(m, std::move(name),
                             std::array<detail::DimRecord, Rank>{detail::make_dim_record(specs)...});
    }

    /// The array forms: unspecified_shape<R>() and from_input_shape<R>(name).
    template <DType DT, int Rank, DimSpec S>
    void declare_input(Input<DT, Rank>& m, std::string name,
                       const std::array<S, static_cast<std::size_t>(Rank)>& specs) {
        add_input<DT, Rank>(m, std::move(name), records_of<Rank>(specs));
    }
    template <DType DT, int Rank, DimSpec S>
    void declare_output(Output<DT, Rank>& m, std::string name,
                        const std::array<S, static_cast<std::size_t>(Rank)>& specs) {
        add_output<DT, Rank>(m, std::move(name), records_of<Rank>(specs));
    }

    /// The whole-shape value-driven marker.
    template <DType DT, int Rank>
    void declare_input(Input<DT, Rank>& m, std::string name, FromInputArray marker) {
        add_input<DT, Rank>(m, std::move(name), detail::expand_from_input_array<Rank>(marker));
    }
    template <DType DT, int Rank>
    void declare_output(Output<DT, Rank>& m, std::string name, FromInputArray marker) {
        add_output<DT, Rank>(m, std::move(name), detail::expand_from_input_array<Rank>(marker));
    }

    /// Ragged ports take the array form too: cs::unspecified_shape<R>() (every dimension supplied from outside).
    template <DType DT, int Rank, DimSpec S>
    void declare_input(RaggedInput<DT, Rank>& m, std::string name, const std::array<S, static_cast<std::size_t>(Rank)>& specs) {
        register_entry(std::make_unique<detail::RaggedInputEntry<DT, Rank>>(name, records_of<Rank>(specs), &m), &m, "declare_input");
    }
    template <DType DT, int Rank, DimSpec S>
    void declare_output(RaggedOutput<DT, Rank>& m, std::string name, const std::array<S, static_cast<std::size_t>(Rank)>& specs) {
        register_entry(std::make_unique<detail::RaggedOutputEntry<DT, Rank>>(name, records_of<Rank>(specs), &m), &m, "declare_output");
    }

    /// Ragged ports: RaggedInput / RaggedOutput members, Rank 2..3, with the same DimSpec
    /// arguments (scalar and per-index mixed: PerIndex, FromInputValues on dims >= 1).
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_input(RaggedInput<DT, Rank>& m, std::string name, Specs... specs) {
        register_entry(std::make_unique<detail::RaggedInputEntry<DT, Rank>>(
                           name, std::array<detail::DimRecord, Rank>{detail::make_dim_record(specs)...}, &m),
                       &m, "declare_input");
    }
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_output(RaggedOutput<DT, Rank>& m, std::string name, Specs... specs) {
        register_entry(std::make_unique<detail::RaggedOutputEntry<DT, Rank>>(
                           name, std::array<detail::DimRecord, Rank>{detail::make_dim_record(specs)...}, &m),
                       &m, "declare_output");
    }

    /// The pre_forward-only resize (same setter as automatic derivation).
    template <DType DT, int Rank>
    void reshape(Input<DT, Rank>& m, sycl::range<Rank> shape) {
        reshape_member(&m, shape);
    }
    template <DType DT, int Rank>
    void reshape(Output<DT, Rank>& m, sycl::range<Rank> shape) {
        reshape_member(&m, shape);
    }

    // ---- retrieval & enumeration ---------------------------------------------
    template <DType DT, int Rank>
    Input<DT, Rank>& get_input(std::string_view name) {
        auto* e = dynamic_cast<detail::InputEntry<DT, Rank>*>(&find_kind(name, IOKind::input));
        if (e == nullptr) throw wrong_type(name);
        return e->member();
    }
    template <DType DT, int Rank>
    Output<DT, Rank>& get_output(std::string_view name) {
        auto* e = dynamic_cast<detail::OutputEntry<DT, Rank>*>(&find_kind(name, IOKind::output));
        if (e == nullptr) throw wrong_type(name);
        return e->member();
    }

    using EntryRefs = std::vector<std::reference_wrapper<const detail::IOEntry>>;
    EntryRefs get_inputs() const { return entries_of(IOKind::input); }
    EntryRefs get_outputs() const { return entries_of(IOKind::output); }

    // ---- initial values (legal inside setup() only; r01 §4.4) -----------
    /// Declare the initial value of one of this Component's OWN inputs, by name. Metadata on the
    /// input's root, applied by the framework when the root is allocated from nothing; the
    /// declaration of the highest level wins, it is ignored when the input ends up with a source,
    /// and it is an error to name an output.
    void init_input(std::string_view name, InitSpec init) {
        if (!declaring_)
            throw std::logic_error("naga::cospan: init_input is legal inside setup() only (" + describe() + ")");
        detail::IOEntry& e = find_kind(name, IOKind::input);  // an output throws
        e.init_decls_.push_back(detail::IOEntry::InitDecl{std::move(init), detail::depth_of(*this), path()});
    }

    // ---- Subsystem verbs whose bodies touch no KB-typed state -----------------
    void run_setup() override {
        detail::FlagScope scope{declaring_};
        setup();
    }

    std::unordered_map<sycl::device, std::size_t> get_exec_info(
        const sycl::device& parent_device) override {
        device_ = parent_device;
        std::size_t bytes = 0;
        for (detail::IOEntry* e : order_) {
            e->device_       = parent_device;
            e->storage_kind_ = default_storage_kind();
            const Placement own = e->placement();
            detail::IOEntry&        root = *e->root();
            if (e->local_shape()) {
                root.set_primary_placement(own);
                bytes += root.static_bytes();
            } else if (e->kind() == IOKind::input) {
                // Host-kind placements cost no arena bytes; a device-kind one
                // costs the root's arena bytes once per device.
                if (!has_registered(root, own)) {
                    root.register_placement(own);
                    if (own.kind == sk::kind::device) bytes += root.static_bytes();
                }
            } else if (root.device_ && root.placement() != own) {
                throw std::logic_error("naga::cospan: " + describe() + ": output '" +
                                       std::string{e->name()} + "' is tied to '" +
                                       std::string{root.name()} +
                                       "', which sits on another placement (device, storage "
                                       "kind); producers cannot write one storage from two "
                                       "placements");
            }
        }
        return {{parent_device, bytes}};
    }

    void prepare_exec(sk::queue& queue, Substrates& substrates) override {
        queue_ = &queue;
        subs_  = &substrates;
        ev_.queue_  = &queue;
        // a shape-driving read may need a variable's storage: materialize it on need (r01 §4.5)
        ev_.ensure_ = [this](detail::IOEntry& r) { r.variable().materialize(queue_ref(), nullptr, !replay_); };
        for (detail::IOEntry* e : order_)
            if (e->kind() == IOKind::input && !e->local_shape() && e->tie_alias_ == nullptr && e->has_unspecified())
                e->update_shape();  // takes the connected source's unspecified dims
        // No allocation at setup: a variable is materialized on need, in run_pre_compute (r01 §4.5).
        // Roots only learn what to allocate from.
        for (detail::IOEntry* e : order_)
            if (e->local_shape()) e->bind_substrates(substrates);
        for (detail::IOEntry* e : order_) {
            if (!e->local_shape() && !e->root()->is_dynamic()) e->check_constraints(true);  // fixed dims, at setup
            if (e->kind() == IOKind::input && !e->local_shape() && !e->is_dynamic() &&
                !e->root()->is_dynamic())
                e->validate_connection();
            point_if_stored(*e);
        }
    }

    /// Shape resolution only (a ChunkedSystem's full-range pass, host only, no allocation of
    /// what it owns beyond what a FromInput read needs, and no pre_forward): the tie aliases this
    /// Component owns, then every input, then every output.
    void resolve_shapes() {
        diag::count_resolve_shapes();
        ev_.resolving_ = nullptr;
        // Tie aliases first: the root's owner resolves an alias root's shape BEFORE
        // the entries below read it (and before allocation); a connected alias is checked once.
        resolve_alias_shapes();
        std::unordered_set<const detail::IOEntry*> done, active;
        // Inputs first: derivation sources are inputs only.
        for (detail::IOEntry* e : order_)
            if (e->kind() == IOKind::input) resolve(*e, done, active);
        for (detail::IOEntry* e : order_)
            if (e->kind() == IOKind::output) resolve(*e, done, active);
        ev_.resolving_ = nullptr;
    }

    void run_pre_compute(ExecCtx& owner_ctx) override {
        (void)owner_ctx;
        // REPLAY (a ChunkedSystem with a cached topology, inside its chunk loop): the shapes were applied from the
        // cache, the internal roots' index arrays were uploaded once and shifted per chunk (meta_shift_), and the
        // connections were validated when the topology was built. So no shape resolution, no per-root ragged
        // upload and no validate_connection here. pre_forward still runs (a manual reshape inside a chunked system
        // is not supported), roots are allocated (held roots are present), pending inits are applied and the
        // wrappers are re-pointed.
        if (!replay_) resolve_shapes();
        {
            detail::FlagScope scope{in_pre_forward_};
            pre_forward();
        }
        // MATERIALIZE every variable this Component touches that is not live (r01 §4.5): a free input takes
        // its init at the allocation; a variable live nowhere else is only pointed at. A pending Model::init_input
        // is applied here too. Then a ragged variable's device index arrays follow its (possibly changed)
        // description (version-gated, so unchanged metadata is not sent again).
        {
            std::vector<detail::Variable*> seen;
            for (detail::IOEntry* e : order_) {
                detail::Variable* v = &e->variable();
                if (std::find(seen.begin(), seen.end(), v) != seen.end()) continue;
                seen.push_back(v);
                v->materialize(queue_ref(), nullptr, !replay_);
                if (!replay_) e->root()->upload_ragged(queue_ref());
            }
        }
        for (detail::IOEntry* e : order_) {
            if (!replay_ && !e->local_shape()) e->check_constraints(false);  // specified dims against their supplier
            if (!replay_ && e->kind() == IOKind::input && !e->local_shape() &&
                (e->is_dynamic() || e->root()->is_dynamic()))
                e->validate_connection();
            point_if_stored(*e);
        }
    }

    /// The launch's events go on the storage of every port this Component touched, inputs and
    /// outputs, at this Component's placement (the one the launch ran on): later memory
    /// operations on that storage (the other-placement push below, a resize) are then ordered
    /// after the kernel by SlabKit, and a free waits on them (r01 §4.4). Tied members share a
    /// root, so each (root, placement) is registered once.
    void register_launch_events(std::span<const sycl::event> events) override {
        if (events.empty()) return;
        std::vector<detail::IOEntry*> seen;
        for (detail::IOEntry* e : order_) {
            detail::IOEntry* r = e->root();
            if (std::find(seen.begin(), seen.end(), r) != seen.end()) continue;
            seen.push_back(r);  // one placement per Component, so the root alone identifies the pair
            e->register_launch_events(events);
        }
        last_events_.assign(events.begin(), events.end());
    }
    std::vector<sycl::event> last_events_;  ///< the last launch's events (what collect_inflight reports)

    /// Refresh the other-placement copies of every output root this Component
    /// produces (routed copies: device<->host, cross-device), ordered after the
    /// launch by the stream. Nothing is freed here: a variable lives while a port holds its bundle.
    void run_post_compute(ExecCtx& owner_ctx) override {
        // Write epochs: every launch path ends here, with this Component's kernels submitted. Each OUTPUT's
        // root advances once per call (tied members of one root are separate Components, so N writers give +N).
        {
            std::vector<detail::IOEntry*> bumped;
            for (detail::IOEntry* e : order_) {
                if (e->kind() != IOKind::output) continue;
                detail::IOEntry* r = e->root();
                if (std::find(bumped.begin(), bumped.end(), r) != bumped.end()) continue;
                bumped.push_back(r);
                r->bump_write();
            }
        }
        // The other placements of every output variable this Component wrote follow the primary (once per variable;
        // the push is done on the variable's shadow, which holds the typed storage).
        {
            std::vector<detail::IOEntry*> synced;
            for (detail::IOEntry* e : order_) {
                if (e->kind() != IOKind::output) continue;
                detail::IOEntry* r = e->root();
                if (std::find(synced.begin(), synced.end(), r) != synced.end()) continue;
                synced.push_back(r);
                if (r->storage_placements().size() > 1 && r->has_storage()) r->sync_other_placements(*owner_ctx.queue);
            }
        }

    }

    // ---- what a fused owner needs to know, without knowing the KernelBody type --
    // Answered by Component<KB> from its body's signature. An owner reads
    // these; nothing here knows what fusing is.
    virtual int  kernel_rank() const { return 0; }       ///< 0 for custom-op and host bodies
    virtual bool is_group_body() const { return false; }
    virtual bool has_scratch() const { return false; }
    virtual bool is_custom_op() const { return false; }
    virtual bool is_host() const { return false; }
    /// The type-erased id of this body's stage function (kb_fn<KB>), for a
    /// call sequence. Throws for a custom-op or host body (nothing is compiled).
    virtual detail::JitId definition_id() const {
        throw std::logic_error("naga::cospan: " + describe() + " has no stage function");
    }

protected:
    ComponentBase() = default;

    /// The storage kind of this Component's I/O: device, or host for a host body.
    virtual sk::kind default_storage_kind() const { return sk::kind::device; }

    const sycl::device& device_ref() const {
        if (!device_) throw std::logic_error("naga::cospan: " + describe() + " has no device yet");
        return *device_;
    }
    sk::queue& queue_ref() const {
        if (queue_ == nullptr)
            throw std::logic_error("naga::cospan: " + describe() + " has no queue yet (prepare_exec)");
        return *queue_;
    }

    std::string describe() const { return path().empty() ? std::string{"component"} : "'" + path() + "'"; }

    std::optional<sycl::device> device_;
    sk::queue*                  queue_ = nullptr;
    Substrates*                 subs_  = nullptr;

private:
    /// System resolves paths into a Component's namespace and walks its
    /// entries (transient marking); nothing else reaches these.
    friend class System;
    friend class ChunkedSystem;

    detail::IOEntry* lookup_entry(std::string_view name) const {
        const auto it = by_name_.find(std::string{name});
        return it == by_name_.end() ? nullptr : it->second.get();
    }

    // ---- declaration -------------------------------------------------------
    template <int Rank, class S>
    static std::array<detail::DimRecord, Rank> records_of(
        const std::array<S, static_cast<std::size_t>(Rank)>& specs) {
        std::array<detail::DimRecord, Rank> out{};
        for (int d = 0; d < Rank; ++d) out[d] = detail::make_dim_record(specs[d]);
        return out;
    }

    template <DType DT, int Rank>
    void add_input(Input<DT, Rank>& m, std::string name,
                   const std::array<detail::DimRecord, Rank>& dims) {
        reject_per_index(name, dims);
        register_entry(std::make_unique<detail::InputEntry<DT, Rank>>(name, dims, &m), &m, "declare_input");
    }
    template <DType DT, int Rank>
    void add_output(Output<DT, Rank>& m, std::string name,
                    const std::array<detail::DimRecord, Rank>& dims) {
        reject_per_index(name, dims);
        register_entry(std::make_unique<detail::OutputEntry<DT, Rank>>(name, dims, &m), &m, "declare_output");
    }

    /// a DENSE wrapper cannot hold per-index dims (PerIndex, FromInputValues).
    template <std::size_t N>
    void reject_per_index(const std::string& name, const std::array<detail::DimRecord, N>& dims) const {
        for (const detail::DimRecord& r : dims)
            if (r.how == detail::DimRecord::How::per_index_host || r.how == detail::DimRecord::How::per_index_from_input)
                throw std::invalid_argument("naga::cospan: " + describe() + ": '" + name +
                                            "': per-index dims need RaggedInput/RaggedOutput");
    }

    void register_entry(std::unique_ptr<detail::IOEntry> e, const void* member, const char* what) {
        if (!declaring_)
            throw std::logic_error(std::string{"naga::cospan: "} + what + " is legal inside setup() only");
        const std::string key{e->name()};
        if (by_name_.contains(key))
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + key + "' is declared twice");
        if (by_member_.contains(member))
            throw std::invalid_argument("naga::cospan: " + describe() + ": member wrapper of '" + key +
                                        "' is already declared");
        e->declarer_ = this;  // System's transient marking walks declarer -> owner (§4.4)
        order_.push_back(e.get());
        by_member_.emplace(member, e.get());
        by_name_.emplace(key, std::move(e));
    }

    // ---- lookup -----------------------------------------------------------
    detail::IOEntry& find_kind(std::string_view name, IOKind k) {
        const auto it = by_name_.find(std::string{name});
        if (it == by_name_.end())
            throw std::out_of_range("naga::cospan: " + describe() + " declares no '" + std::string{name} + "'");
        if (it->second->kind() != k)
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + std::string{name} +
                                        "' is an " + (k == IOKind::input ? "output" : "input"));
        return *it->second;
    }

    std::invalid_argument wrong_type(std::string_view name) const {
        return std::invalid_argument("naga::cospan: " + describe() + ": '" + std::string{name} +
                                     "' was asked for with the wrong dtype or rank");
    }

    EntryRefs entries_of(IOKind k) const {
        EntryRefs out;
        for (const detail::IOEntry* e : order_)
            if (e->kind() == k) out.emplace_back(*e);
        return out;
    }

    // ---- phase guards ------------------------------------------------------
    template <class W>
    void reshape_member(const W* member, sycl::range<W::rank> shape) {
        if (!in_pre_forward_)
            throw std::logic_error("naga::cospan: reshape is legal inside pre_forward() only");
        const auto it = by_member_.find(member);
        if (it == by_member_.end())
            throw std::invalid_argument("naga::cospan: " + describe() + ": reshape of a member that was never declared");
        std::array<std::size_t, 3> ext{1, 1, 1};
        for (int d = 0; d < W::rank; ++d) ext[static_cast<std::size_t>(d)] = shape[d];
        it->second->reshape(sycl::range<3>{ext[0], ext[1], ext[2]});
    }

    // ---- storage helpers ---------------------------------------------------
    static bool has_registered(const detail::IOEntry& root, const Placement& p) {
        for (const Placement& e : root.storage_placements())
            if (e == p) return true;
        return false;
    }

    void point_if_stored(detail::IOEntry& e) {
        // only when the variable is live at THIS port's placement (a rebuilt port may sit on a placement that is not live yet)
        if (e.root()->has_storage() && e.variable().live_bundle(e.placement())) e.point_wrapper();
    }

    // ---- shape derivation (r01 §6.5 step 1) ----------------------------------
    detail::IOEntry& sibling_input(const detail::IOEntry& of, const std::string& name) {
        const auto it = by_name_.find(name);
        if (it == by_name_.end() || it->second->kind() != IOKind::input)
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + std::string{of.name()} +
                                        "' derives its shape from '" + name + "', which is not an input of this component");
        return *it->second;
    }

    /// Resolve `e`'s dimensions: its sources first (recursively), then apply.
    void resolve(detail::IOEntry& e, std::unordered_set<const detail::IOEntry*>& done,
                 std::unordered_set<const detail::IOEntry*>& active) {
        if (done.contains(&e)) return;
        ev_.resolving_ = &e;
        if (!active.insert(&e).second)
            throw std::logic_error("naga::cospan: " + describe() + ": shape derivation cycle through '" +
                                   std::string{e.name()} + "'");
        for (int d = 0; d < e.rank(); ++d) {
            detail::DimRecord& r = e.dims_[static_cast<std::size_t>(d)];
            // Inside a ChunkedSystem dim 0 is the chunk length: its declared rule is not evaluated.
            if (d == 0 && e.dim0_override_) continue;
            if (r.how == detail::DimRecord::How::from_input_dim) {
                detail::IOEntry& src = sibling_input(e, r.input);
                resolve(src, done, active);
                if (r.dim >= src.rank())
                    throw std::out_of_range("naga::cospan: '" + std::string{e.name()} +
                                            "' derives from a dimension past the rank of '" + r.input + "'");
                if (detail::dim_pending(src, r.dim)) throw detail::shape_needed(e.full_path(), "dimension " + std::to_string(d), src);
                r.extent = src.extents()[r.dim];
                // a per-index source dimension passes its array on (dim 0 is never per-index).
                if (d >= 1) {
                    r.per_index = src.per_index_raw(r.dim);
                    r.per       = src.dim(r.dim).per;
                }
            } else if (r.how == detail::DimRecord::How::per_index_from_input) {
                detail::IOEntry& src = sibling_input(e, r.input);
                resolve(src, done, active);
                ev_.resolving_  = &e;
                r.per_index = ev_.read_values(src, r.value_index);
                r.per       = true;
                r.extent    = 0;
            } else if (r.how == detail::DimRecord::How::from_input) {
                detail::IOEntry& src = sibling_input(e, r.input);
                resolve(src, done, active);
                ev_.resolving_ = &e;
                r.extent   = ev_.read_value(src, r.value_index);
            }
        }
        e.update_shape();
        if (e.is_ragged() && !e.is_ragged_port())
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + std::string{e.name()} +
                                        "' is ragged; declare it with RaggedInput/RaggedOutput");
        active.erase(&e);
        done.insert(&e);
    }

    // ---- tie aliases ------------------------------------------------
    /// For each tie alias one of this Component's entries belongs to: a connected tie_inputs alias
    /// is checked once against the output it joined. (A tie alias that is a ROOT is resolved by its
    /// owner, the System that declared it: System::materialize.)
    void resolve_alias_shapes() {
        std::vector<detail::IOEntry*> seen;
        for (detail::IOEntry* e : order_) {
            detail::IOEntry* a = e->tie_alias_;
            if (a == nullptr || !a->alias_shape_ || std::find(seen.begin(), seen.end(), a) != seen.end()) continue;
            seen.push_back(a);
            const AliasShape& s = *a->alias_shape_;
            if (a->local_shape()) continue;  // a tie root is resolved and allocated by its owner, the declaring System
            if (!a->alias_checked_) {
                if (s.kind != AliasShape::Kind::unspecified) {
                    const detail::AliasEvaluator::AliasEval want = ev_.alias_eval(*a);
                    const detail::IOEntry& root = *a->root();
                    bool                   same = true;
                    std::string            got_str, why;
                    if (root.is_ragged() || want.per[1] || want.per[2]) {
                        // Scalar dims compare by extent; per-index dims by their ARRAYS (length = dim 0, then element
                        // by element), never by extent.
                        std::array<std::size_t, 3> got{1, 1, 1};
                        std::array<bool, 3>        got_per{false, false, false};
                        for (int d = 0; d < a->rank(); ++d) {
                            const auto u = static_cast<std::size_t>(d);
                            got[u]       = root.extents()[u];
                            got_per[u]   = root.dim_is_per(d);
                            if (got_per[u]) got[u] = 0;
                            if (!same) continue;  // report the first difference only
                            if (want.per[u] != got_per[u]) {
                                same = false;
                                why  = "dimension " + std::to_string(d) + " is per-index on the " +
                                       (want.per[u] ? "alias" : "connection") + " only";
                            } else if (!want.per[u]) {
                                if (want.ext[u] != got[u]) {
                                    same = false;
                                    why  = "dimension " + std::to_string(d) + ": the alias declares " + std::to_string(want.ext[u]) +
                                          ", the connection has " + std::to_string(got[u]);
                                }
                            } else {
                                const std::vector<std::size_t> have = root.per_index_extents(d);
                                const std::vector<std::size_t>& mine = want.per_index[u];
                                if (mine.size() != have.size()) {
                                    same = false;
                                    why  = "per-index dimension " + std::to_string(d) + ": the alias has " + std::to_string(mine.size()) +
                                          " entries, the connection " + std::to_string(have.size());
                                } else {
                                    for (std::size_t k = 0; k < mine.size(); ++k)
                                        if (mine[k] != have[k]) {
                                            same = false;
                                            why  = "per-index dimension " + std::to_string(d) + " differs at index " +
                                                  std::to_string(k) + ": the alias declares " + std::to_string(mine[k]) +
                                                  ", the connection has " + std::to_string(have[k]);
                                            break;
                                        }
                                }
                            }
                        }
                        got_str = detail::AliasEvaluator::alias_shape_str(got, got_per, a->rank());
                    } else {
                        const auto v   = a->view();
                        const auto got = v ? v->extents : root.geometry().shape;
                        same           = want.ext == got;
                        got_str        = detail::AliasEvaluator::alias_shape_str(got, {false, false, false}, 3);
                    }
                    if (!same)
                        throw std::invalid_argument(
                            "naga::cospan: tie alias '" + a->full_path() + "' declares the shape " +
                            detail::AliasEvaluator::alias_shape_str(want.ext, want.per, root.is_ragged() || want.per[1] || want.per[2] ? a->rank() : 3) +
                            " but is connected to '" + std::string{root.name()} + "', whose shape is " + got_str +
                            (why.empty() ? std::string{} : " (" + why + ")"));
                }
                // A shape that reads other entries (values or dims) can change between forwards: check it again
                // every forward. A static one (fixed, per_index host arrays) is checked once.
                if (a->alias_sources().empty()) a->alias_checked_ = true;
            }
        }
    }

    std::unordered_map<std::string, std::unique_ptr<detail::IOEntry>> by_name_;
    std::unordered_map<const void*, detail::IOEntry*>                 by_member_;
    std::vector<detail::IOEntry*>                                     order_;  // declaration order

    bool declaring_       = false;
    bool in_pre_forward_  = false;
    bool chunk_inside_    = false;                       ///< this Component is inside a ChunkedSystem (set at its classification)
    detail::AliasEvaluator ev_;                          ///< alias-shape evaluation, value/shape reads, the chunk-pass recorders
    bool replay_          = false;                       ///< a ChunkedSystem replays its cached topology: run_pre_compute applies, never resolves
};

// -----------------------------------------------------------------------------
// detail::KernelLaunch<KB> — the individual description of a kernel body
// -----------------------------------------------------------------------------

namespace detail {

struct NoScratch {};

template <class KB>
struct scratch_of {
    using type = NoScratch;
};
template <class KB>
    requires HasScratch<KB>
struct scratch_of<KB> {
    using type = typename KB::Scratch;
};
template <class KB>
using scratch_t = typename scratch_of<KB>::type;

/// The Scratch argument of a body that has none.
template <class KB>
scratch_t<KB> NoScratchValue() {
    return scratch_t<KB>{};
}

template <int R>
sycl::range<3> pad3(const sycl::range<R>& r) {
    sycl::range<3> out{1, 1, 1};
    for (int d = 0; d < R; ++d) out[d] = r[d];
    return out;
}

/// A kernel-body launch submitted on its own: the record and the typed
/// Inputs/Outputs/body/Scratch are captured BY VALUE in the outer kernel; there
/// is no staging allocation (fused waves stage through device buffers instead).
template <class KB>
class KernelLaunch final : public LaunchDescription {
public:
    static constexpr int R = kernel_rank_v<KB>;
    using Scratch          = scratch_t<KB>;
    using Config           = sycl::AdaptiveCpp_jit::dynamic_function_config;

    KernelLaunch(Config* cfg, sycl::range<R> range, std::optional<sycl::range<R>> wg,
                 typename KB::Inputs in, typename KB::Outputs out, KB body, Scratch scratch)
        : cfg_{cfg}, payload_{StageRecord{}, in, out, BodyPack{body, scratch}} {
        wg3_ = wg ? pad3<R>(*wg) : default_wg<R>();
        StageRecord& r = payload_.rec;
        r.range        = pad3<R>(range);
        r.flat_extent  = range.size();
        r.rank         = static_cast<uint8_t>(R);
        const std::size_t wg_size = wg3_[0] * wg3_[1] * wg3_[2];
        if constexpr (GroupBody<KB>) {
            r.group_range = sycl::range<3>{(r.range[0] + wg3_[0] - 1) / wg3_[0],
                                           (r.range[1] + wg3_[1] - 1) / wg3_[1],
                                           (r.range[2] + wg3_[2] - 1) / wg3_[2]};
            r.group_count = r.group_range[0] * r.group_range[1] * r.group_range[2];
        } else {
            r.group_count = (r.flat_extent + wg_size - 1) / wg_size;
            r.group_range = sycl::range<3>{r.group_count, 1, 1};
        }
        if constexpr (HasScratch<KB>) {
            scratch_bytes_  = layout_scratch(payload_.body.scratch);
            r.scratch_bytes = scratch_bytes_;
            r.scratch_off   = static_cast<uint32_t>(
                reinterpret_cast<const std::byte*>(&payload_.body.scratch) -
                reinterpret_cast<const std::byte*>(&payload_.body));
        }
        parts_.rank        = R;
        parts_.group_body  = GroupBody<KB>;
        parts_.has_scratch = HasScratch<KB>;
        parts_.record      = r;  // offsets are 0-relative to the spans below
        parts_.inputs      = {reinterpret_cast<const std::byte*>(&payload_.in), sizeof(payload_.in)};
        parts_.outputs     = {reinterpret_cast<const std::byte*>(&payload_.out), sizeof(payload_.out)};
        parts_.body        = {reinterpret_cast<const std::byte*>(&payload_.body), sizeof(payload_.body)};
        if constexpr (HasScratch<KB>)
            parts_.scratch = {reinterpret_cast<const std::byte*>(&payload_.body.scratch),
                              sizeof(Scratch)};
        if (wg) parts_.wg = wg3_;
    }

    KernelLaunch(const KernelLaunch&)            = delete;
    KernelLaunch& operator=(const KernelLaunch&) = delete;

    bool is_custom_op() const override { return false; }
    bool is_host() const override { return false; }
    const KernelParts* kernel() const override { return &parts_; }

    std::vector<sycl::event> submit(SubmitTarget& t) const override {
        if (payload_.rec.flat_extent == 0) return {};  // nothing to run for an empty range
        return {launch_individual<R>(t, *cfg_, wave_nd_range<R>(payload_.rec.group_count, wg3_), payload_,
                                     scratch_bytes_)};
    }

private:
    struct BodyPack {
        KB      body;
        Scratch scratch;
    };
    struct Payload {
        StageRecord           rec;
        typename KB::Inputs   in;
        typename KB::Outputs  out;
        BodyPack              body;
    };
    static_assert(std::is_trivially_copyable_v<Payload>);

    Config*        cfg_;
    Payload        payload_;
    sycl::range<3> wg3_{1, 1, 1};
    uint32_t       scratch_bytes_ = 0;
    KernelParts    parts_;
};

/// A custom-operation launch (D1): the body runs at submission as an
/// AdaptiveCpp custom operation with an interop_handle; it enqueues native
/// backend work on the native stream. Inputs/Outputs/body are captured by
/// value; nothing is compiled.
template <class KB>
class CustomOpLaunch final : public LaunchDescription {
public:
    CustomOpLaunch(typename KB::Inputs in, typename KB::Outputs out, KB body)
        : in_{in}, out_{out}, body_{body} {}

    bool is_custom_op() const override { return true; }
    bool is_host() const override { return false; }

    std::vector<sycl::event> submit(SubmitTarget& t) const override {
        const typename KB::Inputs  in   = in_;
        const typename KB::Outputs out  = out_;
        const KB                   body = body_;
        auto cgf = [=](sycl::handler& h) {
            h.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle& ih) {
                CustomOp op{ih};
                body(op, in, out);
            });
        };
        if (t.group != nullptr) return {t.group->submit(t.device, cgf)};
        return {t.queue->submit(t.device, cgf)};
    }

private:
    typename KB::Inputs  in_;
    typename KB::Outputs out_;
    KB                   body_;
};

/// A host launch (D3): the body runs on the host as an ordered host task.
/// Its Inputs/Outputs point at host-kind USM on the component's device.
/// Without a group the task joins the stream's kernel order
/// (async_host_in_order); under a group, group.async_host (a group has no
/// in-order variant; ordering against kernel tasks is the group's).
template <class KB>
class HostLaunch final : public LaunchDescription {
public:
    HostLaunch(typename KB::Inputs in, typename KB::Outputs out, KB body)
        : in_{in}, out_{out}, body_{body} {}

    bool is_custom_op() const override { return false; }
    bool is_host() const override { return true; }

    std::vector<sycl::event> submit(SubmitTarget& t) const override {
        const typename KB::Inputs  in   = in_;
        const typename KB::Outputs out  = out_;
        const KB                   body = body_;
        auto f = [=]() { body(Host{}, in, out); };
        diag::count_host_tasks();
        if (t.group != nullptr) return {t.group->async_host(t.device, f)};
        return {t.queue->async_host_in_order(t.device, f)};
    }

private:
    typename KB::Inputs  in_;
    typename KB::Outputs out_;
    KB                   body_;
};

}  // namespace detail

// -----------------------------------------------------------------------------
// Component<KB>
// -----------------------------------------------------------------------------

/// One KernelBody type per component: the one-kernel rule, by type.
template <KernelBodyC KB>
class Component : public ComponentBase {
public:
    /// USER, every forward: assemble Inputs/Outputs (and body config, and
    /// Scratch shapes) from the already-synced members; state the launch
    /// parameters. Returns the description the owner will submit.
    virtual std::unique_ptr<LaunchDescription> compute_forward() = 0;

    /// The individual dynamic_function_config: built once, here. A custom-op
    /// or host body has nothing to compile.
    void compile() override {
        if constexpr (CustomOpBody<KB> || HostBody<KB>) {
            // nothing: no kernel is JIT-specialised for these bodies
        } else {
            constexpr int R = kernel_rank_v<KB>;
            cfg_            = std::make_shared<Config>();
            if constexpr (R == 1)
                cfg_->define(&kb_entry_r1, &kb_fn<KB>);
            else if constexpr (R == 2)
                cfg_->define(&kb_entry_r2, &kb_fn<KB>);
            else
                cfg_->define(&kb_entry_r3, &kb_fn<KB>);
        }
    }

    std::unique_ptr<LaunchDescription> run_compute_forward() override { return compute_forward(); }

    // ---- read off the body's signature, for a fused owner --------------------
    int  kernel_rank() const override { return kernel_rank_v<KB>; }
    bool is_group_body() const override { return GroupBody<KB>; }
    bool has_scratch() const override { return HasScratch<KB>; }
    bool is_custom_op() const override { return CustomOpBody<KB>; }
    bool is_host() const override { return HostBody<KB>; }
    detail::JitId definition_id() const override {
        if constexpr (CustomOpBody<KB> || HostBody<KB>)
            throw std::logic_error("naga::cospan: " + describe() +
                                   " is a custom-op or host body: it has no stage function");
        else
            return detail::stage_definition_id<KB>();
    }

protected:
    static constexpr int LR = launch_rank_v<KB>;

    /// Kernel bodies: range, Inputs, Outputs, body (default work-group).
    std::unique_ptr<LaunchDescription> create_launch_config(sycl::range<LR> range,
                                                            typename KB::Inputs in,
                                                            typename KB::Outputs out, KB body = {})
        requires((PlainBody<KB> || GroupBody<KB>) && !HasScratch<KB>)
    {
        return make_launch(range, std::nullopt, in, out, body, detail::NoScratchValue<KB>());
    }

    /// Group bodies without Scratch, with an explicit work-group shape.
    std::unique_ptr<LaunchDescription> create_launch_config(sycl::range<LR> range, sycl::range<LR> wg,
                                                            typename KB::Inputs in,
                                                            typename KB::Outputs out, KB body = {})
        requires(GroupBody<KB> && !HasScratch<KB>)
    {
        return make_launch(range, wg, in, out, body, detail::NoScratchValue<KB>());
    }

    /// Group bodies with Scratch: the Scratch is laid out here.
    std::unique_ptr<LaunchDescription> create_launch_config(sycl::range<LR> range, sycl::range<LR> wg,
                                                            typename KB::Inputs in,
                                                            typename KB::Outputs out,
                                                            detail::scratch_t<KB> scratch,
                                                            KB body = {})
        requires HasScratch<KB>
    {
        return make_launch(range, wg, in, out, body, scratch);
    }

    /// Custom-op bodies (D1): no range; the body enqueues native work.
    std::unique_ptr<LaunchDescription> create_launch_config(typename KB::Inputs in,
                                                            typename KB::Outputs out, KB body = {})
        requires CustomOpBody<KB>
    {
        return std::make_unique<detail::CustomOpLaunch<KB>>(in, out, body);
    }

    /// Host bodies (D3): no range; the body runs on the host over host-kind USM.
    std::unique_ptr<LaunchDescription> create_launch_config(typename KB::Inputs in,
                                                            typename KB::Outputs out, KB body = {})
        requires HostBody<KB>
    {
        return std::make_unique<detail::HostLaunch<KB>>(in, out, body);
    }

    sk::kind default_storage_kind() const override {
        return HostBody<KB> ? sk::kind::host : sk::kind::device;
    }

private:
    using Config = sycl::AdaptiveCpp_jit::dynamic_function_config;

    std::unique_ptr<LaunchDescription> make_launch(sycl::range<LR> range,
                                                   std::optional<sycl::range<LR>> wg,
                                                   typename KB::Inputs in, typename KB::Outputs out,
                                                   KB body, detail::scratch_t<KB> scratch) {
        if (!cfg_)
            throw std::logic_error("naga::cospan: " + describe() +
                                   ": create_launch_config before compile()");
        return std::make_unique<detail::KernelLaunch<KB>>(cfg_.get(), range, wg, in, out, body,
                                                          scratch);
    }

    std::shared_ptr<Config> cfg_;

public:
    void collect_inflight(std::vector<Subsystem::InflightKeep>& out) const override {
        if (cfg_ && !this->last_events_.empty()) out.emplace_back(this->last_events_, cfg_);
    }
};

}  // namespace naga::cospan