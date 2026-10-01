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

#include "naga/cospan/dtype.hpp"
#include "naga/cospan/host_view.hpp"
#include "naga/cospan/init.hpp"
#include "naga/cospan/kernel.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/wrappers.hpp"

namespace naga::cospan {

namespace detail {

constexpr bool is_integer_dtype(DType dt) noexcept {
    const DType v = value_dtype(dt);
    return v != DType::f32 && v != DType::f64;
}

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

    /// The array forms: from_source_shape<R>() and from_input_shape<R>(name).
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

    /// The rank-only form (D139 part 2): the entry has a dtype and a rank (the wrapper's) and no
    /// shape; its shape is its root's, and its root must be a tie alias (System::tie_inputs /
    /// tie_outputs). An untied rank-only entry is a setup error.
    template <DType DT, int Rank>
    void declare_input(Input<DT, Rank>& m, std::string name, Tied) {
        add_input<DT, Rank>(m, std::move(name), detail::tied_dims<Rank>());
    }
    template <DType DT, int Rank>
    void declare_output(Output<DT, Rank>& m, std::string name, Tied) {
        add_output<DT, Rank>(m, std::move(name), detail::tied_dims<Rank>());
    }

    /// The rank-only form of a ragged port (D140): a member of a tie whose alias gives per-index dims
    /// (cs::alias::dims with per_index / output_values / output_dim, or a ragged from_output_shape /
    /// from_connection) must be declared with RaggedInput / RaggedOutput.
    template <DType DT, int Rank>
    void declare_input(RaggedInput<DT, Rank>& m, std::string name, Tied) {
        register_entry(std::make_unique<detail::RaggedInputEntry<DT, Rank>>(name, detail::tied_dims<Rank>(), &m), &m,
                       "declare_input");
    }
    template <DType DT, int Rank>
    void declare_output(RaggedOutput<DT, Rank>& m, std::string name, Tied) {
        register_entry(std::make_unique<detail::RaggedOutputEntry<DT, Rank>>(name, detail::tied_dims<Rank>(), &m), &m,
                       "declare_output");
    }

    /// Ragged ports (D140): RaggedInput / RaggedOutput members, Rank 2..3, with the same DimSpec
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

    // ---- initial values (legal inside setup() only; r01 §4.4, D138) -----------
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
            if (e->is_root()) {
                e->set_primary_placement(own);
                bytes += e->static_bytes();
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
        for (detail::IOEntry* e : order_)
            if (e->kind() == IOKind::input && !e->is_root() && has_source_dim(*e))
                e->update_shape();  // throws for an unconnected input with a FromSourceDim
        // No allocation at setup: every root is allocated, and its init applied, by its OWNER in
        // run_pre_compute (r01 §4.4, D138 amendment). Roots only learn what to allocate from.
        for (detail::IOEntry* e : order_)
            if (e->is_root()) e->bind_substrates(substrates);
        for (detail::IOEntry* e : order_) {
            if (e->kind() == IOKind::input && !e->is_root() && !e->is_dynamic() &&
                !e->root()->is_dynamic())
                e->validate_connection();
            point_if_stored(*e);
        }
    }

    /// Shape resolution only (D140 B2: a ChunkedSystem's full-range pass, host only, no allocation of
    /// what it owns beyond what a FromInput read needs, and no pre_forward): the tie aliases this
    /// Component owns, then every input, then every output.
    void resolve_shapes() {
        diag::count_resolve_shapes();
        resolving_ = nullptr;
        // Tie aliases first (D139 part 2): the root's owner resolves an alias root's shape BEFORE
        // the entries below read it (and before allocation); a connected alias is checked once.
        resolve_alias_shapes();
        std::unordered_set<const detail::IOEntry*> done, active;
        // Inputs first: derivation sources are inputs only.
        for (detail::IOEntry* e : order_)
            if (e->kind() == IOKind::input) resolve(*e, done, active);
        for (detail::IOEntry* e : order_)
            if (e->kind() == IOKind::output) resolve(*e, done, active);
        resolving_ = nullptr;
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
        // ALLOCATE every root this Component OWNS (inputs and outputs) that has no storage, and
        // apply a free input root's init at the allocation (r01 §4.4, D138 amendment). A root it
        // does not own is only pointed at, below.
        for (detail::IOEntry* e : order_) allocate_owned(*e->root());
        // D140: a ragged root's device index arrays follow its (possibly changed) description.
        if (!replay_)
            for (detail::IOEntry* e : order_)
                if (e->root()->owner() == this && e->root()->has_storage()) e->root()->upload_ragged(queue_ref());
        for (detail::IOEntry* e : order_) {
            if (!replay_ && e->kind() == IOKind::input && !e->is_root() &&
                (e->is_dynamic() || e->root()->is_dynamic()))
                e->validate_connection();
            point_if_stored(*e);
        }
    }

    /// Refresh the other-placement copies of every output root this Component
    /// produces (routed copies: device<->host, cross-device), ordered after the
    /// launch by the stream; then RELEASE the transient roots this Component is
    /// the last user of: one stream wait for the batch (synchronous frees, D135).
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
        for (detail::IOEntry* e : order_)
            if (e->kind() == IOKind::output && e->is_root() && e->storage_placements().size() > 1 &&
                e->has_storage())
                e->sync_other_placements(*owner_ctx.queue);

        std::vector<detail::IOEntry*> drop;
        for (detail::IOEntry* e : order_) {
            detail::IOEntry* r = e->root();
            if (r->transient() && r->last_user() == this && r->has_storage() && !r->held() &&  // D140 B3: held roots stay
                std::find(drop.begin(), drop.end(), r) == drop.end())
                drop.push_back(r);
        }
        if (drop.empty()) return;
        diag::count_queue_waits();
        owner_ctx.queue->wait();
        for (detail::IOEntry* r : drop) r->release_storage();
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

    /// D140: a DENSE wrapper cannot hold per-index dims (PerIndex, FromInputValues).
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

    static bool has_source_dim(const detail::IOEntry& e) {
        for (int d = 0; d < e.rank(); ++d)
            if (e.dim(d).how == detail::DimRecord::How::from_source_dim) return true;
        return false;
    }

    /// The OWNER's allocation of a root (r01 §4.4, D139): through the one allocation routine (a
    /// size-0 root becomes present with empty storage). For a free input root (an input root: it
    /// has no source): if it was just allocated its init is applied (else zeros); else if an init
    /// is pending (Model::init_input) the init is applied to the existing storage. Either way the
    /// pending flag is cleared. A root this Component does not own is left alone; a connected
    /// root (an output root) never gets an init applied.
    void allocate_owned(detail::IOEntry& root) {
        if (root.owner() != this) return;
        root.allocate_if_absent(queue_ref());
        if (root.kind() == IOKind::input && root.init_pending()) {
            detail::apply_root_init(root, queue_ref());
            root.clear_init_pending();
        }
    }

    void point_if_stored(detail::IOEntry& e) {
        if (e.root()->has_storage()) e.point_wrapper();
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
        resolving_ = &e;
        if (!active.insert(&e).second)
            throw std::logic_error("naga::cospan: " + describe() + ": shape derivation cycle through '" +
                                   std::string{e.name()} + "'");
        for (int d = 0; d < e.rank(); ++d) {
            detail::DimRecord& r = e.dims_[static_cast<std::size_t>(d)];
            // Inside a ChunkedSystem dim 0 is the chunk length (D140 B3): its declared rule is not evaluated.
            if (d == 0 && e.dim0_override_) continue;
            if (r.how == detail::DimRecord::How::from_input_dim) {
                detail::IOEntry& src = sibling_input(e, r.input);
                resolve(src, done, active);
                if (r.dim >= src.rank())
                    throw std::out_of_range("naga::cospan: '" + std::string{e.name()} +
                                            "' derives from a dimension past the rank of '" + r.input + "'");
                r.extent = src.extents()[r.dim];
                // D140: a per-index source dimension passes its array on (dim 0 is never per-index).
                if (d >= 1) {
                    r.per_index = src.per_index_raw(r.dim);
                    r.per       = src.dim(r.dim).per;
                }
            } else if (r.how == detail::DimRecord::How::per_index_from_input) {
                detail::IOEntry& src = sibling_input(e, r.input);
                resolve(src, done, active);
                resolving_  = &e;
                r.per_index = read_values(src, r.value_index);
                r.per       = true;
                r.extent    = 0;
            } else if (r.how == detail::DimRecord::How::from_input) {
                detail::IOEntry& src = sibling_input(e, r.input);
                resolve(src, done, active);
                resolving_ = &e;
                r.extent   = read_value(src, r.value_index);
            }
        }
        e.update_shape();
        if (e.is_ragged() && !e.is_ragged_port())
            throw std::invalid_argument("naga::cospan: " + describe() + ": '" + std::string{e.name()} +
                                        "' is ragged; declare it with RaggedInput/RaggedOutput");
        active.erase(&e);
        done.insert(&e);
    }

    // ---- tie aliases (D139 part 2) ------------------------------------------------
    /// A tie alias's shape as its AliasShape gives it right now (D140: scalar extents, and per-index arrays
    /// for dims >= 1 that are per-index; the extent of a per-index dim is 0).
    struct AliasEval {
        std::array<std::size_t, 3>              ext{1, 1, 1};
        std::array<bool, 3>                     per{false, false, false};
        std::array<std::vector<std::size_t>, 3> per_index;
    };

    /// Inside a ChunkedSystem an internal alias holds ONE chunk [start, start + len) of the leading dimension: a
    /// per-index array that covers the whole run (a boundary root's) is cut to it; one that is already
    /// chunk-sized (an inside entry's, resolved under the chunk) is kept as it is.
    std::vector<std::size_t> chunk_slice(const detail::IOEntry& a, std::vector<std::size_t> v) const {
        if (!a.dim0_override_ || v.size() == *a.dim0_override_) return v;
        const std::size_t first = a.dim0_start_, len = *a.dim0_override_;
        if (v.size() < first + len)
            throw std::invalid_argument("naga::cospan: tie alias '" + a.full_path() + "': a per-index source has " +
                                        std::to_string(v.size()) + " entries but the chunk reaches index " +
                                        std::to_string(first + len));
        return std::vector<std::size_t>(v.begin() + static_cast<std::ptrdiff_t>(first),
                                        v.begin() + static_cast<std::ptrdiff_t>(first + len));
    }

    /// Slot `to` of `out` takes dimension `d` of `up` (per-index too, if that dim is). SHAPE inheritance only:
    /// it never reads a value, so an internal entry of a ChunkedSystem is a fine source. `input`: `up` is an
    /// INPUT source, whose ROOT is read (the input's own shape may not be resolved yet); a sliced input
    /// contributes its view's extents.
    void copy_alias_dim(AliasEval& out, std::size_t to, const detail::IOEntry& src, int d, const detail::IOEntry& a,
                        bool input) const {
        if (input && src.view()) {
            note_shape_dep(src.root());
            out.ext[to] = src.view()->extents[static_cast<std::size_t>(d)];
            return;
        }
        const detail::IOEntry& up = input ? *src.root() : src;
        note_shape_dep(up.root());  // a SHAPE dependency of the topology cache
        if (up.dim_is_per(d)) {
            if (to == 0)
                throw std::invalid_argument("naga::cospan: a per-index dimension of '" + up.full_path() +
                                            "' cannot become dimension 0 of a tie alias");
            out.per[to]       = true;
            out.per_index[to] = chunk_slice(a, up.per_index_extents(d));
            out.ext[to]       = 0;
        } else {
            out.ext[to] = up.extents()[static_cast<std::size_t>(d)];
        }
    }

    /// What the alias's AliasShape gives right now. Every upstream output it reads is a producer that has
    /// already run (checked at setup: it is earlier in registration order); the value reads happen on the
    /// host from a snapshot, as FromInputValues does.
    AliasEval alias_eval(const detail::IOEntry& a) {
        const AliasShape& s = *a.alias_shape_;
        AliasEval         out;
        resolving_ = &a;  // names the alias in a "computed inside a chunked system" error
        switch (s.kind) {
            case AliasShape::Kind::fixed: out.ext = s.extents; break;
            case AliasShape::Kind::from_output_shape:
                for (int d = 0; d < a.rank(); ++d) copy_alias_dim(out, static_cast<std::size_t>(d), *a.alias_src_, d, a, false);
                break;
            case AliasShape::Kind::from_input_shape:
                for (int d = 0; d < a.rank(); ++d) copy_alias_dim(out, static_cast<std::size_t>(d), *a.alias_src_, d, a, true);
                break;
            case AliasShape::Kind::from_output_values:
                for (int d = 0; d < a.rank(); ++d)
                    out.ext[static_cast<std::size_t>(d)] = read_value(*a.alias_src_, static_cast<std::size_t>(d));
                break;
            case AliasShape::Kind::dims:
                for (int d = 0; d < a.rank(); ++d) {
                    const auto      u  = static_cast<std::size_t>(d);
                    const AliasDim& sp = s.dim_specs[u];
                    switch (sp.how) {
                        case AliasDim::How::fixed: out.ext[u] = sp.extent; break;
                        case AliasDim::How::per_index_host: {
                            std::vector<std::size_t> v = sp.values;
                            if (a.dim0_override_) {  // inside a ChunkedSystem: this chunk's slice of the host array
                                const std::size_t first = a.dim0_start_, len = *a.dim0_override_;
                                if (v.size() < first + len)
                                    throw std::invalid_argument("naga::cospan: tie alias '" + a.full_path() + "': per-index dimension " +
                                                                std::to_string(d) + " has " + std::to_string(v.size()) +
                                                                " extents but the chunk reaches index " + std::to_string(first + len));
                                v = std::vector<std::size_t>(v.begin() + static_cast<std::ptrdiff_t>(first),
                                                             v.begin() + static_cast<std::ptrdiff_t>(first + len));
                            }
                            out.per[u]       = true;
                            out.per_index[u] = std::move(v);
                            break;
                        }
                        // An input source is read through its ROOT (host snapshot), as an output is.
                        case AliasDim::How::output_value:
                            out.ext[u] = read_value(sp.input ? *sp.src->root() : *sp.src, sp.index);
                            break;
                        case AliasDim::How::output_values:
                            out.per[u]       = true;
                            out.per_index[u] = chunk_slice(a, read_values(sp.input ? *sp.src->root() : *sp.src, sp.index));
                            break;
                        case AliasDim::How::output_dim:
                            copy_alias_dim(out, u, *sp.src, static_cast<int>(sp.index), a, sp.input);
                            break;
                    }
                }
                break;
            case AliasShape::Kind::from_connection:
                throw std::logic_error("naga::cospan: '" + a.full_path() + "' takes its shape from a connection, which it does not have");
        }
        // A per-index dim has no scalar extent, whichever spec made it (copy_alias_dim sets 0 as well).
        for (std::size_t u = 0; u < 3; ++u)
            if (out.per[u]) out.ext[u] = 0;
        resolving_ = nullptr;
        return out;
    }

    static std::string alias_shape_str(const std::array<std::size_t, 3>& ext, const std::array<bool, 3>& per, int rank) {
        std::string out;
        for (int d = 0; d < rank; ++d) {
            if (d > 0) out += "x";
            out += per[static_cast<std::size_t>(d)] ? std::string{"(per-index)"} : std::to_string(ext[static_cast<std::size_t>(d)]);
        }
        return out;
    }

    /// A ROOT tie alias takes the shape its rule gives now. Per-index dims are resolved on the host and applied like
    /// any ragged root's: update_shape computes the block offsets and resizes the storage; the owner uploads the
    /// device copies (run_pre_compute) once the root is allocated.
    void apply_alias_root(detail::IOEntry& a) {
        const AliasEval ev = alias_eval(a);
        for (int d = 0; d < a.rank(); ++d) {
            const auto         u = static_cast<std::size_t>(d);
            detail::DimRecord& r = a.dims_[u];
            r.extent             = ev.ext[u];
            if (d >= 1) {
                r.per       = ev.per[u];
                r.per_index = ev.per_index[u];
            }
        }
        a.update_shape();
    }

    /// ChunkedSystem: resolve the shape of an OUTSIDE tie alias (a boundary root) from its rule alone, through this
    /// Component's alias resolution. Its value / shape reads go to `deps` / `shapes` (the topology cache) when given,
    /// and a source computed inside the chunked system is refused as in the full pass.
    void resolve_alias_root(detail::IOEntry& a, std::vector<detail::IOEntry*>* deps, std::vector<const detail::IOEntry*>* shapes) {
        dep_recorder_   = deps;
        shape_recorder_ = shapes;
        chunk_pass_     = true;
        const auto reset = [&] {
            dep_recorder_   = nullptr;
            shape_recorder_ = nullptr;
            chunk_pass_     = false;
        };
        try {
            apply_alias_root(a);
        } catch (...) {
            reset();
            throw;
        }
        reset();
    }

    /// For each tie alias one of this Component's entries belongs to: an alias that is a ROOT
    /// this Component owns gets its shape resolved (and its storage resized); a connected
    /// tie_inputs alias is checked once against the output it joined.
    void resolve_alias_shapes() {
        std::vector<detail::IOEntry*> seen;
        for (detail::IOEntry* e : order_) {
            detail::IOEntry* a = e->tie_alias_;
            if (a == nullptr || !a->alias_shape_ || std::find(seen.begin(), seen.end(), a) != seen.end()) continue;
            seen.push_back(a);
            const AliasShape& s = *a->alias_shape_;
            if (a->is_root()) {
                if (a->owner() != this) continue;
                if (s.kind == AliasShape::Kind::fixed && !a->dim0_override_) continue;
                // An OUTSIDE tie alias that is a boundary root of a chunked system is resolved ONCE by that system,
                // before its pass, from its own rule (not by an inside owner, once per chunk).
                if (chunk_inside_ && a->chunk_outside_tie_) continue;
                apply_alias_root(*a);
            } else if (!a->alias_checked_) {
                if (s.kind != AliasShape::Kind::from_connection) {
                    const AliasEval        want = alias_eval(*a);
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
                        got_str = alias_shape_str(got, got_per, a->rank());
                    } else {
                        const auto v   = a->view();
                        const auto got = v ? v->extents : root.geometry().shape;
                        same           = want.ext == got;
                        got_str        = alias_shape_str(got, {false, false, false}, 3);
                    }
                    if (!same)
                        throw std::invalid_argument(
                            "naga::cospan: tie alias '" + a->full_path() + "' declares the shape " +
                            alias_shape_str(want.ext, want.per, root.is_ragged() || want.per[1] || want.per[2] ? a->rank() : 3) +
                            " but is connected to '" + std::string{root.name()} + "', whose shape is " + got_str +
                            (why.empty() ? std::string{} : " (" + why + ")"));
                }
                // A shape that reads other entries (values or dims) can change between forwards: check it again
                // every forward. A static one (fixed, per_index host arrays) is checked once.
                if (a->alias_sources().empty()) a->alias_checked_ = true;
            }
        }
    }

    /// The VALUES of an integer-dtype input of rank 1 [N] (column 0) or rank 2 [N, k] (column < k),
    /// on the host: one per row (D140, FromInputValues). Negative values are an error.
    std::vector<std::size_t> read_values(detail::IOEntry& src, std::size_t column) {
        check_not_chunk_internal(src);
        note_value_dep(src.root());  // a VALUE dependency of the topology cache
        if (!detail::is_integer_dtype(src.dtype()))
            throw std::invalid_argument("naga::cospan: '" + std::string{src.name()} +
                                        "' drives per-index extents by value and must have an integer dtype");
        if (src.rank() > 2)
            throw std::invalid_argument("naga::cospan: '" + std::string{src.name()} +
                                        "' drives per-index extents by value and must be rank 1 [N] or rank 2 [N,k]");
        const std::size_t k = src.rank() == 2 ? src.extents()[1] : 1;
        if (column >= k)
            throw std::out_of_range("naga::cospan: '" + std::string{src.name()} + "' has no column " +
                                    std::to_string(column));
        allocate_owned(*src.root());
        if (!src.root()->has_storage())
            throw std::logic_error("naga::cospan: '" + std::string{src.name()} +
                                   "' drives per-index extents by value but has no storage yet");
        std::vector<std::size_t> out(src.extents()[0]);
        detail::dispatch_dtype(src.dtype(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            // Read through the root's cached host copy and its geometry (as take_snapshot does), building only
            // the small output: no whole-root temporary.
            const detail::IOEntry::Geometry g     = src.geometry();
            const std::vector<T>&           whole = detail::storage_of<T>(src).read_primary(queue_ref());
            const std::size_t               count = detail::element_count(g);
            for (std::size_t i = 0; i < out.size(); ++i) {
                if (i * k + column >= count)
                    throw std::out_of_range("naga::cospan: '" + std::string{src.name()} + "' has no element " +
                                            std::to_string(i * k + column));
                const std::size_t off = detail::element_offset(i * k + column, g);
                if (off >= whole.size())
                    throw std::out_of_range("naga::cospan: '" + src.full_path() +
                                            "' presents elements past its root's storage");
                const T v = whole[off];
                if constexpr (std::is_signed_v<T>)
                    if (v < 0)
                        throw std::invalid_argument("naga::cospan: '" + std::string{src.name()} +
                                                    "' holds a negative extent (row " + std::to_string(i) + ")");
                out[i] = static_cast<std::size_t>(v);
            }
        });
        return out;
    }

    /// D140 B2: during a ChunkedSystem's full-range pass a shape may not be driven by a value of a root
    /// that is internal to the chunked system (it would be computed per chunk).
    void check_not_chunk_internal(const detail::IOEntry& src) const {
        if (!chunk_pass_) return;
        // INTERNAL roots, and the re-rooted OUTPUT roots (an inside output promoted or consumed outside):
        // both are computed per chunk.
        const detail::IOEntry& root = *src.root();
        std::string            by{root.chunk_internal_by_};
        if (by.empty() && root.chunk_rerooted_ && root.kind() == IOKind::output)
            by = root.declarer() != nullptr ? root.declarer()->path() : std::string{"?"};
        if (by.empty()) return;
        const std::string who = resolving_ != nullptr ? std::string{resolving_->name()} : std::string{src.name()};
        throw std::invalid_argument("naga::cospan: '" + who + "' is sized by a value computed inside chunked system '" +
                                    by +
                                    "' (the value of '" + std::string{src.name()} +
                                    "'); chunked sizes must come from outside");
    }

    /// The VALUE of element `index` of an integer-dtype input, on the host.
    std::size_t read_value(detail::IOEntry& src, std::size_t index) {
        check_not_chunk_internal(src);
        note_value_dep(src.root());  // a VALUE dependency of the topology cache
        if (!detail::is_integer_dtype(src.dtype()))
            throw std::invalid_argument("naga::cospan: '" + std::string{src.name()} +
                                        "' drives a shape by value and must have an integer dtype");
        allocate_owned(*src.root());  // only when this Component owns it
        if (!src.root()->has_storage())
            throw std::logic_error("naga::cospan: '" + std::string{src.name()} +
                                   "' drives a shape by value but has no storage yet");
        std::size_t out = 0;
        detail::dispatch_dtype(src.dtype(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            // The root's cached host copy through the entry's geometry: one element, no whole-root temporary.
            const detail::IOEntry::Geometry g = src.geometry();
            if (index >= detail::element_count(g))
                throw std::out_of_range("naga::cospan: '" + std::string{src.name()} + "' has no element " +
                                        std::to_string(index));
            const std::vector<T>& whole = detail::storage_of<T>(src).read_primary(queue_ref());
            const std::size_t     off   = detail::element_offset(index, g);
            if (off >= whole.size())
                throw std::out_of_range("naga::cospan: '" + src.full_path() + "' presents elements past its root's storage");
            const T v = whole[off];
            if constexpr (std::is_signed_v<T>)
                if (v < 0)
                    throw std::invalid_argument("naga::cospan: '" + std::string{src.name()} +
                                                "' holds a negative extent");
            out = static_cast<std::size_t>(v);
        });
        return out;
    }

    std::unordered_map<std::string, std::unique_ptr<detail::IOEntry>> by_name_;
    std::unordered_map<const void*, detail::IOEntry*>                 by_member_;
    std::vector<detail::IOEntry*>                                     order_;  // declaration order

    bool declaring_       = false;
    bool in_pre_forward_  = false;
    bool chunk_pass_      = false;                       ///< a ChunkedSystem's full-range pass is running (D140 B2)
    bool chunk_inside_    = false;                       ///< this Component is inside a ChunkedSystem (set at its classification)
    bool replay_          = false;                       ///< a ChunkedSystem replays its cached topology: run_pre_compute applies, never resolves
    /// Set by the ChunkedSystem around resolve_shapes in its full pass: every value read (read_value / read_values,
    /// which every alias and port value read goes through) pushes the root it read; copy_alias_dim pushes the roots
    /// whose SHAPE it copied. Single-threaded, like all submission.
    /// One recorded read: the root read, the entry being resolved when it was read, and which kind of dependency.
    struct DepRead {
        const detail::IOEntry* root;
        const detail::IOEntry* by;
        bool                   shape;
    };
    mutable std::vector<DepRead> read_log_;  ///< the same reads with their readers (the ChunkedSystem's dependency self-check)
    void note_value_dep(detail::IOEntry* root) const {
        if (dep_recorder_ == nullptr) return;
        dep_recorder_->push_back(root);
        read_log_.push_back(DepRead{root, resolving_, false});
    }
    void note_shape_dep(const detail::IOEntry* root) const {
        if (shape_recorder_ == nullptr) return;
        shape_recorder_->push_back(root);
        read_log_.push_back(DepRead{root, resolving_, true});
    }
    std::vector<detail::IOEntry*>*       dep_recorder_   = nullptr;
    std::vector<const detail::IOEntry*>* shape_recorder_ = nullptr;
    const detail::IOEntry* resolving_ = nullptr;         ///< the entry resolve() is working on
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

    void submit(SubmitTarget& t) const override {
        if (payload_.rec.flat_extent == 0) return;  // nothing to run for an empty range
        launch_individual<R>(t, *cfg_, wave_nd_range<R>(payload_.rec.group_count, wg3_), payload_,
                             scratch_bytes_);
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

    void submit(SubmitTarget& t) const override {
        const typename KB::Inputs  in   = in_;
        const typename KB::Outputs out  = out_;
        const KB                   body = body_;
        auto cgf = [=](sycl::handler& h) {
            h.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle& ih) {
                CustomOp op{ih};
                body(op, in, out);
            });
        };
        if (t.group != nullptr)
            t.group->submit(t.device, cgf);
        else
            t.queue->submit(t.device, cgf);
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

    void submit(SubmitTarget& t) const override {
        const typename KB::Inputs  in   = in_;
        const typename KB::Outputs out  = out_;
        const KB                   body = body_;
        auto f = [=]() { body(Host{}, in, out); };
        diag::count_host_tasks();
        if (t.group != nullptr)
            t.group->async_host(t.device, f);
        else
            t.queue->async_host_in_order(t.device, f);
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
            cfg_            = std::make_unique<Config>();
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

    std::unique_ptr<Config> cfg_;
};

}  // namespace naga::cospan
