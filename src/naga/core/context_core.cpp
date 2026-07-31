#include "naga/core/context.hpp"
#include "naga/core/component.hpp"
#include "naga/core/kernel_profile.hpp"
#include <iostream>

namespace naga::core {

Context::Context(std::shared_ptr<slabkit::Context> sk_ctx, size_t device_index)
    : sk_ctx_(std::move(sk_ctx))
    , device_index_(device_index)
    , phase_(Phase::Declaring)
    , max_layer_width_(0)
{
    if (!sk_ctx_) {
        throw std::invalid_argument("Context: SkCtx cannot be null");
    }
    if (device_index_ >= sk_ctx_->devices.size()) {
        throw std::invalid_argument("Context: device_index out of range");
    }
    queue_pool_.push_back(make_pool_queue());
}

Context::~Context() {
    // Dump-and-reset while queues/runtime are alive; the dtor body runs before
    // member destruction, so the queue pool still exists here. Multiple
    // Contexts each dump their own era (stats are zeroed after each dump).
    if (prof::enabled()) {
        prof::dump_and_reset(std::cerr);
    }
}

sycl::queue Context::make_pool_queue() const {
    const auto& dev = sk_ctx_->devices[device_index_];
    if (prof::enabled()) {
        // Profiling needs real per-operation events with timestamps — keep
        // fine-grained events on this path.
        return sk_ctx_->make_queue_with_props(
            dev, sycl::property_list{sycl::property::queue::in_order(),
                                     sycl::property::queue::enable_profiling()});
    }
    // ACPP_EXT_COARSE_GRAINED_EVENTS: the framework discards every event
    // returned from submission (kernel.hpp choke point) and synchronizes via
    // queue::wait() on in-order queues only, so per-operation backend event
    // creation is pure overhead. Coarse-grained events let AdaptiveCpp elide
    // it, lowering kernel launch latency.
    return sk_ctx_->make_queue_with_props(
        dev, sycl::property_list{
                 sycl::property::queue::in_order(),
                 sycl::property::queue::AdaptiveCpp_coarse_grained_events()});
}

void Context::connect(std::string_view source_name, std::string_view target_name) {
    assert_phase(Phase::Declaring, "connect()");

    std::string src(source_name);
    std::string tgt(target_name);

    // Find source and target entries
    Entry* source = find_entry(src);
    Entry* target = find_entry(tgt);

    if (!source) {
        throw std::invalid_argument("Context::connect(): source '" + src + "' not found");
    }
    if (!target) {
        throw std::invalid_argument("Context::connect(): target '" + tgt + "' not found");
    }
    if (source->kind == EntryKind::InternalBuffer)
        throw std::invalid_argument(
            "Context::connect(): InternalBuffer entry '" + src + "' cannot be used as a connection source");
    if (target->kind == EntryKind::InternalBuffer)
        throw std::invalid_argument(
            "Context::connect(): InternalBuffer entry '" + tgt + "' cannot be used as a connection target");

    if (target->kind == EntryKind::Promotion && target->is_shareable)
        throw std::invalid_argument(
            "Context::connect(): target '" + tgt +
            "' is a shared-storage owner and cannot be connected directly");

    if (source->kind != EntryKind::Output
        && source->kind != EntryKind::PassthroughVar
        && source->kind != EntryKind::Promotion
        && source->kind != EntryKind::StateVar)
        throw std::invalid_argument(
            "Context::connect(): source '" + src + "' is not a connectable source entry");

    if (target->kind == EntryKind::StateVar)
        throw std::invalid_argument(
            "Context::connect(): cannot connect to StateVar entry '" + tgt +
            "' — StateVar entries are not connectable targets");

    if (target->kind != EntryKind::Input && target->kind != EntryKind::PassthroughVar && target->kind != EntryKind::Promotion)
        throw std::invalid_argument("Context::connect(): target '" + tgt + "' is not an input or passthrough var");

    // Rule: each input/passthrough can be a target of only one upstream
    if (target->is_target) {
        throw std::invalid_argument(
            "Context::connect(): target '" + tgt + "' is already connected from '" +
            target->sources[0] + "'");
    }

    // Validate size (not shape!)
    if (shape3_size(source->shape3) != shape3_size(target->shape3)) {
        throw std::invalid_argument("Context::connect(): size mismatch between '" + src +
                                  "' and '" + tgt + "' (" +
                                  std::to_string(shape3_size(source->shape3)) + " vs " +
                                  std::to_string(shape3_size(target->shape3)) + ")");
    }

    // Validate dtype — exact match required. A connection is pointer
    // aliasing; anything other than an exact element-type match is a lie.
    // Explicit casts are a user-authored cast component (FRAMEWORK_CLEANUP S1).
    if (source->dtype != target->dtype) {
        throw std::invalid_argument(
            std::string("Context::connect(): dtype mismatch between '") + src +
            "' (" + dtype_name(source->dtype) + ") and '" + tgt + "' (" +
            dtype_name(target->dtype) + ") — connections require exact dtype "
            "match; use an explicit cast component to convert");
    }

    // Log connection for debugging
    std::cerr << "[CONN] " << src << " -> " << tgt << "\n";

    // Record connection
    source->targets.push_back(tgt);
    target->sources.push_back(src);
    target->is_target = true;
    target->is_design_variable = false;  // target is driven by source — no longer independently controllable

    // NOTE: connecting does NOT unmark entry points. An entry point with
    // upstream connections is an interior register: its inputs read the
    // previous tick by contract (see mark_entry_point).
}

void Context::disconnect(std::string_view source_name, std::string_view target_name) {
    assert_phase(Phase::Declaring, "disconnect()");

    std::string src(source_name);
    std::string tgt(target_name);

    Entry* source = find_entry(src);
    Entry* target = find_entry(tgt);

    if (!source) {
        throw std::invalid_argument("Context::disconnect(): source '" + src + "' not found");
    }
    if (!target) {
        throw std::invalid_argument("Context::disconnect(): target '" + tgt + "' not found");
    }

    // Verify the connection exists
    bool found = false;
    for (auto it = source->targets.begin(); it != source->targets.end(); ++it) {
        if (*it == tgt) {
            source->targets.erase(it);
            found = true;
            break;
        }
    }
    if (!found) {
        throw std::invalid_argument(
            "Context::disconnect(): no connection from '" + src + "' to '" + tgt + "'");
    }

    // Remove source from target's sources list
    for (auto it = target->sources.begin(); it != target->sources.end(); ++it) {
        if (*it == src) {
            target->sources.erase(it);
            break;
        }
    }

    // Restore target's is_target flag
    target->is_target = !target->sources.empty();

    std::cerr << "[DISCONN] " << src << " -> " << tgt << "\n";
}

void Context::assign_objective(std::string_view name) {
    assert_phase(Phase::Declaring, "assign_objective()");

    std::string n(name);
    Entry* entry = find_entry(n);

    if (!entry) {
        throw std::invalid_argument("Context::assign_objective(): entry '" + n + "' not found");
    }
    if (entry->kind != EntryKind::Output) {
        throw std::invalid_argument("Context::assign_objective(): '" + n + "' is not an output");
    }

    // Validate scalar (size=1)
    size_t entry_size = shape3_size(entry->shape3);
    if (entry_size != 1) {
        throw std::invalid_argument(
            "Context::assign_objective(): objective must be scalar (size=1), entry '" +
            n + "' has size " + std::to_string(entry_size));
    }
    if (entry->dtype != DType::f32)
        throw std::invalid_argument(
            "Context::assign_objective(): '" + n + "' has dtype " +
            dtype_name(entry->dtype) + " — objectives are f32-only");

    entry->is_objective = true;
}

void Context::set_max_backward_iterations(size_t k) {
    assert_phase(Phase::Declaring, "set_max_backward_iterations()");
    if (k == 0) throw std::invalid_argument("K_backward must be >= 1");
    K_backward_ = k;
}

void Context::mark_entry_point(const std::string& component_name) {
    assert_phase(Phase::Declaring, "mark_entry_point()");
    // A component with upstream connections is a legal (interior) entry
    // point: marking it declares a register — all its incoming connections
    // will read the previous tick. A component with no upstream connections
    // is a pure execution root (the degenerate case).
    if (!component_has_io(component_name))
        throw std::invalid_argument(
            "mark_entry_point: '" + component_name +
            "' is not a registered component with I/O");
    entry_points_.insert(component_name);
}

void Context::unmark_entry_point(const std::string& component_name) {
    assert_phase(Phase::Declaring, "unmark_entry_point()");
    entry_points_.erase(component_name);
}

void Context::assign_design_var(std::string_view name) {
    assert_phase(Phase::Declaring, "assign_design_var()");
    std::string n(name);
    Entry* entry = find_entry(n);
    if (!entry)
        throw std::invalid_argument("Context::assign_design_var(): entry '" + n + "' not found");
    if (entry->kind == EntryKind::Output || entry->kind == EntryKind::PassthroughVar)
        throw std::invalid_argument("Context::assign_design_var(): entry '" + n
            + "' is an Output or PassthroughVar — only Input or Promotion entries can be design variables");
    if (entry->dtype != DType::f32)
        throw std::invalid_argument(
            "Context::assign_design_var(): '" + n + "' has dtype " +
            dtype_name(entry->dtype) +
            " — design variables are f32-only (DVs remain f32-only in v1)");
    entry->is_design_variable = true;
}

void Context::unassign_objective(std::string_view name) {
    assert_phase(Phase::Declaring, "unassign_objective()");
    std::string n(name);
    Entry* entry = find_entry(n);
    if (!entry) throw std::invalid_argument("Context::unassign_objective(): entry '" + n + "' not found");
    entry->is_objective = false;
}

void Context::unassign_design_var(std::string_view name) {
    assert_phase(Phase::Declaring, "unassign_design_var()");
    std::string n(name);
    Entry* entry = find_entry(n);
    if (!entry) throw std::invalid_argument("Context::unassign_design_var(): entry '" + n + "' not found");
    entry->is_design_variable = false;
}

void Context::freeze_dv(const std::string& name) {
    for (auto& entry : entries_) {
        if (entry.name == name) {
            if (!entry.is_design_variable) {
                throw std::invalid_argument(
                    "freeze_dv(): entry '" + name + "' is not a design variable");
            }
            entry.is_frozen = true;
            return;
        }
    }
    throw std::invalid_argument(
        "freeze_dv(): design variable '" + name + "' not found");
}

void Context::thaw_dv(const std::string& name) {
    for (auto& entry : entries_) {
        if (entry.name == name) {
            if (!entry.is_design_variable) {
                throw std::invalid_argument(
                    "thaw_dv(): entry '" + name + "' is not a design variable");
            }
            entry.is_frozen = false;
            return;
        }
    }
    throw std::invalid_argument(
        "thaw_dv(): design variable '" + name + "' not found");
}

void Context::freeze_all_dvs_except(const std::vector<std::string>& keep_thawed) {
    for (auto& entry : entries_) {
        if (entry.is_design_variable) {
            bool keep = false;
            for (const auto& name : keep_thawed) {
                if (entry.name == name) {
                    keep = true;
                    break;
                }
            }
            entry.is_frozen = !keep;
        }
    }
}

void Context::thaw_all_dvs() {
    for (auto& entry : entries_) {
        if (entry.is_design_variable) {
            entry.is_frozen = false;
        }
    }
}

bool Context::is_frozen(const std::string& name) const {
    for (const auto& entry : entries_) {
        if (entry.name == name && entry.is_design_variable) {
            return entry.is_frozen;
        }
    }
    throw std::invalid_argument(
        "is_frozen(): entry '" + name + "' not found");
}

void Context::register_component(Component* component_ptr,
                                 const std::string& component_name,
                                 const std::vector<IOMetadata>& inputs,
                                 const std::vector<IOMetadata>& outputs) {
    assert_phase(Phase::Declaring, "register_component()");
    if (inputs.empty() && outputs.empty()) return;
    ComponentNode node;
    node.ptr = component_ptr;
    node.name = component_name;
    node.inputs = inputs;
    node.outputs = outputs;
    component_nodes_.push_back(std::move(node));
}

void Context::patch_data(std::string_view name, float* new_data) {
    std::string n(name);
    Entry* entry = find_entry(n);

    if (!entry) {
        throw std::invalid_argument("Context::patch_data(): entry '" + n + "' not found");
    }
    require_dtype(*entry, DType::f32, "patch_data()");

    entry->data = new_data;
}

void Context::patch_grad(std::string_view name, float* new_grad) {
    std::string n(name);
    Entry* entry = find_entry(n);

    if (!entry) {
        throw std::invalid_argument("Context::patch_grad(): entry '" + n + "' not found");
    }
    require_dtype(*entry, DType::f32, "patch_grad()");

    entry->grad = new_grad;
}

std::vector<Context::EntrySummary> Context::entries() const {
    std::vector<EntrySummary> summaries;
    summaries.reserve(entries_.size());

    for (const auto& entry : entries_) {
        EntrySummary summary;
        summary.name = entry.name;
        summary.shape = entry.shape3;
        summary.size = shape3_size(entry.shape3);
        summary.is_input = (entry.kind == EntryKind::Input);
        summary.is_objective = entry.is_objective;
        summary.dtype = entry.dtype;
        summary.data = entry.data;
        summary.grad = entry.grad;
        summaries.push_back(summary);
    }

    return summaries;
}

bool Context::is_connected(std::string_view name) const {
    auto it = entry_map_.find(std::string(name));
    if (it == entry_map_.end()) return false;
    return entries_[it->second].is_target;
}

bool Context::component_has_io(const std::string& name) const {
    for (const auto& node : component_nodes_) {
        if (node.name == name) {
            return !node.inputs.empty() || !node.outputs.empty();
        }
    }
    return false;
}

std::string Context::owning_component(const std::string& fqn) const {
    std::string name = fqn;
    while (true) {
        auto pos = name.rfind('/');
        if (pos == std::string::npos) return "";
        name = name.substr(0, pos);
        if (component_has_io(name)) return name;
    }
}

std::vector<std::string> Context::recurrent_entries() const {
    std::vector<std::string> result;
    for (const auto& entry : entries_) {
        if (entry.is_target) continue;
        if (entry.is_recurrent_output || entry.kind == EntryKind::StateVar)
            result.push_back(entry.name);
    }
    return result;
}

void Context::assert_phase(Phase expected, const char* fn) const {
    if (phase_ != expected) {
        throw std::logic_error("Context::" + std::string(fn) + ": wrong phase");
    }
}

Context::Entry* Context::find_entry(const std::string& name) {
    auto it = entry_map_.find(name);
    if (it != entry_map_.end()) {
        return &entries_[it->second];
    }
    return nullptr;
}

const Context::Entry* Context::find_entry(const std::string& name) const {
    auto it = entry_map_.find(name);
    if (it != entry_map_.end()) {
        return &entries_[it->second];
    }
    return nullptr;
}

std::string Context::resolve_source_component(const std::string& src_fqn) const {
    // If this is a Promotion entry, follow sources[0] recursively before owning_component().
    auto pit = entry_map_.find(src_fqn);
    if (pit != entry_map_.end() && entries_[pit->second].kind == EntryKind::Promotion) {
        const Entry& pentry = entries_[pit->second];
        if (!pentry.sources.empty())
            return resolve_source_component(pentry.sources[0]);
    }
    // FQN walk-up first — catches direct component entries whose entries share
    // the component's namespace prefix.
    std::string owner = owning_component(src_fqn);
    if (!owner.empty()) return owner;
    // No component ancestor in naming hierarchy — follow sources[0] upstream
    // (system-level passthrough or top-level promoted var). FQN walk-up is
    // applied again at each step of the trace.
    auto it = entry_map_.find(src_fqn);
    if (it == entry_map_.end()) return "";
    const Entry& entry = entries_[it->second];
    if (!entry.sources.empty())
        return resolve_source_component(entry.sources[0]);
    return "";
}

std::vector<std::string> Context::resolve_target_components(const std::string& tgt_fqn) const {
    // If this is a Promotion entry, iterate targets recursively instead of calling owning_component().
    auto pit = entry_map_.find(tgt_fqn);
    if (pit != entry_map_.end() && entries_[pit->second].kind == EntryKind::Promotion) {
        const Entry& pentry = entries_[pit->second];
        std::vector<std::string> result;
        for (const auto& child_fqn : pentry.targets) {
            auto resolved = resolve_target_components(child_fqn);
            result.insert(result.end(), resolved.begin(), resolved.end());
        }
        return result;
    }
    // FQN walk-up first — catches direct component entries and spoofed components.
    std::string owner = owning_component(tgt_fqn);
    if (!owner.empty()) return {owner};
    // No component ancestor in naming hierarchy — walk DOWN targets
    // (system-level passthrough or top-level promoted var). FQN walk-up is
    // applied again at each step of the trace.
    auto it = entry_map_.find(tgt_fqn);
    if (it == entry_map_.end()) return {};
    const Entry& entry = entries_[it->second];
    std::vector<std::string> result;
    for (const auto& child_fqn : entry.targets) {
        auto resolved = resolve_target_components(child_fqn);
        result.insert(result.end(), resolved.begin(), resolved.end());
    }
    return result;
}

float* Context::resolve_read_data(const Entry& entry) const {
    const Entry& owner = get_owning_entry(entry);
    auto it = primal_slot_idx_.find(owner.name);
    if (it != primal_slot_idx_.end()) {
        // Per-connection read discipline: a read is deferred (previous sweep,
        // prev slot) iff THIS target entry's connection was cut — its
        // component is a marked entry point, or the traversal cut it to break
        // an unmarked cycle. Every other read is fresh (current sweep, curr
        // slot), ordered by the topological sweep. Routing on the TARGET
        // entry (not the owner) is what allows one source to feed a deferred
        // consumer and a fresh consumer simultaneously without either
        // inheriting the other's sweep semantics.
        // StateVars in their owning component use get_state_var() which
        // bypasses this path entirely and directly indexes both slots.
        if (entry.is_deferred_read) {
          return primal_slots_.at(owner.name)[it->second.prev].get();
        } else {
          return primal_slots_.at(owner.name)[it->second.curr].get();
        }
    }
    return owner.data;
}

float* Context::resolve_write_data(const Entry& entry) const {
    const Entry& owner = get_owning_entry(entry);
    auto it = primal_slot_idx_.find(owner.name);
    if (it != primal_slot_idx_.end()) {
        return primal_slots_.at(owner.name)[it->second.curr].get();
    }
    return owner.data;
}

float* Context::resolve_read_grad(const Entry& entry) const {
    const Entry& owner = get_owning_entry(entry);
    if (owner.is_recurrent_output || owner.kind == EntryKind::StateVar) {
        const auto& idx = grad_slot_idx_.at(owner.name);
        // Grad CURR slot: the total gradient consumed this adjoint iteration
        // (the transported iterate rotated in at advance, plus any
        // fresh-consumer contributions accumulated during this reverse sweep).
        return grad_slots_.at(owner.name)[idx.curr].get();
    }
    return owner.grad;
}

float* Context::resolve_accum_grad(const Entry& entry) const {
    const Entry& owner = get_owning_entry(entry);
    if (owner.is_recurrent_output || owner.kind == EntryKind::StateVar) {
        const auto& idx = grad_slot_idx_.at(owner.name);
        // The adjoint mirror of the primal read discipline, routed by the
        // TARGET entry's connection role:
        //   - back-edge (deferred) consumer → grad PREV slot: its
        //     contribution is the transport Jᵀt_k, consumed by the owner at
        //     the NEXT adjoint iteration (after rotation);
        //   - fresh consumer → grad CURR slot: consumed by the owner's
        //     backward in the SAME iteration (reverse layer order runs
        //     consumers before producers).
        // The old scheme routed ALL consumers to one accumulation slot,
        // which inserted a spurious one-iteration delay on every fresh
        // gradient path through a recurrent output (2026-07-20).
        if (entry.is_deferred_read)
            return grad_slots_.at(owner.name)[idx.prev].get();
        return grad_slots_.at(owner.name)[idx.curr].get();
    }
    return owner.grad;
}

void Context::reset_state_var(std::string_view fqn) {
    assert_phase(Phase::Finalized, "reset_state_var()");
    std::string n(fqn);
    auto it = entry_map_.find(n);
    if (it == entry_map_.end())
        throw std::invalid_argument(
            "Context::reset_state_var(): entry '" + n + "' not found");
    Entry& entry = entries_[it->second];
    if (entry.kind != EntryKind::StateVar)
        throw std::invalid_argument(
            "Context::reset_state_var(): '" + n + "' is not a StateVar");
    size_t count = shape3_size(entry.shape3);
    // Zero both primal slots via the standard primal_slots_ machinery
    auto& slots = primal_slots_.at(n);
    for (auto& slot : slots)
        queue_pool_[0].fill(slot.get(), 0.0f, count);
    queue_pool_[0].wait();
}

void Context::share_outputs(const std::string& owner_fqn,
                            const std::vector<std::string>& member_fqns) {
    assert_phase(Phase::Declaring, "share_outputs()");

    if (member_fqns.empty())
        throw std::invalid_argument(
            "Context::share_outputs(): no members given for owner '" +
            owner_fqn + "'");

    if (entry_map_.find(owner_fqn) != entry_map_.end())
        throw std::invalid_argument(
            "Context::share_outputs(): owner '" + owner_fqn +
            "' already exists in the context");

    // Resolve each member to its enrollable entry (promote_output minus the
    // single-source constraint — see the header contract). Plain Promotions
    // are walked through their sources[0] chain to the leaf Output so the
    // whole chain aliases the owner; nested share_outputs owners enroll
    // directly.
    std::vector<size_t> member_indices;
    std::vector<std::string> resolved_fqns;
    member_indices.reserve(member_fqns.size());
    resolved_fqns.reserve(member_fqns.size());
    sycl::range<3> owner_shape{1, 1, 1};

    for (size_t p = 0; p < member_fqns.size(); ++p) {
        const std::string& m = member_fqns[p];
        Entry* member = find_entry(m);
        if (!member)
            throw std::invalid_argument(
                "Context::share_outputs(): member '" + m + "' not found");

        // Walk plain-promotion chains (promote_output aliases) to the leaf.
        while (member->kind == EntryKind::Promotion && !member->is_shareable) {
            if (member->sources.empty())
                throw std::invalid_argument(
                    "Context::share_outputs(): member '" + m +
                    "' is a sourceless plain Promotion — nothing to alias");
            Entry* next = find_entry(member->sources[0]);
            if (!next)
                throw std::invalid_argument(
                    "Context::share_outputs(): member '" + m +
                    "' resolves through unknown source '" +
                    member->sources[0] + "'");
            member = next;
        }

        const bool leaf_output = member->kind == EntryKind::Output;
        const bool nested_output_share_owner =
            member->kind == EntryKind::Promotion && member->is_shareable;
        if (!leaf_output && !nested_output_share_owner)
            throw std::invalid_argument(
                "Context::share_outputs(): member '" + m +
                "' does not resolve to an Output entry or a share_outputs "
                "owner (resolved to '" + member->name + "')");
        if (member->dtype != DType::f32)
            throw std::invalid_argument(
                "Context::share_outputs(): member '" + m + "' has dtype " +
                dtype_name(member->dtype) +
                " — shared output storage is f32-only in P1");
        if (member->is_target)
            throw std::invalid_argument(
                "Context::share_outputs(): member '" + m + "' (resolved '" +
                member->name + "') is already a target of '" +
                member->sources[0] + "'");
        if (p == 0) {
            owner_shape = member->shape3;
        } else {
            if (shape3_size(member->shape3) != shape3_size(owner_shape))
                throw std::invalid_argument(
                    "Context::share_outputs(): size mismatch between owner '" +
                    owner_fqn + "' and member '" + m + "'");
        }
        member_indices.push_back(entry_map_.at(member->name));
        resolved_fqns.push_back(member->name);
    }

    // Create the storage-owner Promotion entry. Single buffer: NOT a
    // recurrent output at declare time (a back-edge cut at finalize may
    // still mark it one — e.g. the u field behind the register cut —
    // granting the prev/curr slot pair then).
    {
        Entry owner;
        owner.name = owner_fqn;
        owner.shape3 = owner_shape;
        owner.kind = EntryKind::Promotion;
        owner.is_shareable = true;
        owner.is_recurrent_output = false;
        entries_.emplace_back(std::move(owner));
        entry_map_[owner_fqn] = entries_.size() - 1;
    }

    // After emplace_back, entries_ may have reallocated — re-acquire by index.
    size_t owner_idx = entry_map_.at(owner_fqn);
    const size_t count = resolved_fqns.size();

    for (size_t p = 0; p < count; ++p) {
        Entry* owner_ptr = &entries_[owner_idx];
        Entry* member_ptr = &entries_[member_indices[p]];

        owner_ptr->targets.push_back(resolved_fqns[p]);
        member_ptr->sources.push_back(owner_fqn);
        member_ptr->is_target = true;
        member_ptr->is_share_member = true;
        member_ptr->is_design_variable = false;
        member_ptr->share_index = p;
        member_ptr->share_count = count;
    }
}



} // namespace naga::core
