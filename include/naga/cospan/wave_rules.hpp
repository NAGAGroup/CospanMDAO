#pragma once

// =============================================================================
// naga/cospan/wave_rules.hpp — the metadata rules of the two concurrent owners
// =============================================================================
//
// FusedSystem (parallel mode) and ParallelSystem run their children unordered,
// and FusedSystem (both modes) reads value-driven sizes on the host before the
// one launch. The checks below read only the public IOEntry accessors (root,
// peers, declarer, kind, dim) and are shared so the two owners refuse the same
// wirings with the same words. Each takes the owner's own children (its
// "stages") and the owner's description for the message.
//
//   check_no_stage_connections   an input whose aliasing set holds an output of
//                                ANOTHER stage: the stages run unordered, so
//                                there is nothing to read yet. No override.
//   check_tied_outputs           two or more stages whose outputs share one
//                                root (tie_outputs): unordered writers. The
//                                owner's allow_tied_outputs() skips this one.
//   check_value_driven_sizes     an entry with a value-driven dim (FromInput)
//                                whose driving input has an output of another
//                                stage in its aliasing set: the value is read
//                                on the host before the launch, so it cannot
//                                come from a stage of the same wave.
//
// The aliasing set of an entry is its root and the root's peers (§4). An output
// counts as "declared by a stage" when its declarer() is one of the stages
// (the tie alias of tie_outputs is declared by a System, but its member
// outputs are peers of it and are declared by the stages).

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

#include "naga/cospan/component.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/subsystem.hpp"

namespace naga::cospan::detail {

inline bool is_stage(const Subsystem* s, std::span<Subsystem* const> stages) noexcept {
    for (const Subsystem* c : stages)
        if (c == s) return true;
    return false;
}

/// The first OUTPUT in the aliasing set of `e` (its root, then the root's
/// peers) that is declared by a stage other than `self`; null if there is none.
inline const IOEntry* output_of_other_stage(const IOEntry& e, const Subsystem* self,
                                            std::span<Subsystem* const> stages) {
    const IOEntry* root = e.root();
    const auto     hit  = [&](const IOEntry* x) {
        return x->kind() == IOKind::output && x->declarer() != self && is_stage(x->declarer(), stages);
    };
    if (hit(root)) return root;
    for (const IOEntry* p : root->peers())
        if (hit(p)) return p;
    return nullptr;
}

/// A stage is a Component by construction (the owners' add_subsystem constraint).
inline const ComponentBase* stage_component(const Subsystem* s) {
    const auto* c = dynamic_cast<const ComponentBase*>(s);
    if (c == nullptr)
        throw std::logic_error("naga::cospan: child '" + s->path() + "' is not a Component");
    return c;
}

/// (D140) FusedSystem and ParallelSystem: a child with any ragged port (declared with RaggedInput /
/// RaggedOutput, or with a per-index dimension spec) is a setup error for now.
inline void check_no_ragged_ports(const std::string& owner, std::span<Subsystem* const> stages) {
    for (const Subsystem* s : stages) {
        const ComponentBase* c = stage_component(s);
        const auto scan = [&](const auto& entries) {
            for (const auto& ref : entries) {
                const IOEntry& e = ref.get();
                if (e.is_ragged_port() || e.declared_ragged())
                    throw std::invalid_argument("naga::cospan: " + owner + ": '" + std::string{e.name()} + "' of '" +
                                                s->path() +
                                                "' is a ragged port: ragged ports are not supported in fused/parallel systems yet");
            }
        };
        scan(c->get_inputs());
        scan(c->get_outputs());
    }
}

/// (c) PARALLEL only. `what` names the owner kind in the message, e.g.
/// "parallel-mode wave" or "ParallelSystem".
inline void check_no_stage_connections(const std::string& owner, const char* what,
                                       std::span<Subsystem* const> stages) {
    for (const Subsystem* s : stages) {
        const ComponentBase* c = stage_component(s);
        for (const auto& ref : c->get_inputs()) {
            const IOEntry& e = ref.get();
            if (const IOEntry* o = output_of_other_stage(e, s, stages))
                throw std::invalid_argument(
                    "naga::cospan: " + owner + ": forward: input '" + std::string{e.name()} + "' of '" + s->path() +
                    "' is connected to '" + std::string{o->name()} + "', an output of '" + o->declarer()->path() +
                    "'; the stages of a " + what +
                    " run unordered, so no stage may read another's output (connect them through a sequential "
                    "System)");
        }
    }
}

/// (d) PARALLEL only, skipped when the owner's allow_tied_outputs() was called.
inline void check_tied_outputs(const std::string& owner, const char* what, std::span<Subsystem* const> stages) {
    for (const Subsystem* s : stages) {
        const ComponentBase* c = stage_component(s);
        for (const auto& ref : c->get_outputs()) {
            const IOEntry& e = ref.get();
            if (const IOEntry* o = output_of_other_stage(e, s, stages))
                throw std::invalid_argument(
                    "naga::cospan: " + owner + ": forward: output '" + std::string{e.name()} + "' of '" + s->path() +
                    "' and output '" + std::string{o->name()} + "' of '" + o->declarer()->path() +
                    "' share one root ('" + std::string{e.root()->name()} + "'), so the stages of a " + what +
                    " would write it unordered; call allow_tied_outputs() if the writers are atomic or otherwise "
                    "known not to race");
        }
    }
}

/// (h) Both modes of a FusedSystem, at compile(): every entry of every stage with
/// a value-driven dim (FromInput) is followed to its driving input, and the
/// driving input's aliasing set must not hold an output of another stage.
/// FromInputDim and FromSourceDim (through connections) are always allowed. A tied entry (D139
/// part 2) is checked through its tie alias: a from_output_shape / from_output_values source must
/// not be an output of another stage of the same owner.
inline void check_value_driven_sizes(const std::string& owner, std::span<Subsystem* const> stages) {
    for (const Subsystem* s : stages) {
        const ComponentBase* c = stage_component(s);
        const auto scan = [&](const auto& entries) {
            for (const auto& ref : entries) {
                const IOEntry& e = ref.get();
                for (int d = 0; d < e.rank(); ++d) {
                    const DimRecord& r = e.dim(d);
                    if (r.how == DimRecord::How::from_root) {
                        // A tied entry's shape comes from its tie alias; a from_output_* alias reads an
                        // upstream output on the host, which must not belong to a stage of the same wave.
                        const IOEntry* alias = e.tie_alias();
                        if (alias == nullptr || !alias->alias_shape() || alias->alias_source() == nullptr) continue;
                        if (const IOEntry* o = output_of_other_stage(*alias->alias_source(), s, stages))
                            throw std::invalid_argument(
                                "naga::cospan: " + owner + ": '" + std::string{e.name()} + "' of '" + s->path() +
                                "' takes its shape from tie alias '" + alias->full_path() + "', which reads the output '" +
                                alias->alias_shape()->path + "' (an output of '" + o->declarer()->path() +
                                "' in the same fuse); the shape is read on the host before the wave launches, so it "
                                "cannot come from a stage of the same wave (feed it from outside the fuse)");
                        break;
                    }
                    if (r.how != DimRecord::How::from_input) continue;
                    for (const auto& in_ref : c->get_inputs()) {
                        const IOEntry& driver = in_ref.get();
                        if (driver.name() != r.input) continue;
                        if (const IOEntry* o = output_of_other_stage(driver, s, stages))
                            throw std::invalid_argument(
                                "naga::cospan: " + owner + ": '" + std::string{e.name()} + "' of '" + s->path() +
                                "' takes dimension " + std::to_string(d) + " of its shape from the VALUE of input '" +
                                r.input + "', whose source '" + std::string{o->name()} + "' is an output of '" +
                                o->declarer()->path() +
                                "' in the same fuse; the value is read on the host before the wave launches, so it "
                                "cannot come from a stage of the same wave (feed it from outside the fuse)");
                        break;
                    }
                }
            }
        };
        scan(c->get_inputs());
        scan(c->get_outputs());
    }
}

}  // namespace naga::cospan::detail
