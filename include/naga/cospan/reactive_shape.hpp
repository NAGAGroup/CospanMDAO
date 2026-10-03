#pragma once
// =============================================================================
// naga/cospan/reactive_shape.hpp: the one error for "a shape is needed and is not known" (r01 §4.5)
// =============================================================================
//
// Shapes are reactive: nothing is checked against registration order at setup. A shape is read when something
// needs it (a Component resolving a derived dimension, a tie alias evaluating its rule, a chunked system
// reading the length of a boundary root, a variable being materialized); if it is not known then, THIS error says
// so, in three parts:
//   (a) what needed the shape: a port path, a tie alias path, or a chunked System's path;
//   (b) which dimension (or value) it needed;
//   (c) what it was waiting on: the variable by a port path, and the path of its producer if it has one, saying
//       whether that producer does not currently exist or has not run yet in this forward.

#include <stdexcept>
#include <string>
#include <string_view>

#include "naga/cospan/metadata.hpp"

namespace naga::cospan::detail {

/// The port that writes the variable now (a connected source, else an output port that joined it), or null.
inline const IOEntry* producer_port(const Variable& v) {
    if (v.source_port != nullptr) return v.source_port;
    for (const IOEntry* p : v.ports)
        if (p->kind() == IOKind::output) return p;
    return nullptr;
}
/// The variable has a producer: one that exists, or one that was deleted (its name is kept).
inline bool has_producer(const Variable& v) { return producer_port(v) != nullptr || !v.source_path.empty(); }

/// Part (c): how the variable `on` is fed, and why that has not produced a shape yet.
inline std::string producer_clause(const IOEntry& on) {
    const Variable& v = on.variable();
    if (const IOEntry* p = producer_port(v))
        return "fed by '" + p->full_path() + "', which has not run yet in this forward";
    if (!v.source_path.empty())
        return "fed by '" + v.source_path +
               "', which does not currently exist (its producer was deleted; rearm it first)";
    return "which no producer feeds";
}

/// The error. `needed_by`: part (a); `what`: part (b), e.g. "dimension 1" or "the value of element 0";
/// `waiting_on`: the port whose variable the shape or value was needed from.
inline std::logic_error shape_needed(std::string_view needed_by, const std::string& what, const IOEntry& waiting_on) {
    return std::logic_error("naga::cospan: shape not known yet: '" + std::string{needed_by} + "' needs " + what +
                            " of '" + waiting_on.full_path() + "', " + producer_clause(waiting_on));
}

/// A dimension that a producer has not resolved yet: scalar, derived or unspecified (not fixed), extent 0, and the
/// variable has a producer but no data anywhere (so that producer has not run, or no longer exists).
inline bool dim_pending(const IOEntry& e, int d) {
    const Variable& v = e.variable();
    if (v.any_live()) return false;
    if (!has_producer(v)) return false;
    if (e.dim_is_per(d)) return false;
    return e.extents()[static_cast<std::size_t>(d)] == 0 && e.dim(d).how != DimRecord::How::fixed;
}

}  // namespace naga::cospan::detail
