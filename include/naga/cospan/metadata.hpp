#pragma once

// =============================================================================
// naga/cospan/metadata.hpp — the declaration types
// =============================================================================
//
// The declaration types: dimension specs, ragged declarations, tie shape specs, marks, initializer shapes. No storage,
// no state. A variable's data and shared metadata live in variable.hpp, an instance in instance.hpp, and an
// initializer in init.hpp (docs/design-review-2026-10-03.md, sections 1 and 10).
//
// THE SHAPE VOCABULARY (section 10). A declaration says, per dimension, only SPECIFIED (an integer) or UNSPECIFIED
// (cs::unspecified). There are no derived specs. A ragged declaration is rank 2 only: a dense leading dimension plus one
// per-index dimension, given as nothing, a leading integer, cs::counts(v) or cs::offsets(v). A variable's shape
// DEFINITION (detail::DimRecord, one per dimension) is fixed once setup ends; a dimension is specified, unspecified, or
// owned by exactly one shape hook (shape_hook.hpp).

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace naga::cospan {

class System;
class ComponentBase;
class Subsystem;

namespace detail {
class Instance;  // instance.hpp
class VarMeta;   // variable.hpp
}  // namespace detail

enum class IOKind : uint8_t { input, output };

// -----------------------------------------------------------------------------
// Dimension specification — DimSpec is a CONCEPT, not a type.
// -----------------------------------------------------------------------------

/// UNSPECIFIED: this dimension places no constraint and is supplied from outside: by the connection an input is connected
/// to, by the tie the entry belongs to, by a shape hook, by its producer's prepare_forward, or, for a free input, by its
/// initializer's shape. A SPECIFIED dimension (an integer) is a contract: it forces that extent on anything joined to it.
struct Unspecified {};
inline constexpr Unspecified unspecified{};

template <class T>
concept DimSpec = std::integral<T> || std::same_as<T, Unspecified>;

/// Sugar: an array of Rank unspecified dimensions (for the array forms of the declaration calls).
template <int Rank>
constexpr std::array<Unspecified, Rank> unspecified_shape() {
    return std::array<Unspecified, Rank>{};
}

// -----------------------------------------------------------------------------
// Ragged declarations (rank 2: a dense leading dimension N and one per-index dimension)
// -----------------------------------------------------------------------------

/// The per-index extents as a host array, one count per index: the array's length is the leading dimension N.
struct Counts { std::vector<std::size_t> v; };
inline Counts counts(std::vector<std::size_t> v) { return Counts{std::move(v)}; }

/// The same as N + 1 block offsets (offsets[0] == 0, non-decreasing); they are converted to counts internally. The
/// leading dimension is v.size() - 1.
struct Offsets { std::vector<std::size_t> v; };
inline Offsets offsets(std::vector<std::size_t> v) {
    if (v.empty()) throw std::invalid_argument("naga::cospan: cs::offsets needs at least one entry (N + 1 offsets)");
    if (v.front() != 0) throw std::invalid_argument("naga::cospan: cs::offsets must start at 0");
    for (std::size_t i = 1; i < v.size(); ++i)
        if (v[i] < v[i - 1])
            throw std::invalid_argument("naga::cospan: cs::offsets must be non-decreasing (entry " + std::to_string(i) +
                                        " is below entry " + std::to_string(i - 1) + ")");
    return Offsets{std::move(v)};
}

namespace detail {

/// Per-dimension storage form of a shape DEFINITION. `fixed`: specified (dimension d >= 1 of a ragged variable is fixed
/// with host counts held on the variable, VarMeta::def_counts, and `extent` unused). `hook`: owned by the shape hook with
/// id `hook_owner`; it must be unspecified when the hook is declared.
struct DimRecord {
    enum class How : uint8_t { fixed, unspecified, hook };
    How           how = How::fixed;
    std::size_t   extent = 0;      ///< fixed: the declared extent
    std::uint32_t hook_owner = 0;  ///< hook: the owning hook's id (never 0 for a hook)
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
    } else {
        r.how = DimRecord::How::unspecified;
    }
    return r;
}

}  // namespace detail

/// The shape a tie alias declares (tie_inputs, tie_outputs): per-dimension specified or unspecified, like a port.
/// `dims` is empty for cs::shape() (every dimension unspecified). For a ragged tie: one integer (the leading dimension),
/// or counts / offsets (`ragged_counts` is set and `dims` holds the leading dimension, counts.size()). Build one with
/// cs::shape(...).
struct ShapeSpec {
    std::vector<detail::DimRecord>                 dims;
    std::optional<std::vector<std::size_t>>        ragged_counts;
};

namespace detail {
template <DimSpec S>
void shape_add(ShapeSpec& out, const S& s) {
    if (out.ragged_counts) throw std::invalid_argument("naga::cospan: cs::shape: counts / offsets must be the only argument");
    out.dims.push_back(make_dim_record(s));
}
inline void shape_add(ShapeSpec& out, const Counts& c) {
    if (!out.dims.empty() || out.ragged_counts)
        throw std::invalid_argument("naga::cospan: cs::shape: counts / offsets must be the only argument");
    out.dims.push_back(make_dim_record(c.v.size()));
    out.ragged_counts = c.v;
}
inline void shape_add(ShapeSpec& out, const Offsets& o) {
    std::vector<std::size_t> c(o.v.size() - 1);
    for (std::size_t i = 0; i + 1 < o.v.size(); ++i) c[i] = o.v[i + 1] - o.v[i];
    shape_add(out, Counts{std::move(c)});
}
}  // namespace detail

/// cs::shape(): every dimension unspecified. cs::shape(a, b, ...): one DimSpec per dimension (dense). For a ragged tie:
/// cs::shape(), cs::shape(n), cs::shape(cs::counts(v)) or cs::shape(cs::offsets(v)).
template <class... S>
ShapeSpec shape(S... specs) {
    static_assert(sizeof...(S) <= 3, "cospan: rank is 1..3");
    ShapeSpec out;
    (detail::shape_add(out, specs), ...);
    return out;
}

// -----------------------------------------------------------------------------
// Marks
// -----------------------------------------------------------------------------

/// Per-element or scalar bounds/scaling; host-side vectors sized to the marked
/// region's flat size (scalar forms broadcast).
struct Bounds { std::optional<std::vector<double>> lower, upper; };
struct Scaling { std::vector<double> ref{1.0}, ref0{0.0}; };

/// One mark on one variable (Model::MarkRecord aliases this type). Finalize holds the marked
/// instance by weak pointer (docs/design-review-2026-10-03.md, section 2); slices are not ported.
struct MarkRecord {
    Bounds                             bounds;
    std::optional<std::vector<double>> equals;  // constraints only
    Scaling                            scaling;
};

/// The shape an initializer carries: extents 1-padded, and the rank they were given for. A RAGGED shape is rank 2: dimension
/// 1 is per-index (`per`, `per_index`: one extent per index of dim 0, extents[1] = 0); dimension 0 is always a scalar.
/// `flat`: the shape of bare data (from_host(vec)): rank 1, it only describes a rank-1 root.
struct InitShape {
    std::array<std::size_t, 3> extents{1, 1, 1};
    int                        rank = 0;
    bool                       per  = false;
    std::vector<std::size_t>   per_index{};
    bool                       flat = false;

    bool ragged() const noexcept { return per; }
    /// The number of elements: the product of the extents, or the sum of the counts of a ragged shape.
    std::size_t elements() const {
        if (!per) return extents[0] * extents[1] * extents[2];
        std::size_t total = 0;
        for (const std::size_t c : per_index) total += c;
        return total;
    }
};

}  // namespace naga::cospan
