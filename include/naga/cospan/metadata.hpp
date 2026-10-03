#pragma once

// =============================================================================
// naga/cospan/metadata.hpp — the declaration types
// =============================================================================
//
// The declaration types: dimension specs, tie-alias shape specs, marks, initializer
// shapes. No storage, no state. A variable's data and shared metadata live in
// variable.hpp, an instance in instance.hpp, and an initializer in init.hpp
// (docs/design-review-2026-10-03.md, sections 1 and 6).

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
}  // namespace detail

enum class IOKind : uint8_t { input, output };

// -----------------------------------------------------------------------------
// Dimension specification — DimSpec is a CONCEPT, not a type.
// -----------------------------------------------------------------------------

/// UNSPECIFIED (shape authority): this dimension is supplied from outside: by the connection (or slice) an input
/// is connected to, by the tie the entry belongs to, or, for a free input, by its initializer's shape. A SPECIFIED
/// dimension (an integer, FromInputDim, FromInput, PerIndex, FromInputValues) is a constraint on what supplies it.
struct Unspecified {};
inline constexpr Unspecified unspecified{};

/// This dimension = dim `dim` of sibling input `input`'s shape.
struct FromInputDim { std::string_view input; uint8_t dim; };

/// This dimension's extent = the VALUE of element `index` (flat) of sibling
/// input `input`, which SHALL have an integer value dtype.
struct FromInput { std::string_view input; std::size_t index; };

/// RAGGED DIMENSIONS. A dimension d >= 1 of a ragged port may have a different extent for every
/// index of dimension 0: PerIndex gives the extents as a host array at declaration (its length must equal
/// dimension 0's extent), FromInputValues takes them from the VALUES of an integer sibling input of rank 1
/// [N] (column 0) or rank 2 [N, k] (column < k). Negative values are an error. Neither is allowed on dim 0.
struct PerIndex { std::vector<std::size_t> values; };

inline PerIndex per_index(std::vector<std::size_t> values) { return PerIndex{std::move(values)}; }

struct FromInputValues { std::string_view input; std::size_t column = 0; };

template <class T>
concept DimSpec = std::integral<T> || std::same_as<T, Unspecified> ||
                  std::same_as<T, FromInputDim> || std::same_as<T, FromInput> ||
                  std::same_as<T, PerIndex> || std::same_as<T, FromInputValues>;

/// The whole-shape value-driven marker: on a rank-R declaration it expands to
/// FromInput{input,0}..{input,R-1}. Accepted alone by the declaration
/// overloads; expansion against the member's Rank happens in the declaration.
struct FromInputArray { std::string_view input; };

/// Sugar. The two shape helpers return arrays of specs (spread them with
/// std::apply); from_input_array returns the marker.
template <int Rank>
constexpr std::array<Unspecified, Rank> unspecified_shape() {
    return std::array<Unspecified, Rank>{};
}

template <int Rank>
constexpr std::array<FromInputDim, Rank> from_input_shape(std::string_view input) {
    std::array<FromInputDim, Rank> a{};
    for (int d = 0; d < Rank; ++d) a[d] = FromInputDim{input, static_cast<uint8_t>(d)};
    return a;
}

constexpr FromInputArray from_input_array(std::string_view input) { return {input}; }

/// The per-dimension specs of cs::alias::dims (tie aliases take the shape rules of ports). An integer
/// or cs::per_index(...) is also a spec; these name an upstream OUTPUT, in the declaring System's namespace.
namespace alias {
/// The extent is the VALUE of element `index` (flat) of an upstream integer output.
struct OutputValue { std::string path; std::size_t index = 0; };
/// Per-index extents (dims >= 1): the VALUES of an upstream integer output of rank 1 [N] (column 0) or rank 2 [N, k].
struct OutputValues { std::string path; std::size_t column = 0; };
/// Dim `dim` of an upstream output's shape (per-index too, if that dim is per-index).
struct OutputDim { std::string path; std::size_t dim = 0; };

inline OutputValue output_value(std::string_view path, std::size_t index = 0) { return {std::string{path}, index}; }
inline OutputValues output_values(std::string_view path, std::size_t column = 0) { return {std::string{path}, column}; }
inline OutputDim output_dim(std::string_view path, std::size_t d) { return {std::string{path}, d}; }

/// The same three specs reading an INPUT of the declaring System instead (a promoted input or a tie_inputs
/// alias, connected from upstream or free): values and shapes are read from the entry's ROOT on the host.
struct InputValue { std::string path; std::size_t index = 0; };
struct InputValues { std::string path; std::size_t column = 0; };
struct InputDim { std::string path; std::size_t dim = 0; };

inline InputValue input_value(std::string_view path, std::size_t index = 0) { return {std::string{path}, index}; }
inline InputValues input_values(std::string_view path, std::size_t column = 0) { return {std::string{path}, column}; }
inline InputDim input_dim(std::string_view path, std::size_t d) { return {std::string{path}, d}; }
}  // namespace alias

template <class T>
concept AliasDimSpec = std::integral<T> || std::same_as<T, PerIndex> || std::same_as<T, alias::OutputValue> ||
                       std::same_as<T, alias::OutputValues> || std::same_as<T, alias::OutputDim> ||
                       std::same_as<T, alias::InputValue> || std::same_as<T, alias::InputValues> ||
                       std::same_as<T, alias::InputDim>;

/// An upstream instance an alias shape reads (or a tie member), by path and by weak pointer: a deleted
/// child's instance expires, and nothing here keeps it alive (docs/design-review-2026-10-03.md, section 1).
struct AliasSrc {
    std::string                    path;
    std::weak_ptr<detail::Instance> inst;
    explicit operator bool() const noexcept { return !inst.expired(); }
};

/// One dimension of an alias::dims shape, type-erased.
struct AliasDim {
    enum class How : uint8_t { fixed, per_index_host, output_value, output_values, output_dim };
    How                      how    = How::fixed;
    std::size_t              extent = 1;   ///< fixed
    std::vector<std::size_t> values;       ///< per_index_host
    std::string              path;         ///< output_*: the upstream output (input = true: the System's input)
    std::size_t              index  = 0;   ///< output_value: the element; output_values: the column; output_dim: the dim
    bool                     input  = false;  ///< the path names an INPUT entry (input_value / input_values / input_dim)
    AliasSrc                 src;               ///< output_*: resolved by the System at setup
};

template <AliasDimSpec S>
AliasDim make_alias_dim(const S& s) {
    AliasDim r;
    if constexpr (std::integral<S>) {
        if constexpr (std::is_signed_v<S>) {
            if (s < 0) throw std::invalid_argument("naga::cospan: a fixed extent is negative");
        }
        r.how    = AliasDim::How::fixed;
        r.extent = static_cast<std::size_t>(s);
    } else if constexpr (std::same_as<S, PerIndex>) {
        r.how    = AliasDim::How::per_index_host;
        r.values = s.values;
    } else if constexpr (std::same_as<S, alias::OutputValue> || std::same_as<S, alias::InputValue>) {
        r.how   = AliasDim::How::output_value;
        r.path  = s.path;
        r.index = s.index;
        r.input = std::same_as<S, alias::InputValue>;
    } else if constexpr (std::same_as<S, alias::OutputValues> || std::same_as<S, alias::InputValues>) {
        r.how   = AliasDim::How::output_values;
        r.path  = s.path;
        r.index = s.column;
        r.input = std::same_as<S, alias::InputValues>;
    } else {
        r.how   = AliasDim::How::output_dim;
        r.path  = s.path;
        r.index = s.dim;
        r.input = std::same_as<S, alias::InputDim>;
    }
    return r;
}

/// How a tie alias gets its shape; build one with the cs::alias factories.
struct AliasShape {
    enum class Kind : uint8_t { fixed, unspecified, from_output_shape, from_output_values, dims, from_input_shape };
    Kind                       kind = Kind::fixed;
    int                        rank = 0;            ///< fixed / dims: how many extents / specs were given
    std::array<std::size_t, 3> extents{1, 1, 1};    ///< fixed
    std::string                path;                ///< from_output_* / from_input_shape: an entry, in the declaring System's namespace
    std::vector<AliasDim>      dim_specs;           ///< dims: one per dimension
};

namespace detail {
/// tie_inputs' default shape: the shape of whatever supplies the tie (a connected output, else an initializer on it).
inline AliasShape unspecified_alias_shape() {
    AliasShape s;
    s.kind = AliasShape::Kind::unspecified;
    return s;
}
}  // namespace detail

namespace alias {
/// The per-dimension form: one spec per dim, count = the alias's Rank. A spec is an integer (fixed),
/// cs::per_index(values) (dims >= 1), alias::output_value(path, index), alias::output_values(path, column)
/// (dims >= 1) or alias::output_dim(path, d).
template <AliasDimSpec... S>
AliasShape dims(const S&... specs) {
    static_assert(sizeof...(S) >= 1 && sizeof...(S) <= 3, "cospan: rank is 1..3");
    AliasShape s;
    s.kind = AliasShape::Kind::dims;
    s.rank = static_cast<int>(sizeof...(S));
    (s.dim_specs.push_back(make_alias_dim(specs)), ...);
    return s;
}
/// The alias has this shape (its rank is the number of extents).
inline AliasShape fixed(std::size_t e0) {
    AliasShape s;
    s.rank = 1;
    s.extents = {e0, 1, 1};
    return s;
}
inline AliasShape fixed(std::size_t e0, std::size_t e1) {
    AliasShape s;
    s.rank = 2;
    s.extents = {e0, e1, 1};
    return s;
}
inline AliasShape fixed(std::size_t e0, std::size_t e1, std::size_t e2) {
    AliasShape s;
    s.rank = 3;
    s.extents = {e0, e1, e2};
    return s;
}
/// The shape of an upstream output (it must have the alias's rank).
inline AliasShape from_output_shape(std::string_view path) {
    AliasShape s;
    s.kind = AliasShape::Kind::from_output_shape;
    s.path = std::string{path};
    return s;
}
/// The shape (ragged-aware) of an INPUT of the declaring System: a promoted input or a tie_inputs alias, read
/// from its root (it must have the alias's rank).
inline AliasShape from_input_shape(std::string_view path) {
    AliasShape s;
    s.kind = AliasShape::Kind::from_input_shape;
    s.path = std::string{path};
    return s;
}
/// The extents are the VALUES of elements 0..Rank-1 of an upstream output of integer dtype.
inline AliasShape from_output_values(std::string_view path) {
    AliasShape s;
    s.kind = AliasShape::Kind::from_output_values;
    s.path = std::string{path};
    return s;
}
}  // namespace alias

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

/// The shape an initializer carries: extents 1-padded, and the rank they were given for. Each dim >= 1 is a
/// scalar (extents[d]) or a per-index vector (per[d], per_index[d]: one extent per index of dim 0, extents[d] = 0);
/// dim 0 is always a scalar. `flat`: the shape of bare data (from_host(vec)): rank 1, it only describes a rank-1 root.
struct InitShape {
    std::array<std::size_t, 3>              extents{1, 1, 1};
    int                                     rank = 0;
    std::array<bool, 3>                     per{false, false, false};
    std::array<std::vector<std::size_t>, 3> per_index{};
    bool                                    flat = false;

    bool ragged() const noexcept { return per[1] || per[2]; }
    /// The number of elements: the product of the extents, or the sum of the block sizes of a ragged shape.
    std::size_t elements() const {
        if (!ragged()) return extents[0] * extents[1] * extents[2];
        std::size_t total = 0;
        for (std::size_t i = 0; i < extents[0]; ++i) {
            std::size_t block = 1;
            for (std::size_t d = 1; d < 3; ++d) block *= per[d] ? per_index[d][i] : extents[d];
            total += block;
        }
        return total;
    }
};

namespace detail {

/// Per-dimension storage form of a declared DimSpec.
struct DimRecord {
    /// unspecified: supplied by the root (a connection, a slice, a tie) or, on a free input root, by its
    /// initializer's shape.
    /// per_index_host / per_index_from_input: dimension d >= 1 has one extent per index of dim 0.
    enum class How : uint8_t { fixed, unspecified, from_input_dim, from_input, per_index_host,
                               per_index_from_input };
    How         how = How::fixed;
    std::size_t extent = 0;  ///< fixed: declared; derived: last resolved (0 = unknown)
    std::string input;       ///< from_input_*: sibling input name (owned)
    uint8_t     dim = 0;     ///< from_input_dim
    std::size_t value_index = 0;  ///< from_input; per_index_from_input: the column
    /// The resolved per-index extents of a dimension d >= 1; EMPTY means the dimension is scalar
    /// (its extent is `extent`). Filled by per_index_host at declaration, by per_index_from_input when the
    /// owning Component resolves it, and copied from the source dimension by from_source_dim /
    /// from_input_dim / from_root when that dimension is per-index.
    std::vector<std::size_t> per_index;
    /// The dimension is per-index (even when `per_index` is empty because N is 0): true from
    /// declaration for per_index_host / per_index_from_input, copied with the array by the derived hows.
    bool per = false;
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
    } else if constexpr (std::same_as<S, Unspecified>) {
        r.how = DimRecord::How::unspecified;
    } else if constexpr (std::same_as<S, FromInputDim>) {
        r.how   = DimRecord::How::from_input_dim;
        r.input = std::string{s.input};
        r.dim   = s.dim;
    } else if constexpr (std::same_as<S, PerIndex>) {
        r.how       = DimRecord::How::per_index_host;
        r.per_index = s.values;
        r.per       = true;
    } else if constexpr (std::same_as<S, FromInputValues>) {
        r.how         = DimRecord::How::per_index_from_input;
        r.input       = std::string{s.input};
        r.value_index = s.column;
        r.per         = true;
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

}  // namespace detail
}  // namespace naga::cospan
