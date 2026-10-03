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
#include <functional>
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

#include "naga/cospan/diagnostics.hpp"
#include "naga/cospan/dtype.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/wrappers.hpp"

namespace naga::cospan {

namespace sk = ::naga::slabkit;

class System;
class ChunkedSystem;
class ComponentBase;
class Model;
class Subsystem;

namespace detail {
class IOEntry;
class AliasEvaluator;  // alias_eval.hpp
struct EntryAccess;
}  // namespace detail

enum class IOKind : uint8_t { input, output };

/// Where one copy of a root's storage lives: a device and a USM kind. A host
/// component's I/O is (its device, host); its producer's is (the same device,
/// device). Two placements on one device are two storages.
struct Placement {
    sycl::device device;
    sk::kind     kind = sk::kind::device;
    friend bool operator==(const Placement& a, const Placement& b) {
        return a.device == b.device && a.kind == b.kind;
    }
};

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

namespace detail {
class Variable;
}

/// An upstream entry an alias shape reads (or a tie member). It holds the entry's VARIABLE, never a raw
/// port, so a child's deletion cannot dangle it. `port` is the declared port, kept ONLY for the setup-time
/// ordering checks (declarer, registration order); alias_sources() hands it out only while it is still a live
/// port of the variable. `view_extents`: an input source's view, captured at declaration.
struct AliasSrc {
    std::shared_ptr<detail::Variable>        var;
    const detail::IOEntry*                   port = nullptr;
    std::optional<std::array<std::size_t, 3>> view_extents;
    explicit operator bool() const noexcept { return var != nullptr; }
    detail::IOEntry*       shadow() const;     // the variable's shadow (its shape and data)
    const detail::IOEntry* live_port() const;  // `port` while registered, else the shadow
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

/// The initial value of a free input root (r01 §4.4): type-erased, made
/// by the cs::init factories (init.hpp). apply() runs on a root that has just
/// been allocated from nothing; the caller refreshes the other placements.
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

class InitSpec {
public:
    using Fn = std::function<void(detail::IOEntry& root, sk::queue& q)>;

    InitSpec() = default;
    explicit InitSpec(Fn f, std::optional<InitShape> shape = std::nullopt) : fn_{std::move(f)}, shape_{shape} {}
    /// The shape this initializer supplies to a free input's unspecified dimensions, if it has one.
    const std::optional<InitShape>& shape() const noexcept { return shape_; }

    void apply(detail::IOEntry& root, sk::queue& q) const {
        if (fn_) fn_(root, q);
    }
    explicit operator bool() const noexcept { return static_cast<bool>(fn_); }

private:
    Fn                       fn_;
    std::optional<InitShape> shape_;
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
    /// per_index_host: the declared full-length array. Inside a ChunkedSystem `per_index` is the chunk's slice of it
    /// ([0, dim0)); outside, `per_index` is this array itself.
    std::vector<std::size_t> host_full;
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
        r.host_full = s.values;
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

// The Variable holds types defined above and needs IOEntry only by pointer.
#include "naga/cospan/variable.hpp"

namespace naga::cospan {
namespace detail {

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

    virtual ~IOEntry() {
        if (var_) var_->unregister_port(this);
    }
    IOEntry(const IOEntry&)            = delete;
    IOEntry& operator=(const IOEntry&) = delete;

    // ---- identity & classification ---------------------------------------
    std::string_view name() const noexcept { return name_; }
    /// The entry's dotted path from the root System: its declarer's path plus its name, or the name alone at the root.
    std::string full_path() const {
        if (home_ != nullptr) {  // a shadow answers with a port's path: the authority's, else the one it had, else any
            if (home_->authority != nullptr) return home_->authority->full_path();
            if (!home_->authority_path.empty()) return home_->authority_path;
            if (!home_->ports.empty()) return home_->ports.front()->full_path();
        }
        return (declarer_ == nullptr || declarer_->path().empty())
                   ? std::string{name()}
                   : declarer_->path() + "." + std::string{name()};
    }
    IOKind           kind() const noexcept { return io_kind_; }
    DType            dtype() const noexcept { return dtype_; }  ///< atomic-ness preserved
    int              rank() const noexcept { return rank_; }    ///< AUTHORITATIVE (§3.1)

    // ---- shape -------------------------------------------------------------
    /// A promoted alias mirrors its member: its shape is read live from it.
    sycl::range<3> extents() const {
        const auto& e = shape_ref();
        return sycl::range<3>{e[0], e[1], e[2]};
    }
    /// Element step in the OWNING allocation (a sliced peer reports its root's).
    sycl::range<3> stride() const {
        const Geometry g = geometry();
        return sycl::range<3>{g.stride[0], g.stride[1], g.stride[2]};
    }
    /// A ragged entry: the TOTAL number of elements (block_offsets().back()). A promoted alias
    /// mirrors its member.
    std::size_t flat_size() const noexcept {
        if (mirror_of_ != nullptr) return mirror_of_->flat_size();
        if (ragged_) return static_cast<std::size_t>(offsets_.back());
        return extents_[0] * extents_[1] * extents_[2];
    }

    // ---- ragged dimensions ------------------------------------------
    /// Any dimension d >= 1 is resolved per-index. extents() reports N on dim 0 and 0 on every per-index
    /// dim; geometry() throws (a ragged entry has no rectangular geometry).
    bool is_ragged() const noexcept { return mirror_of_ != nullptr ? mirror_of_->is_ragged() : ragged_; }
    /// Declared with RaggedInput / RaggedOutput (whatever it resolves to).
    bool is_ragged_port() const noexcept { return ragged_port_; }
    /// Declared with a per-index dimension spec (statically known raggedness).
    bool declared_ragged() const noexcept {
        for (const DimRecord& r : dims_)
            if (r.per) return true;
        return false;
    }
    /// The extent of dimension d >= 1 for every index of dimension 0 (host copy): the per-index array, or,
    /// for a scalar dimension, its constant. Dimension 0 throws.
    std::vector<std::size_t> per_index_extents(int d) const {
        if (mirror_of_ != nullptr) return mirror_of_->per_index_extents(d);
        if (d < 1 || d >= rank_) throw std::out_of_range("naga::cospan: per_index_extents is for dimensions 1..rank-1");
        const DimRecord& r = dims_[static_cast<std::size_t>(d)];
        if (r.per) return r.per_index;
        return std::vector<std::size_t>(extents_[0], r.extent);
    }
    /// N + 1 entries: the exclusive scan of the block sizes (block i = the product of its extents over
    /// dims >= 1). The last entry is the total. Dense entries get the constant-block offsets.
    std::vector<std::uint64_t> block_offsets() const {
        if (mirror_of_ != nullptr) return mirror_of_->block_offsets();
        if (ragged_) return offsets_;
        std::uint64_t block = 1;
        for (int d = 1; d < rank_; ++d) block *= extents_[static_cast<std::size_t>(d)];
        std::vector<std::uint64_t> out(extents_[0] + 1, 0);
        for (std::size_t i = 0; i < extents_[0]; ++i) out[i + 1] = out[i] + block;
        return out;
    }
    /// The block offset of index i (0 <= i <= N): offsets()[i] of a ragged entry, i * the block size of a
    /// dense one. No copy of the whole array.
    std::uint64_t block_offset_at(std::size_t i) const {
        if (mirror_of_ != nullptr) return mirror_of_->block_offset_at(i);
        if (ragged_) return offsets_.at(i);
        std::uint64_t block = 1;
        for (int d = 1; d < rank_; ++d) block *= extents_[static_cast<std::size_t>(d)];
        return i * block;
    }
    /// The runtime leading range [a, b) this connected input sees of its root; none = whole.
    std::optional<std::pair<std::size_t, std::size_t>> leading_range() const { return lead_; }
    /// A root a ChunkedSystem made at its own level.
    bool is_rerooted() const noexcept { return chunk_rerooted_; }
    /// The raw per-index array of dimension d (empty: scalar). Internal.
    const std::vector<std::size_t>& per_index_raw(int d) const { return dims_[static_cast<std::size_t>(d)].per_index; }
    const DimRecord& dim(int d) const {
        if (d < 0 || d >= rank_) throw std::out_of_range("naga::cospan: dim index >= rank");
        return dims_[static_cast<std::size_t>(d)];
    }
    /// How this entry's dimension d was declared, for an error message.
    std::string declared_rule(int d) const {
        if (is_root() && home_->authority != nullptr) return home_->authority->declared_rule(d);  // the shadow's dims are inert
        const DimRecord& r = dims_[static_cast<std::size_t>(d)];
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
    bool is_dynamic() const noexcept {
        for (const DimRecord& r : dims_)
            if (r.how != DimRecord::How::fixed) return true;
        return false;
    }
    /// Some dimension is unspecified (supplied from outside).
    bool has_unspecified() const noexcept {
        for (const DimRecord& r : dims_)
            if (r.how == DimRecord::How::unspecified) return true;
        return false;
    }
    /// The tie alias this entry was tied into (members, and their promotions); null if untied.
    const IOEntry* tie_alias() const noexcept { return tie_alias_; }
    /// A tie alias: how its shape is made, and the upstream output a from_output_* shape reads.
    /// On the shadow they answer for its Variable's authority port (the tie alias).
    const std::optional<AliasShape>& alias_shape() const noexcept {
        static const std::optional<AliasShape> none;
        if (is_root()) return home_->authority != nullptr ? home_->authority->alias_shape_ : none;
        return alias_shape_;
    }
    const IOEntry* alias_source() const noexcept {
        return alias_shape().has_value() ? shape_holder()->alias_src_.live_port() : nullptr;
    }
    /// The port that holds the tie's rule: itself, or (on the shadow) the Variable's authority.
    const IOEntry* shape_holder() const noexcept {
        return is_root() && home_->authority != nullptr ? home_->authority : this;
    }
    /// Every upstream output a tie alias's shape reads (path, entry): one for from_output_*, one per
    /// output-reading dim of an alias::dims shape.
    std::vector<std::pair<std::string_view, const IOEntry*>> alias_sources() const {
        std::vector<std::pair<std::string_view, const IOEntry*>> out;
        const IOEntry* h = shape_holder();
        if (!h->alias_shape_) return out;
        if (h->alias_src_) out.emplace_back(h->alias_shape_->path, h->alias_src_.live_port());
        for (const AliasDim& d : h->alias_shape_->dim_specs)
            if (d.src) out.emplace_back(d.path, d.src.live_port());
        return out;
    }
    /// Dimension d >= 1 is per-index and resolved (a promoted alias asks its member).
    bool dim_is_per(int d) const noexcept {
        if (mirror_of_ != nullptr) return mirror_of_->dim_is_per(d);
        return ragged_ && d >= 1 && d < rank_ && dims_[static_cast<std::size_t>(d)].per;
    }

    /// The wrapper geometry of this entry: whole-root contiguous, or, for a
    /// peer with a view, the view's extents over the root's strides.
    Geometry geometry() const {
        if (mirror_of_ != nullptr) return mirror_of_->geometry();  // a promotion shows its member's geometry, live
        if (ragged_)
            throw std::logic_error("naga::cospan: '" + name_ + "' is ragged: ragged entries have no rectangular geometry");
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
    /// The set's ROOT is its Variable's SHADOW (internal plumbing); every declared entry is a port.
    IOEntry*       root() noexcept { return root_; }
    const IOEntry* root() const noexcept { return root_; }
    bool           is_root() const noexcept { return root_ == this; }
    bool           owns_data() const noexcept { return is_root(); }
    /// The port the shadow's shape is adopted from (the entry that was the set's root before the split).
    bool is_authority() const noexcept { return var_ && var_->authority == this; }
    /// Computes its shape locally, as a root did: the shadow, or the authority port.
    bool local_shape() const noexcept { return is_root() || is_authority(); }
    /// The shared Variable: a port holds it; the shadow points at its home.
    Variable& variable() noexcept { return home_ != nullptr ? *home_ : *var_; }
    const Variable& variable() const noexcept { return home_ != nullptr ? *home_ : *var_; }
    /// Root (shadow) only; empty otherwise: the Variable's live ports.
    std::span<IOEntry* const> peers() const noexcept {
        return is_root() ? std::span<IOEntry* const>{home_->ports} : std::span<IOEntry* const>{};
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
    // ---- registration metadata (ROOTS; recorded by System::resolve, r01 §4.4) ----
    /// The winning initial value of a free input root (highest level; none = zeros).
    const std::optional<InitSpec>& init_spec() const noexcept { return root_->init_spec_; }

    /// Roots: the memory substrates allocation draws on (bound at prepare_exec).
    void bind_substrates(Substrates& s) noexcept {
        substrates_ = &s;
        if (root_ != nullptr) root_->substrates_ = &s;  // the shadow materializes: it needs them
    }
    /// PRESENT: the root's storage state. True once its owner has run the allocation path
    /// (even for a size-0 root, whose storage is empty) until a transient root is released. Host
    /// views of a root that is not present throw.
    bool present() const noexcept { return variable().any_live(); }
    /// Whether the ROOT of this entry's set is present.
    bool has_storage() const noexcept { return variable().any_live(); }
    /// An initial value waiting to be applied by the owner: set when a free input root is
    /// allocated from nothing, and by Model::init_input. It stays set while the root has no elements.
    bool init_pending() const noexcept { return root_->init_pending_; }
    void clear_init_pending() noexcept { root_->init_pending_ = false; }
    /// Model::init_input: the model's initial value replaces any declared one, wins over
    /// them, cannot be overwritten by resolve(), and is applied at the owner's next run.
    void set_model_init(InitSpec spec) {
        root_->init_spec_    = std::move(spec);
        root_->model_init_   = true;
        root_->init_pending_ = true;
    }

    /// this root's capacity is RESERVED by its ChunkedSystem for the chunk loop (capacity role only).
    bool held() const noexcept { return root_->held_; }
    /// The elements the root's primary storage can hold without reallocating (0 when not present).
    std::size_t capacity_elems() const noexcept { return root_->capacity_elems_impl(); }
    /// How many times the root's storage was allocated or grew past its capacity (a test hook).
    std::size_t allocation_count() const noexcept { return root_->allocations_; }
    /// The ROOT's write epoch: a counter that advances whenever the root's data may have changed (a
    /// component that writes it has been submitted, an init was applied, an allocation, a
    /// release or a size change). It is what a host-copy cache compares against; it never decreases.
    /// Not bumped by sync_other_placements (same logical content) nor by ragged-metadata uploads (shape,
    /// not data). Single-threaded submission is assumed, as everywhere in CoSpan.
    std::uint64_t write_epoch() const noexcept { return root_->write_epoch_; }
    /// Advance the root's write epoch (always the canonical root, never a peer, view or alias member).
    void bump_write() noexcept { ++root_->write_epoch_; }
    /// The ROOT's shape version: advances whenever the root's extents change or its ragged description changes
    /// (and when apply_shape sets a different state). A topology cache compares it; it never decreases.
    std::uint64_t shape_version() const noexcept { return root_->shape_version_; }

    /// The host description of an entry's shape: exactly the fields the shape rules and update_shape write.
    /// A ChunkedSystem captures one per inside entry after its full-range pass and replays slices of it per chunk.
    struct ShapeState {
        struct Dim {
            std::size_t              extent = 0;
            bool                     per    = false;
            std::vector<std::size_t> per_index;
            bool operator==(const Dim&) const = default;
        };
        std::array<std::size_t, 3>              extents{1, 1, 1};
        bool                                    ragged = false;
        std::vector<std::uint64_t>              offsets;
        std::array<std::vector<std::size_t>, 3> applied_per;
        std::array<Dim, 3>                      dims;
        bool operator==(const ShapeState&) const = default;
    };
    ShapeState capture_shape() const {
        ShapeState s;
        s.extents     = extents_;
        s.ragged      = ragged_;
        s.offsets     = offsets_;
        s.applied_per = applied_per_;
        for (std::size_t d = 0; d < dims_.size() && d < 3; ++d)
            s.dims[d] = ShapeState::Dim{dims_[d].extent, dims_[d].per, dims_[d].per_index};
        return s;
    }
    /// Set the shape fields directly: no rule is evaluated, nothing is validated, no storage is resized and
    /// ragged_version_ is NOT touched (the device index arrays are not re-uploaded). shape_version_ advances
    /// (on a root) only when the state differs.
    void apply_shape(const ShapeState& s) {
        const bool differs = extents_ != s.extents || ragged_ != s.ragged || offsets_ != s.offsets || applied_per_ != s.applied_per ||
                             [&] {
                                 for (std::size_t d = 0; d < dims_.size() && d < 3; ++d)
                                     if (dims_[d].extent != s.dims[d].extent || dims_[d].per != s.dims[d].per ||
                                         dims_[d].per_index != s.dims[d].per_index)
                                         return true;
                                 return false;
                             }();
        if (!differs) return;
        extents_     = s.extents;
        ragged_      = s.ragged;
        offsets_     = s.offsets;
        applied_per_ = s.applied_per;
        for (std::size_t d = 0; d < dims_.size() && d < 3; ++d) {
            dims_[d].extent    = s.dims[d].extent;
            dims_[d].per       = s.dims[d].per;
            dims_[d].per_index = s.dims[d].per_index;
        }
        if (is_root()) ++shape_version_;
    }
    /// Test probes: the data / offsets pointer the entry's wrapper was last pointed at.
    /// ChunkedSystem replay, internal roots: the first index of the current chunk (see meta_shift_).
    std::size_t          meta_shift() const noexcept { return meta_shift_; }
    const void*          probe_data() const noexcept { return probe_data_; }
    const std::uint64_t* probe_offsets() const noexcept { return probe_offs_; }
    /// the path of the ChunkedSystem this root is internal to; empty otherwise.
    std::string_view chunk_internal_by() const noexcept { return root_->chunk_internal_by_; }
    /// The Subsystem that registered this entry (a Component for I/O, a System for aliases).
    const Subsystem* declarer() const noexcept { return declarer_; }

    /// This entry's own placement: its subsystem's device and storage kind.
    Placement placement() const { return Placement{device(), storage_kind_}; }

    /// THE ROOT HOLDS ALL STORAGE, PER PLACEMENT. \throws for a peer, or for a
    /// placement the root has no allocation on. May be null for an empty extent.
    void* storage_on(const Placement& p) const {
        if (!is_root())
            throw std::logic_error("naga::cospan: storage_on is for roots; '" + name_ +
                                   "' is a peer of '" + root_->name_ + "'");
        return raw_storage_on(p);
    }
    /// Roots only; primary (the declarer's own placement) first.
    std::span<const Placement> storage_placements() const noexcept { return placements_; }

    /// Register the events of a launch on the storage this entry's PORT holds: the root's usm_vector
    /// at this entry's own placement (and its ragged index arrays), through SlabKit's
    /// usm_vector::register_events. Later memory operations on that storage are ordered after the
    /// events, and a free waits on them (r01 §4.4). Any entry; a no-op when the root has no storage
    /// there.
    void register_launch_events(std::span<const sycl::event> events) {
        if (events.empty()) return;
        root_->register_events_on(placement(), events);
    }
    /// A test hook: how many registered events the storage this entry's port holds is keeping
    /// (0 when the root has no storage at this entry's placement).
    std::size_t registered_event_count() const { return root_->registered_events_on(placement()); }

    /// Arena route (r01 §4.4): a root, fully static, in device-kind storage (arenas are device-kind), that every
    /// PORT holding it holds eternally (the holder and every subsystem above it run every forward, never
    /// deleted), and that is not internal to a ChunkedSystem (those are sized per chunk). Marks keep nothing alive.
    bool arena_eligible() const noexcept {
        if (!is_root() || is_dynamic() || storage_kind_ != sk::kind::device || !chunk_internal_by_.empty()) return false;
        for (const IOEntry* p : home_->ports)
            if (p->declarer_ != nullptr && !p->declarer_->runs_eternally()) return false;
        return true;
    }
    /// The bytes ONE device's arena must hold for this root; 0 when the root
    /// takes another route. get_exec_info counts this once per device.
    std::size_t static_bytes() const noexcept {
        if (!arena_eligible()) return 0;
        arena_planned_ = true;  // the arena is sized once, at setup, with these bytes: only a planned variable may use it
        return flat_size() * dtype_size(dtype_);
    }
    mutable bool arena_planned_ = false;

    // ---- capabilities --------------------------------------------------------

    /// Automatic shape update: applies the DimRecords. An unspecified dim is resolved here from its supplier (§3.6); FromInputDim / FromInput extents are resolved by the entry's Component first (set_resolved_extent) and applied here.
    virtual void update_shape() = 0;

    /// Resize. Growing past capacity reallocates through the root's allocator
    /// route; shrinking keeps capacity (usm_vector's contract).
    virtual void reshape(sycl::range<3> new_extents) = 0;

    /// INPUT PEERS ONLY, against root(): flat-size equality (the view's size
    /// for a slice, which must also lie inside the root). Throws.
    virtual void validate_connection() const = 0;

    /// Shape authority: the dimensions this entry SPECIFIES are constraints on what supplies its root's shape
    /// (the connection or slice, the tie, the initializer). Throws std::invalid_argument naming both sides. Non-roots
    /// only. `fixed_only`: setup time, only fixed dimensions against a fixed root.
    virtual void check_constraints(bool fixed_only) const = 0;

    /// ROOTS ONLY (no-op on peers): one usm_vector per placement in
    /// storage_placements(), each from that placement's device substrate
    /// (host-kind placements take the OS host route on the device).
    virtual bool allocate_missing(sk::queue& q, Substrates& substrates, std::vector<std::shared_ptr<void>>& fresh) = 0;

    /// Write {data pointer, shape, stride} into the registered member wrapper.
    virtual void point_wrapper() = 0;

    /// ROOTS ONLY: bring every placement's device copies of the ragged index arrays (the offsets and
    /// one array per per-index dimension) up to date with the host description; drops them when the root
    /// is not ragged. A no-op when nothing changed or the root is not present.
    virtual void upload_ragged(sk::queue& q) = 0;

    /// ROOTS ONLY: the device copy of the offsets (which == 0) or of the per-index array of
    /// dimension `which` (1..2) on placement `p`; null when there is none.
    virtual const std::uint64_t* ragged_ptr_on(const Placement& p, int which) const = 0;

    /// The root's producer copies primary -> every other placement's storage
    /// (a routed copy; the ordinary copy rule).
    virtual void sync_other_placements(sk::queue& q) = 0;


    /// ROOTS ONLY (no-op on peers): drop the storage of a transient root (§4.4). The CALLER
    /// waits the stream first (synchronous frees): a releasing Component waits once
    /// for its whole batch.
    virtual std::size_t capacity_elems_impl() const noexcept = 0;

protected:
    /// `shadow_home` non-null makes this entry the SHADOW of that Variable; otherwise it is a port and
    /// gets a fresh Variable of its own (its typed factory then makes the shadow).
    IOEntry(std::string name, IOKind k, DType dt, int rank, std::vector<DimRecord> dims,
            bool can_source, bool can_target, Variable* shadow_home = nullptr)
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
            if (d == 0 && (r.how == DimRecord::How::per_index_host || r.how == DimRecord::How::per_index_from_input))
                throw std::invalid_argument("naga::cospan: '" + name_ +
                                            "': a per-index extent (PerIndex / FromInputValues) is not allowed on dimension 0");
            if (r.how == DimRecord::How::from_input_dim && r.dim > 2)
                throw std::invalid_argument("naga::cospan: '" + name_ +
                                            "': a dimension index is out of range");
            extents_[static_cast<std::size_t>(d)] = r.extent;  // derived dims start at 0
        }
        if (shadow_home != nullptr) {
            home_ = shadow_home;
            root_ = this;
        } else {
            var_ = std::make_shared<Variable>();
            var_->register_port(this);
            var_->authority = this;
        }
    }

    /// A port's factory step: make this Variable's SHADOW (a typed entry with this port's shape state, no
    /// declarer and no wrapper) and root the port at it. Defined after make_alias_entry.
    void install_shadow();

    virtual void* raw_storage_on(const Placement& p) const = 0;

    /// ROOTS ONLY: usm_vector::register_events on the storage at `p` (and its ragged index arrays).
    virtual void register_events_on(const Placement& p, std::span<const sycl::event> events) = 0;
    /// ROOTS ONLY: the number of events the storage at `p` is keeping registered.
    virtual std::size_t registered_events_on(const Placement& p) const = 0;

    /// The only writer of wrapper fields (friendship, r01 §2.1).
    template <class W, class P>
    static void write_wrapper(W& w, const Geometry& g, P* data) {
        w.shape_  = make_range<W::rank>(g.shape);
        w.stride_ = make_range<W::rank>(g.stride);
        w.data_   = data;
    }

    /// The only writer of ragged wrapper fields.
    template <class W, class P>
    static void write_ragged(W& w, std::size_t n, P* data, const std::uint64_t* offsets,
                             const std::array<std::size_t, 2>& ext, const std::array<const std::uint64_t*, 2>& per) {
        w.n_       = n;
        w.data_    = data;
        w.offsets_ = offsets;
        w.ext_     = ext;
        w.per_     = per;
    }

    /// Mark this entry as declared with a ragged wrapper.
    void mark_ragged_port() noexcept { ragged_port_ = true; }

    /// Registered in get_exec_info (System/Component); appends if absent, so
    /// the first registration is the primary.
    void register_placement(const Placement& p) {
        if (!is_root()) throw std::logic_error("naga::cospan: only roots register placements");
        for (const Placement& e : placements_)
            if (e == p) return;
        placements_.push_back(p);
    }

    /// The root's own placement is the primary whatever registered first
    /// (an input peer's owner may run get_exec_info before the root's).
    void set_primary_placement(const Placement& p) {
        if (!is_root()) throw std::logic_error("naga::cospan: only roots register placements");
        for (auto it = placements_.begin(); it != placements_.end(); ++it)
            if (*it == p) {
                placements_.erase(it);
                break;
            }
        placements_.insert(placements_.begin(), p);
    }

    /// The extents this entry reports: its own, or (a promotion) its member's, live.
    const std::array<std::size_t, 3>& shape_ref() const noexcept {
        return mirror_of_ != nullptr ? mirror_of_->shape_ref() : extents_;
    }

    std::array<std::size_t, 3> extents_{1, 1, 1};  ///< 1-padded; a per-index dim reports 0
    // ---- the applied ragged description of THIS entry (host) ----
    bool                       ragged_ = false;          ///< any dim >= 1 is per-index
    std::vector<std::uint64_t> offsets_;                 ///< ragged: N + 1 block offsets
    std::array<std::vector<std::size_t>, 3> applied_per_;  ///< the per-index arrays last applied
    std::uint64_t              ragged_version_ = 0;      ///< bumped whenever the ragged description changes
    bool                       ragged_port_ = false;     ///< declared with RaggedInput / RaggedOutput
    // ---- roots: kept across the chunks of a ChunkedSystem ----
    bool                       held_          = false;   ///< set by the ChunkedSystem: the capacity is reserved for the chunk loop
    std::size_t                reserve_elems_ = 0;       ///< held: the capacity (elements) allocate() takes
    std::size_t                allocations_   = 0;       ///< storage allocations of this root (allocate, growth past capacity)
    std::uint64_t              write_epoch_   = 0;       ///< roots: see write_epoch()
    std::uint64_t              shape_version_ = 0;       ///< roots: see shape_version()
    /// ChunkedSystem replay, INTERNAL roots: the first index of the current chunk. point_ragged adds it to the
    /// device offsets / per-index pointers (the whole run's arrays are uploaded once); the data base is unchanged.
    std::size_t                meta_shift_    = 0;
    mutable const void*          probe_data_ = nullptr;  ///< test probe: the data pointer the wrapper was last given
    mutable const std::uint64_t* probe_offs_ = nullptr;  ///< test probe: the offsets pointer the wrapper was last given
    /// Inside a ChunkedSystem the framework sets dim 0 of every port of every inside component to the
    /// current chunk length (N in the full-range pass, b - a per chunk), whatever rule declared it.
    std::optional<std::size_t> dim0_override_;
    std::vector<DimRecord>     dims_;
    std::vector<Placement>     placements_;        ///< roots; primary first
    std::optional<sycl::device> device_;
    sk::kind                   storage_kind_ = sk::kind::device;
    std::shared_ptr<void>      bundle_;               ///< a port: the bundle of ITS placement (null while not materialized); keeps it alive

private:
    /// The four structural operations and placement are System's (r01 §4.2);
    /// EntryAccess remains the test-only stand-in for the entry-level smoke.
    friend class naga::cospan::System;
    friend class naga::cospan::ComponentBase;
    friend struct EntryAccess;
    friend struct LeadingRangeAccess;
    friend class naga::cospan::ChunkedSystem;
    friend class AliasEvaluator;
    friend class Variable;
    template <class T>
    friend class StorageEntry;

    /// Framework only (through LeadingRangeAccess). A NON-root input peer sees rows / blocks
    /// [a, b) of its root along dim 0. Dense root: its view becomes the SubarrayRef with dim-0 offset a
    /// and extent b - a (composed with a setup-time Slice, which is kept and restored by clear); the
    /// peer's extents follow through update_shape. Ragged root: N = b - a, the host per-index arrays are
    /// the root's [a, b) slice, and point_ragged shifts the device pointers by a. Re-runs the peer's shape
    /// update and validate_connection and re-points its wrapper, so the next forward uses the range.
    void set_leading_range(std::size_t a, std::size_t b, bool refresh = true) {
        // what used to be the root: the shadow, or the authority port (the source output, a tie, a free input), except
        // under a chunked boundary, where the inside member is the authority and is ranged like any peer
        if (is_root() || mirror_of_ != nullptr || (is_authority() && !root_->chunk_rerooted_))
            throw std::logic_error("naga::cospan: '" + name_ + "': a leading range is set on a connected input (or an output peer of a root at a chunked system's level), not on a root");
        const IOEntry&                    r    = *root_;
        const std::optional<SubarrayRef>& base = lead_ ? pre_lead_view_ : view_;
        if (rank_ != r.rank_)
            throw std::invalid_argument("naga::cospan: '" + name_ + "' is rank " + std::to_string(rank_) + " but its root '" +
                                        r.name_ + "' is rank " + std::to_string(r.rank_) + "; a leading range needs the root's rank");
        if (r.is_ragged() && base)
            throw std::invalid_argument("naga::cospan: '" + name_ + "' sees a subarray of the ragged '" + r.name_ +
                                        "'; ragged roots cannot be sliced");
        const std::size_t n = base ? base->extents[0] : r.extents_[0];
        if (a > b)
            throw std::invalid_argument("naga::cospan: '" + name_ + "': leading range [" + std::to_string(a) + ", " +
                                        std::to_string(b) + ") has a > b");
        if (b > n)
            throw std::invalid_argument("naga::cospan: '" + name_ + "': leading range [" + std::to_string(a) + ", " +
                                        std::to_string(b) + ") is past the " + std::to_string(n) +
                                        " leading indices of '" + r.name_ + "'");
        if (!r.is_ragged()) {
            SubarrayRef v;
            v.rank = r.rank_;
            for (std::size_t d = 0; d < 3; ++d) {
                v.offset[d]  = base ? base->offset[d] : 0;
                v.extents[d] = base ? base->extents[d] : r.extents_[d];
            }
            v.offset[0] += a;
            v.extents[0] = b - a;
            if (!lead_) pre_lead_view_ = view_;
            view_ = v;
        } else if (!lead_) {
            pre_lead_view_ = view_;  // empty: ragged roots have no views
        }
        lead_ = std::make_pair(a, b);
        if (refresh) refresh_after_range();  // the replaying ChunkedSystem applies the cached shape instead
    }
    /// Back to the whole range (and the setup-time view, if any).
    void clear_leading_range(bool refresh = true) {
        if (!lead_) return;
        view_ = pre_lead_view_;
        pre_lead_view_.reset();
        lead_.reset();
        if (refresh) refresh_after_range();
    }
    void refresh_after_range() {
        if (io_kind_ == IOKind::input) {
            update_shape();
            validate_connection();
        }  // an output peer's shape is its component's to resolve (from its inputs) at the next forward
        if (root_->present()) point_wrapper();
    }

    /// ChunkedSystem, on an alias ROOT at its level: take the full resolved shape of `m`, the
    /// inside entry it re-rooted (dense: its extents; ragged: N and the per-index arrays),
    /// and apply it (resizing storage when present). The alias dims are inert markers (how from_input),
    /// so the root is dynamic.
    void adopt_shape(const IOEntry& m) {
        if (!is_root() || m.rank_ != rank_)
            throw std::logic_error("naga::cospan: '" + name_ + "': adopt_shape needs a root and a member of its rank");
        for (std::size_t d = 0; d < static_cast<std::size_t>(rank_); ++d) {
            DimRecord&       r = dims_[d];
            const DimRecord& s = m.dims_[d];
            if (d >= 1 && m.ragged_ && s.per) {
                r.per       = true;
                r.per_index = s.per_index;
                r.extent    = 0;
            } else {
                r.per = false;
                r.per_index.clear();
                r.extent = m.extents_[d];
            }
        }
        update_shape();
    }

    /// The authority port, after its shape changed: the shadow takes it (a no-op for any other entry).
    void push_shape_to_shadow() {
        if (is_authority() && root_ != nullptr && root_->rank_ == rank_) root_->adopt_shape(*this);
    }

    /// merge(into, from): the two entries' Variables become one (into's survives).
    static void merge(IOEntry& into, IOEntry& from) {
        Variable& a = into.variable();
        Variable& b = from.variable();
        if (&a == &b) return;
        a.absorb(b);
    }

    std::string name_;
    IOKind      io_kind_;
    DType       dtype_;
    int         rank_;

    IOEntry*              root_ = nullptr;   ///< the Variable's shadow (a shadow points at itself)
    std::shared_ptr<Variable> var_;           ///< a port: its Variable (null on a shadow)
    Variable*             home_ = nullptr;   ///< a shadow: the Variable it belongs to
    bool                  can_be_source_;
    bool                  can_be_target_;
    std::optional<SubarrayRef> view_;
    std::optional<std::pair<std::size_t, std::size_t>> lead_;  ///< the runtime leading range
    std::optional<SubarrayRef> pre_lead_view_;                 ///< the setup-time view saved while lead_ is set
    const Subsystem*      declarer_  = nullptr;

    std::optional<MarkRecord> design_var_, constraint_, objective_;

    // ---- ties own the shape; promotions mirror ----
    const IOEntry*            mirror_of_ = nullptr;  ///< a promoted alias: the entry it was made from
    IOEntry*                  tie_alias_ = nullptr;  ///< members of a tie (and their set): the tie alias
    std::optional<AliasShape> alias_shape_;          ///< a tie alias: where its shape comes from
    AliasSrc                  alias_src_;            ///< a tie alias: the upstream entry of a from_output_* / from_input_shape shape
    std::vector<AliasSrc>     tie_members_;          ///< a tie alias: the members' variables (and ports, for setup checks), as given
    bool                      alias_checked_ = false;  ///< a connected tie_inputs alias: its AliasShape was checked

    /// An initial value declared on THIS entry (ComponentBase::init_input / System::init_input),
    /// with the depth of the declaring subsystem (root System = 0). resolve() picks, per root,
    /// the declaration of the smallest depth.
    using InitDecl = ::naga::cospan::detail::InitDecl;  // variable.hpp
    std::vector<InitDecl>     init_decls_;
    std::optional<InitSpec>   init_spec_;        ///< roots, set by System::resolve or Model::init_input
    bool                      model_init_   = false;  ///< roots: init_spec_ came from Model::init_input (resolve keeps it)
    bool                      init_pending_ = false;  ///< roots: the init is to be applied at the owner's next run
    bool                      chunk_outside_tie_ = false;  ///< a tie alias declared outside a ChunkedSystem whose inside members make it a boundary root: the chunked system resolves its shape
    bool                      chunk_rerooted_ = false;  ///< a root made at a ChunkedSystem's level (its owner allocates it)
    std::string               chunk_internal_by_;       ///< roots internal to the named ChunkedSystem
    Substrates*               substrates_ = nullptr; ///< roots, bound at prepare_exec
};

// -----------------------------------------------------------------------------
// Storage layer: everything that depends on the element type only
// -----------------------------------------------------------------------------

template <class T>
class StorageEntry : public IOEntry {
public:
    void update_shape() override {
        update_shape_local();
        push_shape_to_shadow();
    }

    void update_shape_local() {
        std::array<std::size_t, 3> next = extents_;
        for (int d = 0; d < rank(); ++d) {
            DimRecord& r = dims_[static_cast<std::size_t>(d)];
            if (d == 0 && dim0_override_) {  // inside a ChunkedSystem: dim 0 IS the chunk length,
                next[0] = *dim0_override_;   // whatever rule declared it
                continue;
            }
            switch (r.how) {
                case DimRecord::How::fixed: break;
                case DimRecord::How::unspecified:
                    // a free input root, or the inside member of one a ChunkedSystem re-rooted: its initializer's shape
                    // (setup guarantees one exists)
                    if (local_shape() || (root()->is_rerooted() && root()->kind() == IOKind::input)) {
                        if (const auto& is = init_spec(); is && is->shape() && is->shape()->rank == rank()) {
                            const InitShape&  s = *is->shape();
                            const auto        u = static_cast<std::size_t>(d);
                            if (d >= 1 && s.per[u]) {  // a ragged shape: the per-index array of the full range
                                r.per       = true;
                                r.per_index = s.per_index[u];
                                r.extent    = 0;
                            } else {
                                r.per = false;
                                r.per_index.clear();
                                r.extent = s.extents[u];
                            }
                        }
                    } else {  // supplied by the connection, the slice or the tie
                        check_rank();
                        r.extent = source_extent(static_cast<uint8_t>(d));
                        if (d >= 1) {
                            r.per_index = source_per_index(static_cast<uint8_t>(d));
                            r.per       = source_per_flag(static_cast<uint8_t>(d));
                        }
                    }
                    break;
                case DimRecord::How::from_input_dim:    // resolved by the Component (which also copies a
                case DimRecord::How::from_input:        // per-index source dimension's array)
                    break;
                case DimRecord::How::per_index_host:
                    // Declared at full length; inside a ChunkedSystem dim 0 is the chunk length and the array is cut to the
                    // chunk, as for a tie alias. Outside, it is the declared array again.
                    if (d >= 1) {
                        if (!dim0_override_) {
                            if (r.per_index != r.host_full) r.per_index = r.host_full;
                        } else if (r.host_full.size() != *dim0_override_) {
                            const std::size_t first = 0, len = *dim0_override_;
                            if (r.host_full.size() < first + len)
                                throw std::invalid_argument("naga::cospan: '" + full_path() + "': per-index dimension " +
                                                            std::to_string(d) + " has " + std::to_string(r.host_full.size()) +
                                                            " extents but the chunk reaches index " + std::to_string(first + len));
                            r.per_index.assign(r.host_full.begin() + static_cast<std::ptrdiff_t>(first),
                                               r.host_full.begin() + static_cast<std::ptrdiff_t>(first + len));
                        } else if (r.per_index != r.host_full) {
                            r.per_index = r.host_full;
                        }
                    }
                    break;
                case DimRecord::How::per_index_from_input: break;  // resolved by the Component
            }
            next[static_cast<std::size_t>(d)] = (d >= 1 && r.per) ? 0 : r.extent;
        }

        // ---- the ragged description. Every per-index array must have one entry per index of dim 0.
        bool ragged = false;
        for (int d = 1; d < rank(); ++d)
            if (dims_[static_cast<std::size_t>(d)].per) ragged = true;
        std::vector<std::uint64_t> offs;
        if (ragged) {
            const std::size_t n = next[0];
            for (int d = 1; d < rank(); ++d) {
                const DimRecord& r = dims_[static_cast<std::size_t>(d)];
                if (r.per && r.per_index.size() != n)
                    throw std::invalid_argument("naga::cospan: '" + full_path() + "': per-index dimension " +
                                                std::to_string(d) + " has " + std::to_string(r.per_index.size()) +
                                                " extents but dimension 0 has " + std::to_string(n));
            }
            offs.assign(n + 1, 0);
            for (std::size_t i = 0; i < n; ++i) {
                std::uint64_t block = 1;
                for (int d = 1; d < rank(); ++d) {
                    const DimRecord& r = dims_[static_cast<std::size_t>(d)];
                    block *= r.per ? r.per_index[i] : r.extent;
                }
                offs[i + 1] = offs[i] + block;
            }
        }
        bool per_changed = ragged != ragged_;
        for (int d = 1; d < rank(); ++d)
            if (dims_[static_cast<std::size_t>(d)].per_index != applied_per_[static_cast<std::size_t>(d)]) per_changed = true;
        if (ragged && offs != offsets_) per_changed = true;

        if (next == extents_ && !per_changed) return;
        if (!ragged && next != extents_ && !ragged_) {
            reshape(sycl::range<3>{next[0], next[1], next[2]});  // the dense path, as before
            return;
        }
        extents_ = next;
        ragged_  = ragged;
        offsets_ = std::move(offs);
        for (int d = 0; d < 3; ++d)
            applied_per_[static_cast<std::size_t>(d)] =
                (d >= 1 && d < rank()) ? dims_[static_cast<std::size_t>(d)].per_index : std::vector<std::size_t>{};
        ++ragged_version_;
        if (is_root()) ++shape_version_;
        resize_storage();
    }

    void reshape(sycl::range<3> r) override {
        if (ragged_)
            throw std::logic_error("naga::cospan: '" + full_path() + "' is ragged: its shape follows its per-index dimensions and cannot be reshaped");
        for (int d = rank(); d < 3; ++d)
            if (r[d] != 1)
                throw std::invalid_argument("naga::cospan: '" + std::string{name()} +
                                            "': extents past the rank must be 1");
        const std::array<std::size_t, 3> next{r[0], r[1], r[2]};
        if (next != extents_ && is_root()) ++shape_version_;
        extents_ = next;
        resize_storage();
    }

    /// What supplies the shape of this entry's root, for an error message.
    std::string supplier_str() const {
        const IOEntry& r = *root();
        return "'" + r.full_path() + "' (" +
               (r.alias_shape() ? std::string{"a tie"} : r.kind() == IOKind::output ? std::string{"the connected output"}
                                                                                       : std::string{"its initializer"}) +
               ")";
    }
    static std::string per_str(const std::vector<std::size_t>& v) {
        std::string out = "per-index [";
        for (std::size_t i = 0; i < v.size() && i < 8; ++i) out += (i ? ", " : "") + std::to_string(v[i]);
        if (v.size() > 8) out += ", ...";
        return out + "]";
    }

    /// The rank must always match the supplier's (a view has its root's rank).
    void check_rank() const {
        if (local_shape()) return;
        const auto v   = view();
        const int  sup = v ? v->rank : root()->rank();
        if (sup != rank())
            throw std::invalid_argument("naga::cospan: shape mismatch: '" + full_path() + "' has rank " + std::to_string(rank()) +
                                        " but its shape is supplied by " + supplier_str() + ", which has rank " +
                                        std::to_string(sup) + " (the rank must match)");
    }

    void check_constraints(bool fixed_only) const override {
        if (local_shape()) return;
        check_rank();
        const IOEntry&  r   = *root();
        const auto      v   = view();
        const std::string who = "naga::cospan: shape mismatch: '" + full_path() + "'";
        const auto lr = leading_range();
        for (int d = 0; d < rank(); ++d) {
            const DimRecord& rec = dims_[static_cast<std::size_t>(d)];
            if (rec.how == DimRecord::How::unspecified) continue;
            if (fixed_only && rec.how != DimRecord::How::fixed) continue;
            if (d == 0 && dim0_override_) continue;  // inside a ChunkedSystem dim 0 is the chunk length (checked at the full length)
            const auto u = static_cast<std::size_t>(d);
            const bool want_per = d >= 1 && rec.per;
            const bool sup_per  = d >= 1 && source_per_flag(static_cast<uint8_t>(d));
            const std::string lead = who + " dimension " + std::to_string(d) + " is " + declared_rule(d);
            if (want_per || sup_per) {
                if (fixed_only) continue;
                if (want_per != sup_per)
                    throw std::invalid_argument(lead + (want_per ? " (per-index)" : " = " + std::to_string(extents_[u])) +
                                                " but its shape is supplied by " + supplier_str() + ", whose dimension " +
                                                std::to_string(d) + (sup_per ? " is per-index" : " is dense"));
                const auto have = per_index_extents(d);
                const auto want = source_per_index(static_cast<uint8_t>(d));
                if (have != want)
                    throw std::invalid_argument(lead + " = " + per_str(have) + " but its shape is supplied by " + supplier_str() +
                                                ", whose dimension " + std::to_string(d) + " is " + per_str(want));
                continue;
            }
            std::size_t supplied = r.extents()[u];
            if (lr && d == 0)
                supplied = lr->second - lr->first;
            else if (v)
                supplied = v->extents[u];
            if (extents_[u] != supplied)
                throw std::invalid_argument(lead + " = " + std::to_string(extents_[u]) + " but its shape is supplied by " +
                                            supplier_str() + ", whose dimension " + std::to_string(d) + " is " +
                                            std::to_string(supplied));
        }
    }

    void validate_connection() const override {
        if (kind() != IOKind::input || local_shape()) return;
        const IOEntry& r        = *root();
        if (r.is_ragged() || is_ragged()) {  // the per-index extents must match dim by dim, and the totals
            if (r.is_ragged() != is_ragged())
                throw std::invalid_argument("naga::cospan: input '" + std::string{name()} + "' is " +
                                            (is_ragged() ? "ragged" : "dense") + " but is connected to '" +
                                            std::string{r.name()} + "', which is " + (r.is_ragged() ? "ragged" : "dense"));
            if (view())
                throw std::invalid_argument("naga::cospan: '" + std::string{name()} + "' sees a subarray of the ragged '" +
                                            std::string{r.name()} + "'; ragged roots cannot be sliced");
            const auto        lr = leading_range();  // compare against the root's [a, b)
            const std::size_t a  = lr ? lr->first : 0;
            const std::size_t rn = lr ? lr->second - lr->first : r.extents()[0];
            if (extents()[0] != rn)
                throw std::invalid_argument("naga::cospan: ragged input '" + std::string{name()} + "' has " +
                                            std::to_string(extents()[0]) + " blocks but '" + std::string{r.name()} +
                                            "' has " + std::to_string(rn));
            for (int d = 1; d < rank(); ++d) {
                std::vector<std::size_t> want = r.per_index_extents(d);
                if (lr) {
                    if (lr->second > want.size())
                        throw std::invalid_argument("naga::cospan: ragged input '" + std::string{name()} + "': its leading range is past the blocks of '" + std::string{r.name()} + "'");
                    want = std::vector<std::size_t>(want.begin() + static_cast<std::ptrdiff_t>(a),
                                                    want.begin() + static_cast<std::ptrdiff_t>(lr->second));
                }
                if (per_index_extents(d) != want)
                    throw std::invalid_argument("naga::cospan: ragged input '" + std::string{name()} + "': the per-index extents of dimension " +
                                                std::to_string(d) + " differ from those of '" + std::string{r.name()} + "'");
            }
            const std::size_t want_total =
                lr ? static_cast<std::size_t>(r.block_offset_at(lr->second) - r.block_offset_at(lr->first)) : r.flat_size();
            if (flat_size() != want_total)
                throw std::invalid_argument("naga::cospan: ragged input '" + std::string{name()} + "' holds " +
                                            std::to_string(flat_size()) + " elements but '" + std::string{r.name()} +
                                            "' holds " + std::to_string(want_total));
            return;
        }
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

    /// Create a Bundle for every registered placement that has no live one (r01 §4.5). A placement that comes
    /// back copies from a live one (no init here). The new bundles go into `fresh`, which the caller holds until
    /// every port has taken its own. Returns whether any bundle was created.
    bool allocate_missing(sk::queue& q, Substrates& subs, std::vector<std::shared_ptr<void>>& fresh) override {
        if (!is_root()) return false;
        if (placements_.empty())
            register_placement(Placement{device_ ? *device_ : q.get_device(), storage_kind_});
        subs_ = &subs;
        std::shared_ptr<void> donor_hold;  // a live placement to copy from
        for (const Placement& p : placements_)
            if ((donor_hold = variable().live_bundle(p))) break;
        auto* donor = static_cast<Bundle<T>*>(donor_hold.get());
        bool  any   = false;
        for (const Placement& p : placements_) {
            if (variable().live_bundle(p)) continue;
            // A size-0 root gets an empty vector (usm_vector(0, ...) allocates nothing). A HELD root
            // takes its reserved capacity: per chunk it is resized within it, which never reallocates.
            const std::size_t n = held_ ? reserve_elems_ : flat_size();
            auto b = std::make_shared<Bundle<T>>(p, sk::usm_vector<T>(n, route(q, subs, p, n)));
            ++allocations_;
            diag::count_allocations();
            bump_write();  // a new bundle: whatever a cache held is gone
            if (donor != nullptr && flat_size() != 0) b->data = donor->data;  // routed, ordered copy
            variable().set_live(p, b);
            fresh.push_back(b);
            any = true;
        }
        if (any) upload_ragged(q);
        return any;
    }

    /// Every live bundle of the Variable (typed).
    std::vector<std::shared_ptr<Bundle<T>>> bundles() const {
        std::vector<std::shared_ptr<Bundle<T>>> out;
        for (const auto& l : variable().live)
            if (auto b = l.bundle.lock()) out.push_back(std::static_pointer_cast<Bundle<T>>(b));
        return out;
    }

    void upload_ragged(sk::queue& q) override {
        if (!is_root() || !present() || subs_ == nullptr) return;
        for (const auto& sp : bundles()) {
            Bundle<T>* s = sp.get();
            if (s->version == ragged_version_) continue;
            diag::count_ragged_uploads();
            // The ragged index arrays follow the DENSE storage rules: a vector is created once (empty), kept, and
            // rewritten in place. from_host into the existing capacity is a memory task the in-order stream orders
            // after any kernel still reading the old values, so it needs no wait. Only GROWTH past the capacity
            // releases the old storage, and only then are its readers waited for (once per bundle).
            const auto upload = [&](std::optional<sk::usm_vector<std::uint64_t>>& v, std::span<const std::uint64_t> src) {
                if (!v) v.emplace(0, route_as<std::uint64_t>(q, *subs_, s->pl));  // empty: nothing allocated yet
                if (src.size() > v->capacity()) {                                  // growth, as for a dense root
                    if (v->capacity() > 0) {                                       // existing storage is released
                        diag::count_frees();  // the old storage's free waits on its registered events
                    }
                    diag::count_allocations();
                }
                diag::count_host_tasks();
                v->from_host(src);  // grows when needed; copies into the existing capacity otherwise
            };
            if (ragged_) {
                upload(s->offs, std::span<const std::uint64_t>{offsets_});
                for (int d = 1; d < rank(); ++d) {
                    if (!dims_[static_cast<std::size_t>(d)].per) continue;
                    const auto& src = dims_[static_cast<std::size_t>(d)].per_index;
                    const std::vector<std::uint64_t> conv(src.begin(), src.end());  // from_host copies it into a task-owned buffer
                    upload(s->per[static_cast<std::size_t>(d - 1)], std::span<const std::uint64_t>{conv});
                }
            }
            // not ragged any more: the vectors stay (point_ragged ignores them for a dense entry)
            s->version = ragged_version_;
        }
    }

    const std::uint64_t* ragged_ptr_on(const Placement& p, int which) const override {
        Bundle<T>* s = find_slot(p);
        if (s == nullptr) return nullptr;
        if (which == 0) return s->offs ? s->offs->data() : nullptr;
        if (which < 1 || which > 2) return nullptr;
        const auto& a = s->per[static_cast<std::size_t>(which - 1)];
        return a ? a->data() : nullptr;
    }

    std::size_t capacity_elems_impl() const noexcept override {
        const auto bs = bundles();
        return bs.empty() ? 0 : bs.front()->data.capacity();
    }

    void sync_other_placements(sk::queue&) override {
        if (!is_root()) throw std::logic_error("naga::cospan: sync_other_placements is for roots");
        const auto bs = bundles();
        if (bs.size() < 2 || flat_size() == 0) return;
        Bundle<T>& primary = primary_slot();
        for (const auto& s : bs)
            if (s.get() != &primary) s->data = primary.data;  // routed, ordered copy
    }

    /// A host copy of the WHOLE root, from the primary storage. Device kind:
    /// usm_vector::host_view (an owned copy, ordered after the stream's prior
    /// work). Host kind: queue.wait() then get_access().
    ///
    /// CACHED: the copy is kept on the root and served again while the root's write epoch is unchanged
    /// (no device access, no wait). primary_slot() runs FIRST, so a released or not-yet-present root throws
    /// exactly as before and a hit never bypasses that. The epoch is recorded BEFORE the ordered read, so a
    /// write submitted afterwards is never missed. The reference is valid until the next read_primary of this
    /// root that refreshes, or its release; copy it if you need to keep it. Single-threaded use is assumed
    /// (as everywhere in CoSpan today: submission and host reads come from one thread).
    const std::vector<T>& read_primary(sk::queue& q) const {
        Bundle<T>& s = primary_slot();
        const std::uint64_t e = write_epoch();
        if (cache_epoch_ == e) {
            diag::count_host_read_hits();
            return host_cache_;
        }
        diag::count_host_reads();
        if (s.data.get_kind() == sk::kind::device) {
            auto hv = s.data.host_view();
            host_cache_.assign(hv.begin(), hv.end());
        } else {
            diag::count_queue_waits();
            q.wait();
            auto acc = s.data.get_access();
            host_cache_.assign(acc.begin(), acc.end());
        }
        cache_epoch_ = e;
        return host_cache_;
    }

    /// Replace the primary storage's contents from the host, then refresh the
    /// copies on the other placements.
    void write_primary(sk::queue& q, std::span<const T> src) {
        diag::count_host_tasks();
        primary_slot().data.from_host(src);
        bump_write();
        sync_other_placements(q);
    }

protected:
    StorageEntry(std::string name, IOKind k, DType dt, int rank, std::vector<DimRecord> dims,
                 bool can_source, bool can_target, Variable* shadow_home = nullptr)
        : IOEntry{std::move(name), k, dt, rank, std::move(dims), can_source, can_target, shadow_home} {}


    void register_events_on(const Placement& p, std::span<const sycl::event> events) override {
        if (!is_root() || !present()) return;
        Bundle<T>* s = find_slot(p);
        if (s == nullptr) return;
        s->data.register_events(events);
        if (s->offs) s->offs->register_events(events);  // the kernel reads the index arrays too
        for (auto& a : s->per)
            if (a) a->register_events(events);
    }

    std::size_t registered_events_on(const Placement& p) const override {
        if (!is_root() || !present()) return 0;
        const Bundle<T>* s = find_slot(p);
        return s == nullptr ? 0 : s->data.registered_event_count();
    }

    void* raw_storage_on(const Placement& p) const override {
        Bundle<T>* s = find_slot(p);
        if (s == nullptr)
            throw std::logic_error("naga::cospan: '" + full_path() +
                                   "' has no storage on that placement (it is not materialized)");
        return s->data.data();
    }

private:
    /// Resize the root's storage to flat_size(); growing waits the stream first (synchronous frees).
    void resize_storage() {
        if (!is_root()) return;
        const auto bs = bundles();
        if (!bs.empty()) diag::count_resizes();
        if (!bs.empty() && flat_size() > bs.front()->data.capacity()) {
            ++allocations_;  // a usm_vector resize within its capacity never reallocates; this one does
            diag::count_allocations();
        }
        const bool size_changes = !bs.empty() && bs.front()->data.size() != flat_size();
        for (const auto& s : bs) s->data.resize(flat_size());
        if (size_changes) bump_write();
    }

    /// A per-index array of the SOURCE's dimension d (empty: scalar), for an unspecified dim.
    std::vector<std::size_t> source_per_index(uint8_t d) const {
        if (local_shape()) return {};  // source_extent throws for it before this is reached
        if (view()) return {};
        const std::vector<std::size_t>& full = root()->per_index_raw(d);
        if (const auto lr = leading_range(); lr && source_per_flag(d)) {  // the root's [a, b) slice
            if (lr->second > full.size())
                throw std::out_of_range("naga::cospan: '" + std::string{name()} + "': leading range [" +
                                        std::to_string(lr->first) + ", " + std::to_string(lr->second) +
                                        ") is past the root's " + std::to_string(full.size()) + " blocks");
            return std::vector<std::size_t>(full.begin() + static_cast<std::ptrdiff_t>(lr->first),
                                            full.begin() + static_cast<std::ptrdiff_t>(lr->second));
        }
        return full;
    }

    /// Whether the root's dimension d is per-index AND already resolved (its array has one entry per
    /// index of dim 0): before its owner resolves a FromInputValues dim the array is empty, and a peer
    /// updating its shape at setup must not read that as a length mismatch.
    bool source_per_flag(uint8_t d) const {
        if (local_shape() || view()) return false;
        const DimRecord& rd = root()->dim(d);
        return rd.per && rd.per_index.size() == root()->extents()[0];
    }
    bool root_per_flag(uint8_t d) const { return source_per_flag(d); }

    /// A tied entry's per-index array of dimension d: its root's. Under a runtime leading range (
    /// an inside peer of a boundary root in a ChunkedSystem) it is the root's [a, b) slice, exactly as
    /// source_per_index gives it.
    std::vector<std::size_t> root_per_index(uint8_t d) const {
        if (local_shape()) return {};
        if (view()) return {};
        return source_per_index(d);
    }

    /// The bundle at a placement; valid while a port (or a caller's shared_ptr) holds it.
    Bundle<T>* find_slot(const Placement& p) const { return static_cast<Bundle<T>*>(variable().live_bundle(p).get()); }

    Bundle<T>& primary_slot() const {
        const auto no_data = [&] {
            return std::logic_error("naga::cospan: '" + full_path() +
                                    "' has no data: it is not materialized (it comes into being when something that "
                                    "uses it runs)");
        };
        if (!is_root() || !present() || placements_.empty()) throw no_data();
        Bundle<T>* s = find_slot(placements_.front());
        if (s == nullptr) throw no_data();
        return *s;
    }

    /// A tied entry's dimension d: its root's (or, under a view, the view's) extent. The root of a
    /// tied entry is a tie alias (or the output a tie_inputs alias is connected to), whose shape
    /// the root's owner resolved before this entry's Component runs.
    std::size_t root_extent(uint8_t d) const {
        if (local_shape())
            throw std::logic_error("naga::cospan: '" + full_path() + "' takes a dimension from a tie but is not in a tie");
        if (const auto v = view()) return v->extents[d];
        if (const auto lr = leading_range(); lr && d == 0) return lr->second - lr->first;  // under a range
        return root()->extents()[d];
    }

    std::size_t source_extent(uint8_t d) const {
        if (local_shape())
            throw std::logic_error("naga::cospan: unspecified dimension on the root '" + std::string{name()} + "'");
        if (const auto v = view()) {
            if (d >= v->rank) throw std::out_of_range("naga::cospan: an unspecified dimension is past its supplier's rank");
            return v->extents[d];
        }
        if (d >= root()->rank())
            throw std::out_of_range("naga::cospan: an unspecified dimension is past its supplier's rank");
        if (const auto lr = leading_range(); lr && d == 0) return lr->second - lr->first;  // a ragged root
        return root()->extents()[d];
    }

    /// r01 §4.4 as amended by D3: a host-kind placement takes the OS route
    /// with its device choosing the context; a device-kind placement of a
    /// fully fixed root outside a transient subtree takes the arena (never
    /// falling back: an arena is a budget); everything else the pool if the
    /// device has one, else the OS.
    sk::allocator<T> route(sk::queue& q, Substrates& subs, const Placement& p, std::size_t elems) const {
        if (p.kind != sk::kind::device) return sk::allocator<T>{q, p.kind, p.device};
        // An empty root allocates nothing, so it needs no arena or pool: a fixed size-0 root is
        // arena-eligible but its device has no arena (the arena is sized from the non-empty roots).
        if (elems == 0) return sk::allocator<T>{q, sk::kind::device, p.device};
        const sycl::device& d = p.device;
        const auto it = subs.find(d);
        if (arena_planned_ && arena_eligible()) {
            if (it == subs.end() || !it->second.arena)
                throw std::logic_error("naga::cospan: '" + std::string{name()} +
                                       "' takes the arena route but its device has no arena");
            return sk::allocator<T>{q, *it->second.arena};
        }
        if (it != subs.end() && it->second.pool) return sk::allocator<T>{q, *it->second.pool};
        return sk::allocator<T>{q, sk::kind::device, d};
    }

    /// The route of a ragged index array (never the arena: a ragged root is dynamic).
    template <class U>
    sk::allocator<U> route_as(sk::queue& q, Substrates& subs, const Placement& p) const {
        if (p.kind != sk::kind::device) return sk::allocator<U>{q, p.kind, p.device};
        const auto it = subs.find(p.device);
        if (it != subs.end() && it->second.pool) return sk::allocator<U>{q, *it->second.pool};
        return sk::allocator<U>{q, sk::kind::device, p.device};
    }

    mutable std::vector<T>             host_cache_;                      // the primary's host copy, valid at cache_epoch_
    mutable std::uint64_t              cache_epoch_ = ~std::uint64_t{0};  // the write epoch host_cache_ was read at
    Substrates*                        subs_ = nullptr;  // bound at allocate (the ragged arrays route through it)
};

inline Variable::~Variable() = default;

/// A port leaves (its destructor). The authority and the source stay NAMED (authority_path, source_path): a
/// rebuilt port with that path takes the role again when it joins.
inline void Variable::unregister_port(IOEntry* p) {
    std::erase(ports, p);
    if (authority == p) {
        authority      = nullptr;
        authority_path = p->full_path();
    }
    if (source_port == p) source_port = nullptr;  // source_path keeps naming it
    for (IOEntry* q : ports)
        if (q->mirror_of_ == p) q->mirror_of_ = nullptr;  // a promotion never points at a deleted port
}

inline void Variable::absorb(Variable& other) {
    if (&other == this) return;
    const auto keep = other.shared_from_this();  // the last port leaving must not free `other` mid-loop
    const auto self = shared_from_this();
    for (IOEntry* p : other.ports) {
        p->var_  = self;
        p->root_ = shadow.get();
        ports.push_back(p);
        if (authority == nullptr && !authority_path.empty() && p->full_path() == authority_path) authority = p;
    }
    users.insert(users.end(), other.users.begin(), other.users.end());
    other.users.clear();
    other.ports.clear();
    other.authority   = nullptr;
    other.merged_into = self;
    // Fold the root-only state of the dying shadow into the survivor's (a mark or a Model init the survivor
    // lacks moves over; a MarkRecord keeps naming the set's root, which is now the surviving shadow).
    if (shadow && other.shadow) {
        IOEntry&       to   = *shadow;
        IOEntry&       from = *other.shadow;
        const auto     mv   = [&](std::optional<MarkRecord>& dst, std::optional<MarkRecord>& src) {
            if (!dst && src) {
                dst       = std::move(src);
                dst->root = shadow.get();
            }
        };
        mv(to.design_var_, from.design_var_);
        mv(to.constraint_, from.constraint_);
        mv(to.objective_, from.objective_);
        if (!to.model_init_ && from.model_init_) {
            to.init_spec_    = std::move(from.init_spec_);
            to.model_init_   = true;
            to.init_pending_ = from.init_pending_;
        }
        to.chunk_outside_tie_ = to.chunk_outside_tie_ || from.chunk_outside_tie_;
        to.chunk_rerooted_    = to.chunk_rerooted_ || from.chunk_rerooted_;
        if (to.chunk_internal_by_.empty()) to.chunk_internal_by_ = from.chunk_internal_by_;
    }
    if (source_port == nullptr && other.source_port != nullptr) {
        source_port = other.source_port;
        source_path = other.source_path;
    }
    other.shadow.reset();
}

/// The rank/dtype-typed layer: the wrapper pointer helpers.
template <DType DT, int Rank>
class TypedEntry : public StorageEntry<cpp_type_t<DT>> {
public:
    using value_type = cpp_type_t<DT>;

    /// The writable twin for cs::init::custom: the same geometry as the member
    /// wrapper, over the root's storage on this entry's device.
    Initializer<DT, Rank> make_initializer() const {
        Initializer<DT, Rank> init;
        point(init);
        return init;
    }

protected:
    /// `home` non-null: this entry is that Variable's SHADOW (no shadow of its own).
    TypedEntry(std::string name, IOKind k, const std::array<DimRecord, Rank>& dims,
               bool can_source, bool can_target, Variable* home = nullptr)
        : StorageEntry<value_type>{std::move(name), k, DT, Rank,
                                   std::vector<DimRecord>(dims.begin(), dims.end()), can_source,
                                   can_target, home} {
        if (home == nullptr) this->install_shadow();
    }

    template <class W>
    void point(W& w) const {
        if (this->is_ragged())
            throw std::logic_error("naga::cospan: '" + std::string{this->name()} + "' is ragged; declare it with "
                                   "RaggedInput / RaggedOutput (a dense Input / Output cannot see per-index extents)");
        const IOEntry::Geometry g = this->geometry();
        auto* base = static_cast<value_type*>(this->root()->storage_on(this->placement()));
        this->probe_data_ = base != nullptr ? base + g.offset : nullptr;
        this->probe_offs_ = nullptr;
        IOEntry::write_wrapper(w, g, base != nullptr ? base + g.offset : nullptr);
    }

    /// A ragged wrapper: the root's data, the block count, and the device copies of the index
    /// arrays; a DENSE entry is accepted too (offsets null: block i is row i of the rectangle).
    template <class W>
    void point_ragged(W& w) const {
        const IOEntry* r    = this->root();
        auto*          base = static_cast<value_type*>(this->root()->storage_on(this->placement()));
        std::array<std::size_t, 2>           ext{1, 1};
        std::array<const std::uint64_t*, 2> per{nullptr, nullptr};
        if (this->is_ragged()) {
            const Placement pl = this->placement();
            // under a leading range [a, b) the device arrays are the root's, shifted by a (the
            // wrapper addresses blocks relative to offsets[0]); the data starts at the root's offsets[a].
            const auto        lr = this->leading_range();
            const std::size_t a  = lr ? lr->first : 0;
            // An INTERNAL root of a replaying ChunkedSystem holds the whole run's device arrays; its current chunk
            // starts at meta_shift_. The data base stays the start of the held storage (block(i) is relative to
            // offsets[0]), so only the metadata pointers move.
            const std::size_t shift = a + r->meta_shift();
            for (int d = 1; d < Rank; ++d) {
                if (!this->dim(d).per) {
                    ext[static_cast<std::size_t>(d - 1)] = this->dim(d).extent;
                } else {
                    const std::uint64_t* p = r->ragged_ptr_on(pl, d);
                    per[static_cast<std::size_t>(d - 1)] = p != nullptr ? p + shift : nullptr;
                }
            }
            const std::uint64_t* offs = r->ragged_ptr_on(pl, 0);
            if (offs != nullptr) offs += shift;
            if (base != nullptr && a > 0) base += r->block_offset_at(a);
            this->probe_data_ = base;
            this->probe_offs_ = offs;
            IOEntry::write_ragged(w, this->extents()[0], base, offs, ext, per);
        } else {
            const IOEntry::Geometry g = this->geometry();
            if (g.stride != contiguous_strides(g.shape))
                throw std::logic_error("naga::cospan: '" + std::string{this->name()} +
                                       "': a ragged wrapper on a dense port needs contiguous storage (no sliced view)");
            for (int d = 1; d < Rank; ++d) ext[static_cast<std::size_t>(d - 1)] = g.shape[static_cast<std::size_t>(d)];
            this->probe_data_ = base != nullptr ? base + g.offset : nullptr;
            this->probe_offs_ = nullptr;
            IOEntry::write_ragged(w, g.shape[0], base != nullptr ? base + g.offset : nullptr, nullptr, ext, per);
        }
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

/// RaggedInputEntry holds RaggedInput<DT,Rank>*. Same flags as InputEntry.
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
class RaggedInputEntry final : public TypedEntry<DT, Rank> {
public:
    RaggedInputEntry(std::string name, const std::array<DimRecord, Rank>& dims, RaggedInput<DT, Rank>* member)
        : TypedEntry<DT, Rank>{std::move(name), IOKind::input, dims, false, true}, member_{member} {
        this->mark_ragged_port();
    }
    void point_wrapper() override {
        if (member_ != nullptr) this->point_ragged(*member_);
    }

private:
    RaggedInput<DT, Rank>* member_;
};

/// RaggedOutputEntry holds RaggedOutput<DT,Rank>*. Same flags as OutputEntry.
template <DType DT, int Rank>
class RaggedOutputEntry final : public TypedEntry<DT, Rank> {
public:
    RaggedOutputEntry(std::string name, const std::array<DimRecord, Rank>& dims, RaggedOutput<DT, Rank>* member)
        : TypedEntry<DT, Rank>{std::move(name), IOKind::output, dims, true, false}, member_{member} {
        this->mark_ragged_port();
    }
    void point_wrapper() override {
        if (member_ != nullptr) this->point_ragged(*member_);
    }

private:
    RaggedOutput<DT, Rank>* member_;
};

/// A System-level alias (promotion or tie alias): no member wrapper. Its
/// flags are given (a promotion snapshots its root's, a tie alias is a fresh
/// root); as a root it owns storage like any other.
template <DType DT, int Rank>
class AliasEntry final : public TypedEntry<DT, Rank> {
public:
    AliasEntry(std::string name, IOKind k, const std::array<DimRecord, Rank>& dims,
               bool can_source, bool can_target, Variable* home = nullptr)
        : TypedEntry<DT, Rank>{std::move(name), k, dims, can_source, can_target, home} {}

    void point_wrapper() override {}
};

namespace alias_factory {

template <DType DT, int R>
std::unique_ptr<IOEntry> make_typed(std::string name, IOKind k, const std::vector<DimRecord>& dims,
                                    bool can_source, bool can_target, Variable* home) {
    std::array<DimRecord, R> a{};
    for (int d = 0; d < R; ++d) a[static_cast<std::size_t>(d)] = dims[static_cast<std::size_t>(d)];
    return std::make_unique<AliasEntry<DT, R>>(std::move(name), k, a, can_source, can_target, home);
}

template <DType DT>
std::unique_ptr<IOEntry> make_ranked(std::string name, IOKind k, const std::vector<DimRecord>& dims,
                                     bool can_source, bool can_target, Variable* home) {
    switch (dims.size()) {
        case 1: return make_typed<DT, 1>(std::move(name), k, dims, can_source, can_target, home);
        case 2: return make_typed<DT, 2>(std::move(name), k, dims, can_source, can_target, home);
        default: return make_typed<DT, 3>(std::move(name), k, dims, can_source, can_target, home);
    }
}

template <std::size_t... I>
std::unique_ptr<IOEntry> make_impl(DType dt, std::string name, IOKind k,
                                   const std::vector<DimRecord>& dims, bool can_source,
                                   bool can_target, Variable* home, std::index_sequence<I...>) {
    std::unique_ptr<IOEntry> out;
    (void)((static_cast<std::size_t>(dt) == I
                ? (out = make_ranked<static_cast<DType>(I)>(std::move(name), k, dims, can_source, can_target, home),
                   true)
                : false) ||
           ...);
    return out;
}

}  // namespace alias_factory

/// A type-erased AliasEntry of the given dtype (atomic-ness preserved) and
/// rank (dims.size()): System's promote / tie_* build aliases from resolved
/// entries, whose dtype and rank are runtime values.
inline std::unique_ptr<IOEntry> make_alias_entry(DType dt, std::string name, IOKind k,
                                                 const std::vector<DimRecord>& dims, bool can_source,
                                                 bool can_target, Variable* home = nullptr) {
    if (dims.empty() || dims.size() > 3)
        throw std::invalid_argument("naga::cospan: '" + name + "': rank must be 1..3");
    return alias_factory::make_impl(dt, std::move(name), k, dims, can_source, can_target, home,
                                    std::make_index_sequence<13>{});
}

/// The shadow of this port's Variable: inert dims (its shape is ADOPTED from the authority port).
inline void IOEntry::install_shadow() {
    std::vector<DimRecord> inert = dims_;
    for (DimRecord& r : inert)
        if (r.how != DimRecord::How::fixed) r.how = DimRecord::How::from_input;  // a static shape stays static
    var_->shadow = make_alias_entry(dtype_, name_, io_kind_, inert, can_be_source_, can_be_target_, var_.get());
    root_        = var_->shadow.get();
}

/// INTERNAL / TEST-ONLY. The structural and placement mutators are System's
/// and the Component's (friendship); this stays the sanctioned way for the
/// ENTRY-LEVEL smoke (smoke.cpp) and the Component smoke to drive them without
/// a System. System-level tests use System's own operations, never this.
struct EntryAccess {
    static void merge(IOEntry& into, IOEntry& from) { IOEntry::merge(into, from); }
    static void register_placement(IOEntry& e, const Placement& p) { e.register_placement(p); }
    static void set_device(IOEntry& e, const sycl::device& d) { e.device_ = d; }
    static void set_storage_kind(IOEntry& e, sk::kind k) { e.storage_kind_ = k; }
    static void set_view(IOEntry& e, std::optional<SubarrayRef> v) { e.view_ = std::move(v); }
    /// What the owning Component does for FromInputDim / FromInput dims.
    static void set_resolved_extent(IOEntry& e, int d, std::size_t extent) {
        if (d < 0 || d >= e.rank_) throw std::out_of_range("naga::cospan: dim index >= rank");
        e.dims_[static_cast<std::size_t>(d)].extent = extent;
    }
};

/// INTERNAL. The one runtime-changeable view: the framework (a chunked system) and
/// the tests set the leading range [a, b) a connected input sees of its root along dim 0.
struct LeadingRangeAccess {
    static void set(IOEntry& e, std::size_t a, std::size_t b) { e.set_leading_range(a, b); }
    static void clear(IOEntry& e) { e.clear_leading_range(); }
    /// The replaying ChunkedSystem: the range alone, without re-resolving the peer's shape (it applies the
    /// cached chunk state) and without validate_connection or re-pointing.
    static void set_no_refresh(IOEntry& e, std::size_t a, std::size_t b) { e.set_leading_range(a, b, false); }
    static void clear_no_refresh(IOEntry& e) { e.clear_leading_range(false); }
};

}  // namespace detail

namespace {
// The variable a (possibly retired) one was folded into.
inline detail::Variable* surviving(detail::Variable* v) {
    while (v != nullptr && v->merged_into) v = v->merged_into.get();
    return v;
}
}  // namespace
inline detail::IOEntry* AliasSrc::shadow() const {
    detail::Variable* v = surviving(var.get());
    return v ? v->shadow.get() : nullptr;
}
inline const detail::IOEntry* AliasSrc::live_port() const {
    detail::Variable* v = surviving(var.get());
    if (v == nullptr) return nullptr;
    for (const detail::IOEntry* p : v->ports)
        if (p == port) return port;
    return v->shadow.get();
}

}  // namespace naga::cospan