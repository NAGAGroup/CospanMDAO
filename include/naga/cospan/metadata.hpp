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

/// This dimension = dim `dim` of the upstream connection's shape. Inputs only.
struct FromSourceDim { uint8_t dim; };

/// This dimension = dim `dim` of sibling input `input`'s shape.
struct FromInputDim { std::string_view input; uint8_t dim; };

/// This dimension's extent = the VALUE of element `index` (flat) of sibling
/// input `input`, which SHALL have an integer value dtype.
struct FromInput { std::string_view input; std::size_t index; };

/// RAGGED DIMENSIONS (D140). A dimension d >= 1 of a ragged port may have a different extent for every
/// index of dimension 0: PerIndex gives the extents as a host array at declaration (its length must equal
/// dimension 0's extent), FromInputValues takes them from the VALUES of an integer sibling input of rank 1
/// [N] (column 0) or rank 2 [N, k] (column < k). Negative values are an error. Neither is allowed on dim 0.
struct PerIndex { std::vector<std::size_t> values; };

inline PerIndex per_index(std::vector<std::size_t> values) { return PerIndex{std::move(values)}; }

struct FromInputValues { std::string_view input; std::size_t column = 0; };

template <class T>
concept DimSpec = std::integral<T> || std::same_as<T, FromSourceDim> ||
                  std::same_as<T, FromInputDim> || std::same_as<T, FromInput> ||
                  std::same_as<T, PerIndex> || std::same_as<T, FromInputValues>;

/// The whole-shape value-driven marker: on a rank-R declaration it expands to
/// FromInput{input,0}..{input,R-1}. Accepted alone by the declaration
/// overloads; expansion against the member's Rank happens in the declaration.
struct FromInputArray { std::string_view input; };

/// Sugar. The two shape helpers return arrays of specs (spread them with
/// std::apply); from_input_array returns the marker.
template <int Rank>
constexpr std::array<FromSourceDim, Rank> from_source_shape() {
    std::array<FromSourceDim, Rank> a{};
    for (int d = 0; d < Rank; ++d) a[d] = FromSourceDim{static_cast<uint8_t>(d)};
    return a;
}

template <int Rank>
constexpr std::array<FromInputDim, Rank> from_input_shape(std::string_view input) {
    std::array<FromInputDim, Rank> a{};
    for (int d = 0; d < Rank; ++d) a[d] = FromInputDim{input, static_cast<uint8_t>(d)};
    return a;
}

constexpr FromInputArray from_input_array(std::string_view input) { return {input}; }

/// The rank-only declaration marker (D139 part 2): `declare_input(w, "x", cs::tied)`. The entry
/// names a dtype and a rank (from its wrapper type) and no shape: its shape is whatever its ROOT's
/// shape is, and the root of a tied entry is a tie alias (System::tie_inputs / tie_outputs), which
/// owns the shape.
struct Tied {};
inline constexpr Tied tied{};

/// The per-dimension specs of cs::alias::dims (D140: tie aliases take the shape rules of ports). An integer
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

/// One dimension of an alias::dims shape, type-erased.
struct AliasDim {
    enum class How : uint8_t { fixed, per_index_host, output_value, output_values, output_dim };
    How                      how    = How::fixed;
    std::size_t              extent = 1;   ///< fixed
    std::vector<std::size_t> values;       ///< per_index_host
    std::string              path;         ///< output_*: the upstream output (input = true: the System's input)
    std::size_t              index  = 0;   ///< output_value: the element; output_values: the column; output_dim: the dim
    bool                     input  = false;  ///< the path names an INPUT entry (input_value / input_values / input_dim)
    detail::IOEntry*         src    = nullptr;  ///< output_*: resolved by the System at setup
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

/// How a tie alias gets its shape (D139 part 2); build one with the cs::alias factories.
struct AliasShape {
    enum class Kind : uint8_t { fixed, from_connection, from_output_shape, from_output_values, dims, from_input_shape };
    Kind                       kind = Kind::fixed;
    int                        rank = 0;            ///< fixed / dims: how many extents / specs were given
    std::array<std::size_t, 3> extents{1, 1, 1};    ///< fixed
    std::string                path;                ///< from_output_* / from_input_shape: an entry, in the declaring System's namespace
    std::vector<AliasDim>      dim_specs;           ///< dims: one per dimension
};

namespace alias {
/// The per-dimension form (D140): one spec per dim, count = the alias's Rank. A spec is an integer (fixed),
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
/// tie_inputs only: the shape of whatever output is connected into the alias.
inline AliasShape from_connection() {
    AliasShape s;
    s.kind = AliasShape::Kind::from_connection;
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

/// The initial value of a free input root (r01 §4.4, D138): type-erased, made
/// by the cs::init factories (init.hpp). apply() runs on a root that has just
/// been allocated from nothing; the caller refreshes the other placements.
class InitSpec {
public:
    using Fn = std::function<void(detail::IOEntry& root, sk::queue& q)>;

    InitSpec() = default;
    explicit InitSpec(Fn f) : fn_{std::move(f)} {}

    void apply(detail::IOEntry& root, sk::queue& q) const {
        if (fn_) fn_(root, q);
    }
    explicit operator bool() const noexcept { return static_cast<bool>(fn_); }

private:
    Fn fn_;
};

namespace detail {

/// Per-dimension storage form of a declared DimSpec.
struct DimRecord {
    /// from_root: the entry's shape is its root's shape (cs::tied, D139 part 2).
    /// per_index_host / per_index_from_input (D140): dimension d >= 1 has one extent per index of dim 0.
    enum class How : uint8_t { fixed, from_source_dim, from_input_dim, from_input, from_root, per_index_host,
                               per_index_from_input };
    How         how = How::fixed;
    std::size_t extent = 0;  ///< fixed: declared; derived: last resolved (0 = unknown)
    std::string input;       ///< from_input_*: sibling input name (owned)
    uint8_t     dim = 0;     ///< from_source_dim / from_input_dim
    std::size_t value_index = 0;  ///< from_input; per_index_from_input: the column
    /// The resolved per-index extents of a dimension d >= 1 (D140); EMPTY means the dimension is scalar
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
    } else if constexpr (std::same_as<S, FromSourceDim>) {
        r.how = DimRecord::How::from_source_dim;
        r.dim = s.dim;
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

/// cs::tied against a member's Rank.
template <int Rank>
std::array<DimRecord, Rank> tied_dims() {
    std::array<DimRecord, Rank> a{};
    for (int d = 0; d < Rank; ++d) a[d].how = DimRecord::How::from_root;
    return a;
}

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

    virtual ~IOEntry() = default;
    IOEntry(const IOEntry&)            = delete;
    IOEntry& operator=(const IOEntry&) = delete;

    // ---- identity & classification ---------------------------------------
    std::string_view name() const noexcept { return name_; }
    /// The entry's dotted path from the root System: its declarer's path plus its name, or the name alone at the root.
    std::string full_path() const {
        return (declarer_ == nullptr || declarer_->path().empty())
                   ? std::string{name()}
                   : declarer_->path() + "." + std::string{name()};
    }
    IOKind           kind() const noexcept { return io_kind_; }
    DType            dtype() const noexcept { return dtype_; }  ///< atomic-ness preserved
    int              rank() const noexcept { return rank_; }    ///< AUTHORITATIVE (§3.1)

    // ---- shape -------------------------------------------------------------
    /// A promoted alias mirrors its member: its shape is read live from it (D139 part 2).
    sycl::range<3> extents() const {
        const auto& e = shape_ref();
        return sycl::range<3>{e[0], e[1], e[2]};
    }
    /// Element step in the OWNING allocation (a sliced peer reports its root's).
    sycl::range<3> stride() const {
        const Geometry g = geometry();
        return sycl::range<3>{g.stride[0], g.stride[1], g.stride[2]};
    }
    /// A ragged entry (D140): the TOTAL number of elements (block_offsets().back()). A promoted alias
    /// mirrors its member.
    std::size_t flat_size() const noexcept {
        if (mirror_of_ != nullptr) return mirror_of_->flat_size();
        if (ragged_) return static_cast<std::size_t>(offsets_.back());
        return extents_[0] * extents_[1] * extents_[2];
    }

    // ---- ragged dimensions (D140) ------------------------------------------
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
    /// The runtime leading range [a, b) this connected input sees of its root (D140 B1); none = whole.
    std::optional<std::pair<std::size_t, std::size_t>> leading_range() const { return lead_; }
    /// The raw per-index array of dimension d (empty: scalar). Internal.
    const std::vector<std::size_t>& per_index_raw(int d) const { return dims_[static_cast<std::size_t>(d)].per_index; }
    const DimRecord& dim(int d) const {
        if (d < 0 || d >= rank_) throw std::out_of_range("naga::cospan: dim index >= rank");
        return dims_[static_cast<std::size_t>(d)];
    }
    bool is_dynamic() const noexcept {
        for (const DimRecord& r : dims_)
            if (r.how != DimRecord::How::fixed) return true;
        return false;
    }
    /// Declared with cs::tied: every dimension is "from the root" (a promotion of such an entry
    /// carries its dims, so it is rank-only too).
    bool is_rank_only() const noexcept {
        for (const DimRecord& r : dims_)
            if (r.how != DimRecord::How::from_root) return false;
        return !dims_.empty();
    }
    /// The tie alias this entry was tied into (members, and their promotions); null if untied.
    const IOEntry* tie_alias() const noexcept { return tie_alias_; }
    /// A tie alias: how its shape is made, and the upstream output a from_output_* shape reads.
    const std::optional<AliasShape>& alias_shape() const noexcept { return alias_shape_; }
    const IOEntry*                   alias_source() const noexcept { return alias_src_; }
    /// Every upstream output a tie alias's shape reads (path, entry): one for from_output_*, one per
    /// output-reading dim of an alias::dims shape.
    std::vector<std::pair<std::string_view, const IOEntry*>> alias_sources() const {
        std::vector<std::pair<std::string_view, const IOEntry*>> out;
        if (!alias_shape_) return out;
        if (alias_src_ != nullptr) out.emplace_back(alias_shape_->path, alias_src_);
        for (const AliasDim& d : alias_shape_->dim_specs)
            if (d.src != nullptr) out.emplace_back(d.path, d.src);
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
    IOEntry*       root() noexcept { return root_; }
    const IOEntry* root() const noexcept { return root_; }
    bool           is_root() const noexcept { return root_ == this; }
    bool           owns_data() const noexcept { return is_root(); }
    /// Root only; empty otherwise. The root itself is not in its own list.
    std::span<IOEntry* const> peers() const noexcept {
        return is_root() ? std::span<IOEntry* const>{peers_} : std::span<IOEntry* const>{};
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
    // ---- registration metadata (ROOTS; recorded by System::resolve, r01 §4.4, D138) ----
    /// Every registration in the root's alias set is transient: the root is allocated by the
    /// components that touch it and released after its last user. Never arena-eligible.
    bool     transient() const noexcept { return root_->transient_; }
    /// A transient root's storage is dropped: from resolve until its first user allocates it,
    /// and again after its last user. Host access to a released root throws.
    bool released() const noexcept { return root_->released_; }
    /// Path of the topmost transient subsystem whose registration made the root transient; empty otherwise.
    std::string_view transient_owner() const noexcept { return root_->transient_by_; }
    /// The winning initial value of a free input root (highest level; none = zeros).
    const std::optional<InitSpec>& init_spec() const noexcept { return root_->init_spec_; }
    /// The last Component, in depth-first registration order, that touches the root.
    const Subsystem* last_user() const noexcept { return root_->last_user_; }
    /// The OWNER: the first Component, in depth-first registration order, that touches the root.
    /// It alone allocates the root and applies its init, in its pre_compute (D138 amendment).
    const Subsystem* owner() const noexcept { return root_->owner_; }

    /// Roots: the memory substrates allocation draws on (bound at prepare_exec).
    void bind_substrates(Substrates& s) noexcept { substrates_ = &s; }
    /// PRESENT (D139): the root's storage state. True once its owner has run the allocation path
    /// (even for a size-0 root, whose storage is empty) until a transient root is released. Host
    /// views of a root that is not present throw.
    bool present() const noexcept { return root_->present_; }
    /// Whether the ROOT of this entry's set is present.
    bool has_storage() const noexcept { return root_->present_; }
    /// An initial value waiting to be applied by the owner: set when a free input root is
    /// allocated from nothing, and by Model::init_input. It stays set while the root has no elements.
    bool init_pending() const noexcept { return root_->init_pending_; }
    void clear_init_pending() noexcept { root_->init_pending_ = false; }
    /// Model::init_input (D139): the model's initial value replaces any declared one, wins over
    /// them, cannot be overwritten by resolve(), and is applied at the owner's next run.
    void set_model_init(InitSpec spec) {
        root_->init_spec_    = std::move(spec);
        root_->model_init_   = true;
        root_->init_pending_ = true;
    }

    /// THE ONE ALLOCATION ROUTINE of the root of this entry (route: arena, pool, OS), run by the
    /// owner in its pre_compute. When the root is not present it allocates (an empty storage for a
    /// size-0 root), marks it present and, for a free input root, marks its init pending; returns
    /// whether it did.
    bool allocate_if_absent(sk::queue& q) {
        IOEntry& r = *root_;
        if (r.present_) return false;
        if (r.substrates_ == nullptr)
            throw std::logic_error("naga::cospan: '" + r.full_path() + "' cannot be allocated before setup() completes");
        r.allocate(q, *r.substrates_);
        if (r.io_kind_ == IOKind::input) r.init_pending_ = true;
        return true;
    }
    /// D140 B3: this root is HELD by its ChunkedSystem (internal root kept across the chunks).
    bool held() const noexcept { return root_->held_; }
    /// How many times the root's storage was allocated or grew past its capacity (a test hook).
    std::size_t allocation_count() const noexcept { return root_->allocations_; }
    /// The ROOT's write epoch: a counter that advances whenever the root's data may have changed (a
    /// component that writes it has been submitted, an init was applied, a host write, an allocation, a
    /// release or a size change). It is what a host-copy cache compares against; it never decreases.
    /// Not bumped by sync_other_placements (same logical content) nor by ragged-metadata uploads (shape,
    /// not data). Single-threaded submission is assumed, as everywhere in CoSpan.
    std::uint64_t write_epoch() const noexcept { return root_->write_epoch_; }
    /// Advance the root's write epoch (always the canonical root, never a peer, view or alias member).
    void bump_write() noexcept { ++root_->write_epoch_; }
    /// D140 B2: the path of the ChunkedSystem this root is internal to; empty otherwise.
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

    /// Arena route: a root, fully fixed, outside every transient subtree, in
    /// device-kind storage (arenas are device-kind).
    bool arena_eligible() const noexcept {
        return is_root() && !is_dynamic() && !transient_ && storage_kind_ == sk::kind::device;
    }
    /// The bytes ONE device's arena must hold for this root; 0 when the root
    /// takes another route. get_exec_info counts this once per device.
    std::size_t static_bytes() const noexcept {
        return arena_eligible() ? flat_size() * dtype_size(dtype_) : 0;
    }

    // ---- capabilities --------------------------------------------------------

    /// Automatic shape update: applies the DimRecords. FromSourceDim is
    /// resolved here; FromInputDim / FromInput extents are resolved by the
    /// owning Component first (set_resolved_extent) and applied here.
    virtual void update_shape() = 0;

    /// Resize. Growing past capacity reallocates through the root's allocator
    /// route; shrinking keeps capacity (usm_vector's contract).
    virtual void reshape(sycl::range<3> new_extents) = 0;

    /// INPUT PEERS ONLY, against root(): flat-size equality (the view's size
    /// for a slice, which must also lie inside the root). Throws.
    virtual void validate_connection() const = 0;

    /// ROOTS ONLY (no-op on peers): one usm_vector per placement in
    /// storage_placements(), each from that placement's device substrate
    /// (host-kind placements take the OS host route on the device).
    virtual void allocate(sk::queue& q, Substrates& substrates) = 0;

    /// Write {data pointer, shape, stride} into the registered member wrapper.
    virtual void point_wrapper() = 0;

    /// ROOTS ONLY (D140): bring every placement's device copies of the ragged index arrays (the offsets and
    /// one array per per-index dimension) up to date with the host description; drops them when the root
    /// is not ragged. A no-op when nothing changed or the root is not present.
    virtual void upload_ragged(sk::queue& q) = 0;

    /// ROOTS ONLY (D140): the device copy of the offsets (which == 0) or of the per-index array of
    /// dimension `which` (1..2) on placement `p`; null when there is none.
    virtual const std::uint64_t* ragged_ptr_on(const Placement& p, int which) const = 0;

    /// The root's producer copies primary -> every other placement's storage
    /// (a routed copy; the ordinary copy rule).
    virtual void sync_other_placements(sk::queue& q) = 0;


    /// ROOTS ONLY (no-op on peers): drop the storage of a transient root (§4.4). The CALLER
    /// waits the stream first (synchronous frees, D135): a releasing Component waits once
    /// for its whole batch.
    virtual void release_storage() = 0;

protected:
    IOEntry(std::string name, IOKind k, DType dt, int rank, std::vector<DimRecord> dims,
            bool can_source, bool can_target)
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
            if (r.how == DimRecord::How::from_source_dim && k == IOKind::output)
                throw std::invalid_argument("naga::cospan: '" + name_ +
                                            "': FromSourceDim is ill-formed for an output");
            if ((r.how == DimRecord::How::from_source_dim ||
                 r.how == DimRecord::How::from_input_dim) &&
                r.dim > 2)
                throw std::invalid_argument("naga::cospan: '" + name_ +
                                            "': a dimension index is out of range");
            extents_[static_cast<std::size_t>(d)] = r.extent;  // derived dims start at 0
        }
    }

    virtual void* raw_storage_on(const Placement& p) const = 0;

    /// The only writer of wrapper fields (friendship, r01 §2.1).
    template <class W, class P>
    static void write_wrapper(W& w, const Geometry& g, P* data) {
        w.shape_  = make_range<W::rank>(g.shape);
        w.stride_ = make_range<W::rank>(g.stride);
        w.data_   = data;
    }

    /// The only writer of ragged wrapper fields (D140).
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

    std::array<std::size_t, 3> extents_{1, 1, 1};  ///< 1-padded; a per-index dim reports 0 (D140)
    // ---- D140: the applied ragged description of THIS entry (host) ----
    bool                       ragged_ = false;          ///< any dim >= 1 is per-index
    std::vector<std::uint64_t> offsets_;                 ///< ragged: N + 1 block offsets
    std::array<std::vector<std::size_t>, 3> applied_per_;  ///< the per-index arrays last applied
    std::uint64_t              ragged_version_ = 0;      ///< bumped whenever the ragged description changes
    bool                       ragged_port_ = false;     ///< declared with RaggedInput / RaggedOutput
    // ---- D140 B3 (roots): kept across the chunks of a ChunkedSystem ----
    bool                       held_          = false;   ///< set by the ChunkedSystem: no release by a last user
    std::size_t                reserve_elems_ = 0;       ///< held: the capacity (elements) allocate() takes
    std::size_t                allocations_   = 0;       ///< storage allocations of this root (allocate, growth past capacity)
    std::uint64_t              write_epoch_   = 0;       ///< roots: see write_epoch()
    /// Inside a ChunkedSystem the framework sets dim 0 of every port of every inside component to the
    /// current chunk length (N in the full-range pass, b - a per chunk), whatever rule declared it.
    std::optional<std::size_t> dim0_override_;
    std::size_t                dim0_start_ = 0;  ///< with dim0_override_: the first index of the chunk (host per-index slices)
    std::vector<DimRecord>     dims_;
    std::vector<Placement>     placements_;        ///< roots; primary first
    std::optional<sycl::device> device_;
    sk::kind                   storage_kind_ = sk::kind::device;
    bool                       released_ = false;     ///< transient roots: no storage right now (starts true)
    bool                       present_  = false;     ///< roots: allocated by the owner (D139); cleared by release
    std::string                transient_by_;         ///< transient roots: topmost transient subsystem's path

private:
    /// The four structural operations and placement are System's (r01 §4.2);
    /// EntryAccess remains the test-only stand-in for the entry-level smoke.
    friend class naga::cospan::System;
    friend class naga::cospan::ComponentBase;
    friend struct EntryAccess;
    friend struct LeadingRangeAccess;
    friend class naga::cospan::ChunkedSystem;

    /// D140 B1, framework only (through LeadingRangeAccess). A NON-root input peer sees rows / blocks
    /// [a, b) of its root along dim 0. Dense root: its view becomes the SubarrayRef with dim-0 offset a
    /// and extent b - a (composed with a setup-time Slice, which is kept and restored by clear); the
    /// peer's extents follow through update_shape. Ragged root: N = b - a, the host per-index arrays are
    /// the root's [a, b) slice, and point_ragged shifts the device pointers by a. Re-runs the peer's shape
    /// update and validate_connection and re-points its wrapper, so the next forward uses the range.
    void set_leading_range(std::size_t a, std::size_t b) {
        if (is_root() || mirror_of_ != nullptr)
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
        refresh_after_range();
    }
    /// Back to the whole range (and the setup-time view, if any).
    void clear_leading_range() {
        if (!lead_) return;
        view_ = pre_lead_view_;
        pre_lead_view_.reset();
        lead_.reset();
        refresh_after_range();
    }
    void refresh_after_range() {
        if (io_kind_ == IOKind::input) {
            update_shape();
            validate_connection();
        }  // an output peer's shape is its component's to resolve (from its inputs) at the next forward
        if (root_->present_) point_wrapper();
    }

    /// ChunkedSystem (D140 B2), on an alias ROOT at its level: take the full resolved shape of `m`, the
    /// inside entry that used to be the root (dense: its extents; ragged: N and the per-index arrays),
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

    /// merge(into, from): `from` and its peers become peers of `into`.
    static void merge(IOEntry& into, IOEntry& from) {
        if (!into.is_root() || !from.is_root())
            throw std::logic_error("naga::cospan: merge takes two set roots");
        if (&into == &from) return;
        into.peers_.reserve(into.peers_.size() + from.peers_.size() + 1);
        for (IOEntry* p : from.peers_) {
            p->root_ = &into;
            into.peers_.push_back(p);
        }
        from.peers_.clear();
        from.root_ = &into;
        into.peers_.push_back(&from);
    }

    std::string name_;
    IOKind      io_kind_;
    DType       dtype_;
    int         rank_;

    IOEntry*              root_ = this;
    std::vector<IOEntry*> peers_;  // meaningful iff root_ == this
    bool                  can_be_source_;
    bool                  can_be_target_;
    std::optional<SubarrayRef> view_;
    std::optional<std::pair<std::size_t, std::size_t>> lead_;  ///< D140 B1: the runtime leading range
    std::optional<SubarrayRef> pre_lead_view_;                 ///< the setup-time view saved while lead_ is set
    bool                  transient_ = false;     ///< roots, set by System::resolve
    const Subsystem*      declarer_  = nullptr;

    std::optional<MarkRecord> design_var_, constraint_, objective_;

    // ---- D139 part 2: ties own the shape; promotions mirror ----
    const IOEntry*            mirror_of_ = nullptr;  ///< a promoted alias: the entry it was made from
    IOEntry*                  tie_alias_ = nullptr;  ///< members of a tie (and their set): the tie alias
    std::optional<AliasShape> alias_shape_;          ///< a tie alias: where its shape comes from
    IOEntry*                  alias_src_ = nullptr;  ///< a tie alias: the upstream output of a from_output_* shape
    std::vector<IOEntry*>     tie_members_;          ///< a tie alias: the member entries, as given
    bool                      alias_checked_ = false;  ///< a connected tie_inputs alias: its AliasShape was checked

    /// An initial value declared on THIS entry (ComponentBase::init_input / System::init_input),
    /// with the depth of the declaring subsystem (root System = 0). resolve() picks, per root,
    /// the declaration of the smallest depth.
    struct InitDecl {
        InitSpec    spec;
        std::size_t depth = 0;
        std::string by;  ///< the declaring subsystem's path, for errors
    };
    std::vector<InitDecl>     init_decls_;
    std::optional<InitSpec>   init_spec_;        ///< roots, set by System::resolve or Model::init_input
    bool                      model_init_   = false;  ///< roots: init_spec_ came from Model::init_input (resolve keeps it)
    bool                      init_pending_ = false;  ///< roots: the init is to be applied at the owner's next run
    const Subsystem*          last_user_ = nullptr;  ///< roots, set by System::resolve
    const Subsystem*          owner_     = nullptr;  ///< roots, set by System::resolve: the first toucher
    bool                      chunk_rerooted_ = false;  ///< D140 B2: a root made at a ChunkedSystem's level (its owner allocates it)
    std::string               chunk_internal_by_;       ///< D140 B2: roots internal to the named ChunkedSystem (forced transient)
    Substrates*               substrates_ = nullptr; ///< roots, bound at prepare_exec
};

// -----------------------------------------------------------------------------
// Storage layer: everything that depends on the element type only
// -----------------------------------------------------------------------------

template <class T>
class StorageEntry : public IOEntry {
public:
    void update_shape() override {
        std::array<std::size_t, 3> next = extents_;
        for (int d = 0; d < rank(); ++d) {
            DimRecord& r = dims_[static_cast<std::size_t>(d)];
            if (d == 0 && dim0_override_) {  // inside a ChunkedSystem (D140 B3): dim 0 IS the chunk length,
                next[0] = *dim0_override_;   // whatever rule declared it
                continue;
            }
            switch (r.how) {
                case DimRecord::How::fixed: break;
                case DimRecord::How::from_source_dim:
                    r.extent = source_extent(r.dim);
                    if (d >= 1) {
                        r.per_index = source_per_index(r.dim);
                        r.per       = source_per_flag(r.dim);
                    }
                    break;
                case DimRecord::How::from_root:
                    r.extent = root_extent(static_cast<uint8_t>(d));
                    if (d >= 1) {
                        r.per_index = root_per_index(static_cast<uint8_t>(d));
                        r.per       = root_per_flag(static_cast<uint8_t>(d));
                    }
                    break;
                case DimRecord::How::from_input_dim:    // resolved by the Component (which also copies a
                case DimRecord::How::from_input:        // per-index source dimension's array)
                case DimRecord::How::per_index_host:    // fixed at declaration
                case DimRecord::How::per_index_from_input: break;  // resolved by the Component
            }
            next[static_cast<std::size_t>(d)] = (d >= 1 && r.per) ? 0 : r.extent;
        }

        // ---- D140: the ragged description. Every per-index array must have one entry per index of dim 0.
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
        resize_storage();
    }

    void reshape(sycl::range<3> r) override {
        if (ragged_)
            throw std::logic_error("naga::cospan: '" + full_path() + "' is ragged: its shape follows its per-index dimensions and cannot be reshaped");
        for (int d = rank(); d < 3; ++d)
            if (r[d] != 1)
                throw std::invalid_argument("naga::cospan: '" + std::string{name()} +
                                            "': extents past the rank must be 1");
        extents_ = {r[0], r[1], r[2]};
        resize_storage();
    }

    void validate_connection() const override {
        if (kind() != IOKind::input || is_root()) return;
        const IOEntry& r        = *root();
        if (r.is_ragged() || is_ragged()) {  // D140: the per-index extents must match dim by dim, and the totals
            if (r.is_ragged() != is_ragged())
                throw std::invalid_argument("naga::cospan: input '" + std::string{name()} + "' is " +
                                            (is_ragged() ? "ragged" : "dense") + " but is connected to '" +
                                            std::string{r.name()} + "', which is " + (r.is_ragged() ? "ragged" : "dense"));
            if (view())
                throw std::invalid_argument("naga::cospan: '" + std::string{name()} + "' sees a subarray of the ragged '" +
                                            std::string{r.name()} + "'; ragged roots cannot be sliced");
            const auto        lr = leading_range();  // D140 B1: compare against the root's [a, b)
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

    void allocate(sk::queue& q, Substrates& subs) override {
        if (!is_root()) return;
        if (placements_.empty())
            register_placement(Placement{device_ ? *device_ : q.get_device(), storage_kind_});
        subs_ = &subs;
        for (const Placement& p : placements_) {
            if (find_slot(p) != nullptr) continue;
            // A size-0 root gets an empty vector (usm_vector(0, ...) allocates nothing). A HELD root (D140 B3)
            // takes its reserved capacity: per chunk it is resized within it, which never reallocates.
            const std::size_t n = held_ ? reserve_elems_ : flat_size();
            slots_.push_back(std::make_unique<Slot>(p, sk::usm_vector<T>(n, route(q, subs, p, n))));
            ++allocations_;
            diag::count_allocations();
            bump_write();  // a new slot: whatever a cache held is gone
        }
        released_ = false;
        present_  = true;
        upload_ragged(q);
    }

    void upload_ragged(sk::queue& q) override {
        if (!is_root() || !present_ || subs_ == nullptr) return;
        for (auto& s : slots_) {
            if (s->version == ragged_version_) continue;
            diag::count_ragged_uploads();
            // The ragged index arrays follow the DENSE storage rules: a vector is created once (empty), kept, and
            // rewritten in place. from_host into the existing capacity is a memory task the in-order stream orders
            // after any kernel still reading the old values, so it needs no wait. Only GROWTH past the capacity
            // releases the old storage, and only then are its readers waited for (once per slot).
            bool waited = false;
            const auto upload = [&](std::optional<sk::usm_vector<std::uint64_t>>& v, std::span<const std::uint64_t> src) {
                if (!v) v.emplace(0, route_as<std::uint64_t>(q, *subs_, s->pl));  // empty: nothing allocated yet
                if (src.size() > v->capacity()) {                                  // growth, as for a dense root
                    if (v->capacity() > 0) {                                       // existing storage is released
                        if (!waited) {
                            diag::count_queue_waits();
                            q.wait();
                            waited = true;
                        }
                        diag::count_frees();
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
        Slot* s = find_slot(p);
        if (s == nullptr) return nullptr;
        if (which == 0) return s->offs ? s->offs->data() : nullptr;
        if (which < 1 || which > 2) return nullptr;
        const auto& a = s->per[static_cast<std::size_t>(which - 1)];
        return a ? a->data() : nullptr;
    }

    void release_storage() override {
        if (!is_root() || !present_) return;
        // Synchronous frees (D135): the stream's kernels may still read these roots and a usm_vector waits
        // only for its own tasks, so the CALLER has waited the stream (once per releasing Component's
        // batch); the vectors free here.
        diag::count_frees(slots_.size());
        for (const auto& s : slots_) {  // the metadata vectors that hold real storage go with the slot
            if (s->offs && s->offs->capacity() > 0) diag::count_frees();
            for (const auto& a : s->per)
                if (a && a->capacity() > 0) diag::count_frees();
        }
        if (!slots_.empty()) bump_write();
        host_cache_  = std::vector<T>{};  // free the cached copy with the storage
        cache_epoch_ = ~std::uint64_t{0};
        slots_.clear();
        released_ = true;
        present_  = false;
    }

    void sync_other_placements(sk::queue&) override {
        if (!is_root()) throw std::logic_error("naga::cospan: sync_other_placements is for roots");
        if (slots_.size() < 2 || flat_size() == 0) return;
        Slot& primary = primary_slot();
        for (auto& s : slots_)
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
        Slot& s = primary_slot();
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
                 bool can_source, bool can_target)
        : IOEntry{std::move(name), k, dt, rank, std::move(dims), can_source, can_target} {}

    void* raw_storage_on(const Placement& p) const override {
        Slot* s = find_slot(p);
        if (s == nullptr) {
            if (released_) throw_released();
            throw std::logic_error("naga::cospan: '" + full_path() +
                                   "' has no storage on that placement");
        }
        return s->data.data();
    }

    /// Host access to a released root: the error names the root and the transient subsystem
    /// whose registration made it transient (D57, D138).
    [[noreturn]] void throw_released() const {
        throw std::logic_error("naga::cospan: '" + full_path() + "' belongs to transient subsystem '" +
                               transient_by_ + "' and its storage is released; it exists only while that "
                               "subsystem's components run");
    }

private:
    struct Slot {
        Slot(Placement p, sk::usm_vector<T> v) : pl{std::move(p)}, data{std::move(v)} {}
        Placement         pl;
        sk::usm_vector<T> data;
        // D140: the device copies of the ragged index arrays, allocated with this slot's placement and
        // released with it. `version` is the ragged_version_ they were built for (0 = none).
        std::optional<sk::usm_vector<std::uint64_t>>               offs;
        std::array<std::optional<sk::usm_vector<std::uint64_t>>, 2> per;  // dims 1 and 2
        std::uint64_t                                              version = 0;
    };

    /// Resize the root's storage to flat_size(); growing waits the stream first (synchronous frees, D135).
    void resize_storage() {
        if (!is_root()) return;
        if (!slots_.empty() && flat_size() > slots_.front()->data.capacity()) {
            diag::count_queue_waits();
            slots_.front()->data.get_queue().wait();
            ++allocations_;  // a usm_vector resize within its capacity never reallocates; this one does
            diag::count_allocations();
        }
        const bool size_changes = !slots_.empty() && slots_.front()->data.size() != flat_size();
        for (auto& s : slots_) s->data.resize(flat_size());
        if (size_changes) bump_write();
    }

    /// A per-index array of the SOURCE's dimension d (empty: scalar), for FromSourceDim.
    std::vector<std::size_t> source_per_index(uint8_t d) const {
        if (is_root()) return {};  // source_extent throws for it before this is reached
        if (view()) return {};
        const std::vector<std::size_t>& full = root()->per_index_raw(d);
        if (const auto lr = leading_range(); lr && source_per_flag(d)) {  // D140 B1: the root's [a, b) slice
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
        if (is_root() || view()) return false;
        const DimRecord& rd = root()->dim(d);
        return rd.per && rd.per_index.size() == root()->extents()[0];
    }
    bool root_per_flag(uint8_t d) const { return source_per_flag(d); }

    /// A tied entry's per-index array of dimension d: its root's. Under a runtime leading range (D140 B1/B2:
    /// an inside peer of a boundary root in a ChunkedSystem) it is the root's [a, b) slice, exactly as
    /// source_per_index gives it.
    std::vector<std::size_t> root_per_index(uint8_t d) const {
        if (is_root()) return {};
        if (view()) return {};
        return source_per_index(d);
    }

    Slot* find_slot(const Placement& p) const {
        for (const auto& s : slots_)
            if (s->pl == p) return s.get();
        return nullptr;
    }

    Slot& primary_slot() const {
        if (released_) throw_released();
        const auto no_data = [&] {
            return std::logic_error("naga::cospan: '" + full_path() +
                                    "' has no data until its owner component runs (it is allocated in the owner's "
                                    "first forward)");
        };
        if (!is_root() || !present_ || placements_.empty()) throw no_data();
        Slot* s = find_slot(placements_.front());
        if (s == nullptr) throw no_data();
        return *s;
    }

    /// A tied entry's dimension d: its root's (or, under a view, the view's) extent. The root of a
    /// tied entry is a tie alias (or the output a tie_inputs alias is connected to), whose shape
    /// the root's owner resolved before this entry's Component runs.
    std::size_t root_extent(uint8_t d) const {
        if (is_root())
            throw std::logic_error("naga::cospan: '" + full_path() + "' is tied (cs::tied) but is not in a tie");
        if (const auto v = view()) return v->extents[d];
        if (const auto lr = leading_range(); lr && d == 0) return lr->second - lr->first;  // under a range (D140 B1)
        return root()->extents()[d];
    }

    std::size_t source_extent(uint8_t d) const {
        if (is_root())
            throw std::logic_error("naga::cospan: source-derived dimension on the unconnected input '" +
                                   std::string{name()} + "'");
        if (const auto v = view()) {
            if (d >= v->rank) throw std::out_of_range("naga::cospan: FromSourceDim past the source's rank");
            return v->extents[d];
        }
        if (d >= root()->rank())
            throw std::out_of_range("naga::cospan: FromSourceDim past the source's rank");
        if (const auto lr = leading_range(); lr && d == 0) return lr->second - lr->first;  // a ragged root (D140 B1)
        return root()->extents()[d];
    }

    /// r01 §4.4 as amended by D3: a host-kind placement takes the OS route
    /// with its device choosing the context; a device-kind placement of a
    /// fully fixed root outside a transient subtree takes the arena (never
    /// falling back: an arena is a budget); everything else the pool if the
    /// device has one, else the OS.
    sk::allocator<T> route(sk::queue& q, Substrates& subs, const Placement& p, std::size_t elems) const {
        if (p.kind != sk::kind::device) return sk::allocator<T>{q, p.kind, p.device};
        // An empty root (D139) allocates nothing, so it needs no arena or pool: a fixed size-0 root is
        // arena-eligible but its device has no arena (the arena is sized from the non-empty roots).
        if (elems == 0) return sk::allocator<T>{q, sk::kind::device, p.device};
        const sycl::device& d = p.device;
        const auto it = subs.find(d);
        if (arena_eligible()) {
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

    std::vector<std::unique_ptr<Slot>> slots_;  // parallel to placements_ once allocated
    mutable std::vector<T>             host_cache_;                      // the primary's host copy, valid at cache_epoch_
    mutable std::uint64_t              cache_epoch_ = ~std::uint64_t{0};  // the write epoch host_cache_ was read at
    Substrates*                        subs_ = nullptr;  // bound at allocate (the ragged arrays route through it)
};

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
    TypedEntry(std::string name, IOKind k, const std::array<DimRecord, Rank>& dims,
               bool can_source, bool can_target)
        : StorageEntry<value_type>{std::move(name), k, DT, Rank,
                                   std::vector<DimRecord>(dims.begin(), dims.end()), can_source,
                                   can_target} {}

    template <class W>
    void point(W& w) const {
        if (this->is_ragged())
            throw std::logic_error("naga::cospan: '" + std::string{this->name()} + "' is ragged; declare it with "
                                   "RaggedInput / RaggedOutput (a dense Input / Output cannot see per-index extents)");
        const IOEntry::Geometry g = this->geometry();
        auto* base = static_cast<value_type*>(this->root()->storage_on(this->placement()));
        IOEntry::write_wrapper(w, g, base != nullptr ? base + g.offset : nullptr);
    }

    /// A ragged wrapper (D140): the root's data, the block count, and the device copies of the index
    /// arrays; a DENSE entry is accepted too (offsets null: block i is row i of the rectangle).
    template <class W>
    void point_ragged(W& w) const {
        const IOEntry* r    = this->root();
        auto*          base = static_cast<value_type*>(this->root()->storage_on(this->placement()));
        std::array<std::size_t, 2>           ext{1, 1};
        std::array<const std::uint64_t*, 2> per{nullptr, nullptr};
        if (this->is_ragged()) {
            const Placement pl = this->placement();
            // D140 B1: under a leading range [a, b) the device arrays are the root's, shifted by a (the
            // wrapper addresses blocks relative to offsets[0]); the data starts at the root's offsets[a].
            const auto        lr = this->leading_range();
            const std::size_t a  = lr ? lr->first : 0;
            for (int d = 1; d < Rank; ++d) {
                if (!this->dim(d).per) {
                    ext[static_cast<std::size_t>(d - 1)] = this->dim(d).extent;
                } else {
                    const std::uint64_t* p = r->ragged_ptr_on(pl, d);
                    per[static_cast<std::size_t>(d - 1)] = p != nullptr ? p + a : nullptr;
                }
            }
            const std::uint64_t* offs = r->ragged_ptr_on(pl, 0);
            if (offs != nullptr) offs += a;
            if (base != nullptr && a > 0) base += r->block_offset_at(a);
            IOEntry::write_ragged(w, this->extents()[0], base, offs, ext, per);
        } else {
            const IOEntry::Geometry g = this->geometry();
            if (g.stride != contiguous_strides(g.shape))
                throw std::logic_error("naga::cospan: '" + std::string{this->name()} +
                                       "': a ragged wrapper on a dense port needs contiguous storage (no sliced view)");
            for (int d = 1; d < Rank; ++d) ext[static_cast<std::size_t>(d - 1)] = g.shape[static_cast<std::size_t>(d)];
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

/// RaggedInputEntry holds RaggedInput<DT,Rank>* (D140). Same flags as InputEntry.
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

/// RaggedOutputEntry holds RaggedOutput<DT,Rank>* (D140). Same flags as OutputEntry.
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
               bool can_source, bool can_target)
        : TypedEntry<DT, Rank>{std::move(name), k, dims, can_source, can_target} {}

    void point_wrapper() override {}
};

namespace alias_factory {

template <DType DT, int R>
std::unique_ptr<IOEntry> make_typed(std::string name, IOKind k, const std::vector<DimRecord>& dims,
                                    bool can_source, bool can_target) {
    std::array<DimRecord, R> a{};
    for (int d = 0; d < R; ++d) a[static_cast<std::size_t>(d)] = dims[static_cast<std::size_t>(d)];
    return std::make_unique<AliasEntry<DT, R>>(std::move(name), k, a, can_source, can_target);
}

template <DType DT>
std::unique_ptr<IOEntry> make_ranked(std::string name, IOKind k, const std::vector<DimRecord>& dims,
                                     bool can_source, bool can_target) {
    switch (dims.size()) {
        case 1: return make_typed<DT, 1>(std::move(name), k, dims, can_source, can_target);
        case 2: return make_typed<DT, 2>(std::move(name), k, dims, can_source, can_target);
        default: return make_typed<DT, 3>(std::move(name), k, dims, can_source, can_target);
    }
}

template <std::size_t... I>
std::unique_ptr<IOEntry> make_impl(DType dt, std::string name, IOKind k,
                                   const std::vector<DimRecord>& dims, bool can_source,
                                   bool can_target, std::index_sequence<I...>) {
    std::unique_ptr<IOEntry> out;
    (void)((static_cast<std::size_t>(dt) == I
                ? (out = make_ranked<static_cast<DType>(I)>(std::move(name), k, dims, can_source, can_target),
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
                                                 bool can_target) {
    if (dims.empty() || dims.size() > 3)
        throw std::invalid_argument("naga::cospan: '" + name + "': rank must be 1..3");
    return alias_factory::make_impl(dt, std::move(name), k, dims, can_source, can_target,
                                    std::make_index_sequence<13>{});
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

/// INTERNAL (D140 B1). The one runtime-changeable view: the framework (the chunked system, later) and
/// the tests set the leading range [a, b) a connected input sees of its root along dim 0.
struct LeadingRangeAccess {
    static void set(IOEntry& e, std::size_t a, std::size_t b) { e.set_leading_range(a, b); }
    static void clear(IOEntry& e) { e.clear_leading_range(); }
};

}  // namespace detail
}  // namespace naga::cospan
