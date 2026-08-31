# Spec 01.B — I/O metadata, shape derivation, declaration API

Part of the phase-1 spec (`docs/spec/01-components-io/`). Binding contract;
declarations only. Wrapper types: 01.A. Kernel machinery: 01.C.

## Rank is authoritative

A declaration's rank is type-level (the wrapper's `Rank` parameter) and
semantic: rank-3 `{2,1,1}` and rank-1 `{2}` are different declarations with
different API surfaces (`id<3>` vs `id<1>`). The 1-padded `sycl::range<3>`
in metadata exists for exactly one purpose — uniform flat-size arithmetic
with no per-rank conditionals — and SHALL NOT leak into any API surface,
comparison, or user-visible semantics.

## Validity timing

Self-side validity — spec count equals Rank, dim indices sane against the
declaring entry — is checked at construction (compile-time where
expressible). Source-side validity — the named source exists, is an input,
has compatible rank/size, integer value dtype where required — CANNOT be
known at construction (the source may itself be dynamic): it is checked at
EVERY shape resolution and throws on violation.

```cpp
namespace naga::cospan {

enum class IOKind   : uint8_t { input, output };
enum class Lifetime : uint8_t { durable, transient };
    ///< transient = valid within the owning System's forward invocation;
    ///  storage comes from the dynamic chain, released at scope end.

// -----------------------------------------------------------------------------
// Dimension specification — DimSpec is a CONCEPT, not a type.
// A dimension is specified by anything integral (a fixed extent) or by a
// derivation type. Derivation sources are INPUTS ONLY: an output's shape
// is the result of computation, so it can never determine one.
// -----------------------------------------------------------------------------

/// This dimension = dim `dim` of the upstream connection's shape.
/// Inputs only (outputs have no upstream).
struct FromSourceDim { uint8_t dim; };

/// This dimension = dim `dim` of sibling input `input`'s shape.
struct FromInputDim  { std::string_view input; uint8_t dim; };

/// This dimension's extent = the VALUE of element `index` (flat) of
/// sibling input `input`. The source SHALL have an integer value dtype.
/// The read happens in the declaring Component's own pre-forward
/// codepath: a temporary host variable, one device-to-host copy
/// (synchronizing on the producer's queue), used, discarded. Nothing
/// persistent exists anywhere; only entries using FromInput ever pay the
/// copy. Orchestration is Component-local, always.
struct FromInput { std::string_view input; size_t index; };

template <class T>
concept DimSpec = std::integral<T>
               || std::same_as<T, FromSourceDim>
               || std::same_as<T, FromInputDim>
               || std::same_as<T, FromInput>;

/// Sugar helpers.
///   from_source_shape<R>()        — FromSourceDim{0} .. {R-1}
///   from_input_shape<R>(input)    — FromInputDim{input,0} .. {input,R-1}
///   from_input_array(input)       — whole-shape value-driven marker:
///     on a rank-R declaration expands to FromInput{input,0}..{input,R-1}.
///     ASSUMES the source is rank-1 with flat size == R; checked at every
///     shape resolution per the validity-timing rule. Accepted ALONE by
///     the declaration overloads; expansion against the member's Rank
///     happens inside the declaration.
template <int Rank> constexpr auto from_source_shape();
template <int Rank> constexpr auto from_input_shape(std::string_view input);
constexpr auto from_input_array(std::string_view input);

// -----------------------------------------------------------------------------
// Declaration API (declared on the Component/System base — full base:
// 01.D). Declaration performs ZERO allocation; it records metadata and
// registers the member wrapper's address for sync_wrapper. Rank = the
// member's Rank; the spec count must equal it, at compile time.
// -----------------------------------------------------------------------------

template <DType DT, int Rank, DimSpec... Specs>
    requires(sizeof...(Specs) == Rank)
void declare_input(Input<DT, Rank>& member, std::string name,
                   Specs... specs, Lifetime life = Lifetime::durable);

/// FromSourceDim is ill-formed for outputs.
template <DType DT, int Rank, DimSpec... Specs>
    requires(sizeof...(Specs) == Rank)
void declare_output(Output<DT, Rank>& member, std::string name,
                    Specs... specs, Lifetime life = Lifetime::durable);

/// Whole-shape overloads accepting the from_input_array marker alone.
template <DType DT, int Rank>
void declare_input(Input<DT, Rank>& member, std::string name,
                   /* from_input_array marker */, Lifetime life = Lifetime::durable);
template <DType DT, int Rank>
void declare_output(Output<DT, Rank>& member, std::string name,
                    /* from_input_array marker */, Lifetime life = Lifetime::durable);

/// The escape hatch — and the ONLY resize spelling. Resolves the entry by
/// the member's registered name. Legal ONLY during the pre-forward step;
/// it uses the same underlying setter as automatic derivation, and a
/// manual call runs after the automatic pass, so it wins by ordering.
/// (Users deriving shapes manually SHOULD declare fixed extents to avoid
/// a redundant automatic update.)
template <DType DT, int Rank>
void reshape(Input<DT, Rank>& member, sycl::range<Rank> shape);
template <DType DT, int Rank>
void reshape(Output<DT, Rank>& member, sycl::range<Rank> shape);

// -----------------------------------------------------------------------------
// Metadata records
// -----------------------------------------------------------------------------

namespace detail {

/// Per-dimension storage form of a declared DimSpec.
struct DimRecord {
    enum class How : uint8_t { fixed, from_source_dim, from_input_dim,
                               from_input };
    How      how;
    size_t   extent;        ///< fixed: declared; derived: last resolved
    std::string_view input; ///< from_input_*: sibling input name
    uint8_t  dim;           ///< from_source_dim / from_input_dim
    size_t   value_index;   ///< from_input
};

/// IOEntry — the metadata record behind one declared I/O. One instance
/// per declaration, owned by the declaring Component/System in a
/// name-keyed map. The virtual base is the RTTI anchor: typed retrieval
/// (get_input<DT,Rank>) dynamic_casts the entry; hierarchy resolution
/// (spec 04) is a chain of ordinary lookups ending here.
class IOEntry {
public:
    virtual ~IOEntry();

    // ---- identity & classification -------------------------------------
    std::string_view name() const;
    IOKind   kind() const;
    DType    dtype() const;          ///< atomic-ness preserved
    int      rank() const;           ///< AUTHORITATIVE (see banner)
    Lifetime lifetime() const;

    // ---- shape ----------------------------------------------------------
    sycl::range<3> extents() const;  ///< 1-padded; flat-size arithmetic ONLY
    sycl::range<3> stride() const;   ///< element step in the owning allocation
    size_t flat_size() const;
    const DimRecord& dim(int d) const;  ///< d < rank()
    bool is_dynamic() const;         ///< any dim not How::fixed

    // ---- marks (semantics: spec 03) -------------------------------------
    bool is_design_var() const;
    bool is_constraint() const;
    bool is_objective() const;
    /// CONSTRAINT: a design-var entry SHALL NOT be dynamically shaped.

    // ---- aliasing slots (semantics: spec 02/03) -------------------------
    IOEntry* source() const;         ///< non-null => aliased; data forwarded
    IOEntry* target() const;
    bool owns_data() const;          ///< source() == nullptr

    // ---- capabilities ---------------------------------------------------

    /// Automatic shape update: applies the DimRecords. Invoked by the
    /// OWNING COMPONENT in its pre-forward codepath, BEFORE connection
    /// validation. FromInput reads happen here via the ephemeral host
    /// temp described above.
    virtual void update_shape() = 0;

    /// Resize (automatic derivation and user reshape share this path).
    /// REALLOCATION POLICY: growing past the current allocation
    /// reallocates through the dynamic chain (pool if capable -> OS ->
    /// OOM); shrinking keeps the allocation — capacity is retained,
    /// extents describe the logical shape.
    virtual void reshape(sycl::range<3> new_extents) = 0;

    /// Connection validation — INPUTS ONLY, against the upstream entry:
    /// FLAT-SIZE equality (rank and per-dim extents are free to differ
    /// across a connection); subarray connections compare subarray sizes.
    /// Runs AFTER update_shape; re-runs every forward when either side
    /// is_dynamic().
    virtual void validate_connection() const = 0;

    /// Allocation. NO-OP when aliased (owns_data() == false). Durable,
    /// fully-fixed entries are bump-allocated at setup — their sizes are
    /// what the model's accumulation pass counts; everything else
    /// allocates via the dynamic chain at first need.
    virtual void allocate(/* pool handles — exact types: spec 02 */) = 0;

    /// Push current {data, shape, stride} into the registered member
    /// wrapper. The wrapper is stale until sync'd; the framework SHALL
    /// sync before any bind (spec 01.C) reads wrappers.
    virtual void sync_wrapper() = 0;
};

/// Typed leaves. InputEntry holds Input<DT,Rank>* to the member variable;
/// OutputEntry holds Output<DT,Rank>* plus the owning data handle (RAII
/// smart pointer from slabkit — exact type: spec 02). DIFFERENTIABLE
/// dtypes additionally reserve a gradient allocation slot (same dtype and
/// shape, populated under phase-8 policy); the slot's declarations do not
/// exist for data-tier dtypes.
template <DType DT, int Rank> class InputEntry  final : public IOEntry { /* ... */ };
template <DType DT, int Rank> class OutputEntry final : public IOEntry { /* ... */ };

} // namespace detail

// -----------------------------------------------------------------------------
// Enumerability — THE observation guarantee.
// -----------------------------------------------------------------------------
/// Every Component and System SHALL expose const iteration over its
/// declared entries:   for (const detail::IOEntry& e : x.io_entries())
/// uniformly through the hierarchy — System alias entries included.
/// Recording, visualization, connection checking, and generated
/// documentation build on this guarantee and on nothing else.

} // namespace naga::cospan
```

## Samples

```cpp
// The 25x100 -> 100x5x5 reshape from the design discussion:
Input<DType::f32, 3> reshaped;
declare_input(reshaped, "reshaped_input", FromSourceDim{1}, 5, 5);

// Whole-shape value-driven: "dims" is a rank-1 u64 input of size 3.
Input<DType::f32, 3> grid;
declare_input(grid, "grid", from_input_array("dims"));
```
