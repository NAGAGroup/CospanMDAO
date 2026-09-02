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
/// The read happens in the declaring Component's own run_pre_compute
/// (01.D §5): a device-to-host copy into the DimRecord's dim_scratch, a
/// custom operation applying the resize, one wait on that operation's
/// event. Only entries using FromInput ever pay it. Orchestration is
/// Component-local, always; no producer knows its consumers.
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
// registers the member wrapper's address. The Component maintains TWO
// resolution maps over the same entries: declared name -> IOEntry (path
// resolution, enumeration, connections) and member address -> IOEntry
// (member-reference APIs: reshape and friends). Wrappers themselves carry
// no identity (01.A). Rank = the member's Rank; the spec count must equal
// it, at compile time.
// -----------------------------------------------------------------------------

template <DType DT, int Rank, DimSpec... Specs>
    requires(sizeof...(Specs) == Rank)
void declare_input(Input<DT, Rank>& member, std::string name,
                   Specs... specs, Lifetime life = Lifetime::durable,
                   Coverage cov = Coverage::required);   // spec 03 §2.4

/// FromSourceDim is ill-formed for outputs.
template <DType DT, int Rank, DimSpec... Specs>
    requires(sizeof...(Specs) == Rank)
void declare_output(Output<DT, Rank>& member, std::string name,
                    Specs... specs, Lifetime life = Lifetime::durable);

/// Whole-shape overloads accepting the from_input_array marker alone.
template <DType DT, int Rank>
void declare_input(Input<DT, Rank>& member, std::string name,
                   /* from_input_array marker */, Lifetime life = Lifetime::durable,
                   Coverage cov = Coverage::required);
template <DType DT, int Rank>
void declare_output(Output<DT, Rank>& member, std::string name,
                    /* from_input_array marker */, Lifetime life = Lifetime::durable);

/// The escape hatch — and the ONLY resize spelling. Resolves the entry by
/// the member's registered ADDRESS (the member-address map). Legal ONLY
/// inside the pre_forward() hook; it uses the same underlying setter as
/// automatic derivation, and the hook runs after the automatic pass, so
/// it wins by ordering.
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
    sycl::range<3> dim_scratch; ///< always-valid device-to-host copy
                            ///  destination for from_input reads (01.C);
                            ///  unused for other specs — uniform by
                            ///  design, no detection logic anywhere
};

/// IOEntry — the metadata record behind one declared I/O. One instance
/// per declaration, owned by the declaring Component/System in a
/// name-keyed map. The virtual base is the RTTI anchor: typed retrieval
/// (get_input<DT,Rank>) dynamic_casts the entry; hierarchy resolution
/// (04 §3: get_subsystem chaining) is a chain of ordinary lookups ending
/// here.
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

    // ---- the aliasing set (mechanics: spec 02.C) -------------------------
    /// Every entry belongs to one set with one ROOT (data owner and, for
    /// outputs, producer); peers are names for and consumers of the
    /// root's data. There are no source/target links.
    IOEntry* root();                 ///< self iff is_root()
    bool     is_root() const;
    bool     can_be_source() const;  ///< construction-time (02.C §2)
    bool     can_be_target() const;
    /// Device placement and per-device root storage — device(),
    /// storage_on(), storage_devices() — are declared in 02.C.

    // ---- capabilities ---------------------------------------------------

    /// Automatic shape update: applies the DimRecords. Invoked by the
    /// OWNING COMPONENT in run_pre_compute, BEFORE connection validation.
    /// FromInput dims are resolved there first (D2H into dim_scratch,
    /// custom-op resize, one wait — 01.D §5).
    virtual void update_shape() = 0;

    /// Resize (automatic derivation and user reshape share this path).
    /// REALLOCATION POLICY: growing past the current allocation
    /// reallocates through the dynamic chain (pool if capable -> OS ->
    /// OOM); shrinking keeps the allocation — capacity is retained,
    /// extents describe the logical shape.
    virtual void reshape(sycl::range<3> new_extents) = 0;

    /// Connection validation — INPUT PEERS ONLY, against root():
    /// FLAT-SIZE equality (rank and per-dim extents are free to differ
    /// across a connection); slice connections compare the view's size.
    /// Static pairs once, in prepare_exec; dynamic pairs every forward
    /// after update_shape.
    virtual void validate_connection() const = 0;

    /// Allocation, in prepare_exec: ROOTS ONLY, one allocation per device
    /// the set spans, each from that device's substrate (02.C §2).
    /// Durable ∧ fully-fixed → the arena (its bytes are what
    /// get_exec_info counted); everything else → the dynamic chain at
    /// first need.
    virtual void allocate(Substrates&) = 0;

    /// Write {data pointer, shape, stride} into the registered member
    /// Input/Output wrapper — the pointer being
    /// root()->storage_on(device()) (+ view offset). This is what makes
    /// the user's member variable usable; called at the end of
    /// prepare_exec and after every resize.
    virtual void point_wrapper() = 0;
};

/// Typed leaves. InputEntry holds Input<DT,Rank>* to the member variable;
/// OutputEntry holds Output<DT,Rank>* plus the owning data handle
/// (slabkit::unique_device_ptr<std::byte[]> + capacity, 02.A/02.C).
/// DIFFERENTIABLE dtypes additionally reserve a gradient allocation slot
/// (same dtype and shape, populated under phase-7 policy); the slot's
/// declarations do not exist for data-tier dtypes.
template <DType DT, int Rank> class InputEntry  final : public IOEntry { /* ... */ };
template <DType DT, int Rank> class OutputEntry final : public IOEntry { /* ... */ };

} // namespace detail

// -----------------------------------------------------------------------------
// Enumerability — THE observation guarantee.
// -----------------------------------------------------------------------------
/// Every Component and System SHALL expose const iteration over its
/// declared entries, split by kind, OpenMDAO-style:
///     for (const detail::IOEntry& e : x.get_inputs())  { ... }
///     for (const detail::IOEntry& e : x.get_outputs()) { ... }
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
