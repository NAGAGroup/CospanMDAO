# Spec 03 — Topology semantics & optimization marks

Part of the CospanMDAO specification. Binding contract; declarations
only. Aliasing mechanics: 02.C. Structural APIs: 01.E. `declare_input`'s
coverage option: §2.4.

## 1. The connection spectrum

Three kinds of "dst sees src's data", and they are NOT one mechanism:

- **identity** — `connect(src, dst)`: dst joins the root's set as a peer
  reading the whole root. Zero-copy.
- **whole-input slice** — a rectangular region of the root whose flat
  size equals dst's: dst is a peer whose `view()` is that region, in the
  ROOT's rank; the wrapper gets `root.data + offset` and the root's
  strides (01.A). Zero-copy. This is what the stride metadata exists for.
  Why (for the whole spectrum): ledger §2.3 connection.
- **computed** — everything else: a region of the source into a REGION
  of the input (several sources may feed one input, regions disjoint),
  or an index map (a gather is not affine; a view cannot express it).
  These produce a MAPPING SYSTEM (§2): a copy by necessity, made of
  ordinary components so it is enumerable, fusable and differentiable
  like anything else.

```cpp
namespace naga::cospan {

/// What `connect` accepts beyond identity. Two are VIEWS, the rest are
/// COMPUTED.
struct Slice        { SubarrayRef src; };                     // view
struct SliceInto    { SubarrayRef src; SubarrayRef dst; };     // computed
struct IndexMap     { std::string_view map; };                 // computed, whole dst
struct IndexMapInto { std::string_view map; SubarrayRef dst; };// computed, region
using ConnectionMap = std::variant<Slice, SliceInto, IndexMap, IndexMapInto>;

class System {   // ...continued from 01.E
public:
    void connect(std::string_view src, std::string_view dst);                // identity
    void connect(std::string_view src, std::string_view dst, ConnectionMap); // §1/§2
};

} // namespace naga::cospan
```

`map` names an index-map-valued I/O: any integer-dtype entry (an output
produced upstream — neighbour lists, CSR columns — or an unconnected
input root set from the host). The ledger's "index-map-valued I/O"
pattern is exactly this convention; no dedicated type exists.

## 2. The mapping System

### 2.1 Lifecycle

- **First computed connection on an input `dst`**: the owning System
  creates a hidden mapping System named `<dst>__map`, records it in its
  `input -> mapping system` table, and `add_subsystem`s it (its `setup()`
  is empty — components arrive afterwards; `add_subsystem` on a System
  whose setup has run is legal). Then adds the first mapping component.
- **Each subsequent computed connection on `dst`**: check the new `dst`
  region against every recorded region — overlap is an error naming
  both connections — then add another mapping component.
- **Post-setup pass** (end of the owning System's `run_setup`, after its
  `setup()` hook returns): for each mapping System — `tie_outputs` over all mapping
  components' outputs (each is full-dst-shaped) with
  `TieOutputsPolicy::allow_plain`, invoked BY THE FRAMEWORK: it holds
  the disjointness proof, so it is the party entitled to the override
  (the trust model, used correctly — ledger P4); connect the tie alias to `dst`
  (identity); run the coverage check (§2.4).

### 2.2 Execution position

The mapping System must run after its sources and before whoever reads
`dst`. The post-setup pass INSERTS it into the owner's child list
IMMEDIATELY BEFORE the child that contains the consumer (the first path
segment of `dst`), not at the end. Sources produced by later siblings
are a feedback loop the model already has; solvers own that,
and the insertion rule does not change it. Why: ledger §2.2 sequence.

### 2.3 Mapping components

```cpp
namespace naga::cospan::kernels {

/// SliceInto: the component's input is a PEER WITH A VIEW of the source
/// root (an ordinary slice connection — the read side reuses the view
/// machinery); the output is full-dst-shaped and only the region is
/// written. Launch range = region extents.
template <DType DT, int Rank>
struct RegionCopy {
    struct Inputs  { Input<DT, Rank>  src; };       // already the sliced view
    struct Outputs { Output<DT, Rank> out; };
    sycl::id<Rank> dst_offset;                       // POD config
    void operator()(sycl::id<Rank> i, Inputs a, Outputs o) const
        { o.out[i + dst_offset] = a.src[i]; }
};

/// IndexMap / IndexMapInto: dst[offset + i] = src[map[i]].
/// Launch range = map's range.
template <DType DT, DType IDX, int Rank>
struct GatherInto {
    struct Inputs  { Input<DT, 1> src; Input<IDX, Rank> map; };
    struct Outputs { Output<DT, Rank> out; };
    sycl::id<Rank> dst_offset;
    void operator()(sycl::id<Rank> i, Inputs a, Outputs o) const
        { o.out[i + dst_offset] = a.src[a.map[i]]; }
};

/// Scatter — the additive reverse map, out[map[i]] += in[i]; output
/// atomic BY TYPE (the general-case floor). Not a connection kind — a
/// provided kernel users compose.
template <DType DT, DType IDX, int Rank>
struct Scatter {
    struct Inputs  { Input<DT, Rank> in; Input<IDX, Rank> map; };
    struct Outputs { Output<to_atomic_v<DT>, 1> out; };
    void operator()(sycl::id<Rank> i, Inputs a, Outputs o) const
        { o.out[a.map[i]] += a.in[i]; }
};

} // namespace naga::cospan::kernels
```

Properties that fall out: the mapping System is a real, enumerable
subsystem (observable, debuggable); every mapping component is an
ordinary KernelBody; and because the components write disjoint regions,
the mapping System is PROVABLY legal as a parallel-mode FusedSystem — a
later optimization needing no spec change.

### 2.4 Coverage

Uncovered elements of `dst` hold whatever the tied storage holds — which
may be exactly what the owning component initialized, deliberately. So
coverage is checked BY DEFAULT and opted out per input at declaration
(why: ledger P8, §2.3 initialization):

```cpp
enum class Coverage : uint8_t { required, optional };
/// Every declare_input form (01.B/01.D) carries a trailing option
/// (default `required`):
template <DType DT, int Rank, DimSpec... Specs>
    requires(sizeof...(Specs) == Rank)
void declare_input(Input<DT, Rank>& m, std::string name, Specs... specs,
                   Lifetime life = Lifetime::durable,
                   Coverage cov = Coverage::required);
```

The post-setup pass reads `dst`'s coverage option (the tie alias
inherits it) and, when `required`, errors if the recorded regions do not
cover `dst` entirely ("input `x` is 40% covered by computed
connections"). `optional` skips the check — the escape hatch for the
initialized-remainder case, and a legitimate performance choice.

## 3. View composition

Views compose by offset addition with containment: connecting from a
name whose peer carries view `B`, with a `Slice{A}`, yields
`{B.offset + A.offset, A.extents}` and requires `A ⊆ B` (error naming
both; why: ledger §2.3 connection · P1). Promoting a peer that carries a view produces an alias that
COPIES the view — the alias presents what the promoted entry sees. All
views are in the root's rank; there is no rank reinterpretation through
a view (rank is authoritative, 01.B).

## 4. Cross-level ties

Nothing new: members resolve from the caller's namespace at any depth,
and a member whose root is already a tie alias (a child-level tie) is a
legal member when that alias is unconnected — the merge makes the child
alias a peer of the new root, and the `tie_*` exception marks it like
any named member. Ties layer upward without special rules.

## 5. Optimization marks

```cpp
namespace naga::cospan {

/// Per-element or scalar bounds/scaling; host-side vectors sized to the
/// marked region's flat size (scalar forms broadcast).
struct Bounds  { std::optional<std::vector<double>> lower, upper; };
struct Scaling { std::vector<double> ref{1.0}, ref0{0.0}; };   ///< semantics: with the optimizers

class System {   // ...continued
public:
    /// MARKS RIDE ON THE ROOT: marking any name marks the data. Recorded
    /// on the root's IOEntry (01.B is_design_var/…), enumerated by
    /// Model. `region` restricts a mark to a subarray (root rank) via
    /// the same view machinery; no index maps on marks. Why: ledger
    /// §2.4 design variables, constraints, objectives.

    /// Root must be an INPUT root that is not connected — a free input;
    /// the optimizer is its producer. Dynamically shaped DVs are
    /// ill-formed (01.B). A connected input errors: "connected inputs
    /// are computed, not designed".
    void add_design_var(std::string_view path, Bounds b = {}, Scaling s = {},
                        std::optional<SubarrayRef> region = {});

    /// Root must be an OUTPUT root. `equals` and bounds are exclusive.
    void add_constraint(std::string_view path, Bounds b = {},
                        std::optional<std::vector<double>> equals = {},
                        Scaling s = {}, std::optional<SubarrayRef> region = {});

    /// Root must be an OUTPUT root; the marked region must have flat
    /// size 1 — `region` is how one component of a vector becomes the
    /// objective.
    void add_objective(std::string_view path, Scaling s = {},
                       std::optional<SubarrayRef> region = {});
};

/// Model-level enumeration — the contract an optimizer reads.
/// Order = declaration order, depth-first.
class Model {   // ...continued from 02.B
public:
    struct MarkRecord {
        const detail::IOEntry*     root;
        std::optional<SubarrayRef> region;
        Bounds                     bounds;
        std::optional<std::vector<double>> equals;   // constraints only
        Scaling                    scaling;
    };
    std::span<const MarkRecord> design_vars() const;
    std::span<const MarkRecord> constraints() const;
    std::span<const MarkRecord> objectives()  const;
};

} // namespace naga::cospan
```

## 6. Samples

```cpp
// Two producers assembling one field: rows 0..N/2 from a, the rest from b.
connect("a.field", "solver.field", SliceInto{subarray<2>({0,0},{N/2,M}),
                                             subarray<2>({0,0},{N/2,M})});
connect("b.field", "solver.field", SliceInto{subarray<2>({0,0},{N/2,M}),
                                             subarray<2>({N/2,0},{N/2,M})});
// -> hidden System "solver.field__map" with two RegionCopy components,
//    inserted before "solver"; full coverage verified.

// Neighbour gather: LBVH produces the map, the operator consumes gathered values.
connect("lbvh.neighbors", "op.gathered", IndexMap{"lbvh.neighbor_ids"});

// Sellar's marks.
add_design_var("mda.z", Bounds{.lower = {{-10, 0}}, .upper = {{10, 10}}});
add_design_var("mda.x", Bounds{.lower = {{0}},      .upper = {{10}}});
add_objective ("obj.f");
add_constraint("obj.g1", Bounds{.upper = {{0}}});
add_constraint("obj.g2", Bounds{.upper = {{0}}});
```
