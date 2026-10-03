# CoSpan specification — r01

**What this is.** The CoSpan design. This file describes what the code
implements and nothing else. Notation: `sk::` is SlabKit's
namespace (`sk::context`, `sk::queue`, `sk::queue::group`;
`slabkit::usm_vector`, `DeviceArena`, `DevicePool`); SlabKit's own spec owns
their contracts.

---

## 1. Principles and needs

Binding contract. This section states needs, never solutions: how each need
is met is the business of §2–§11. Later sections cite it as "§1" — `P1`–`P9`
for principles, "§1 need 2.x" for needs.

### 1.1 Principles (the yardsticks solutions are judged by)

Sections cite these where a rule would otherwise look arbitrary.

- **P1 Legibility** — refuse an operation whose meaning would depend on
  knowledge not visible at the use site; what a name yields is what a
  reader at that site would expect.
- **P2 Adoption** — a subsystem written by someone else is placed,
  initialized, promoted, connected, wrapped and re-run without editing
  its source.
- **P3 Pay-for-use** — a model that does not use a capability pays
  nothing for it; no implicit synchronization, ever.
- **P4 Trust model** — the framework checks what it can prove; explicit
  overrides carry the rest, and the party holding the proof is the party
  entitled to override.
- **P5 Boundary** — nothing crosses the owner↔child boundary except the
  framework verbs; owners know nothing of children, children nothing of
  owners.
- **P6 Uniformity over detection** — every entry carries every
  mechanism's slot, used or not; no special-case logic anywhere.
- **P7 Users never classify their work** — no tags on kernels or
  components; the framework infers from signatures and declarations.
- **P8 Safe by default, explicit opt-out** — every check is on unless a
  user says otherwise, in code, at the site.
- **P9 RAII, never ordered destruction** — shared ownership wherever
  destruction order would otherwise matter.

### 1.2 The bare-minimum core (what makes everything else possible)

**Components:** the three map abstractions · general-case completeness
(atomics floor) · work outside device kernels · dynamic I/O sizing ·
derivatives with fallback partials · implicit solves · temporal
components · atomic element types (explicit).

**I/O & execution topology:** connection / tie / promote semantics ·
aliasing correctness · slice connections · marks · index-
map-valued I/O · run policy and lifetimes · initialization · host access ·
enumerability of I/O (the one hard requirement of every tool built over the
model).

**Systems:** the system base contract — arbitrary extension that
integrates without touching existing code (THE most important item) ·
parallel · fused · placement · temporal systems · solving implicit
systems from cycles.

**Cross-cutting:** chunked execution over bounded scratch.

Everything not in this section is a composite, served by helpers built on
it.

### 1.3 The two-tier rule

Every pattern axis carries two obligations:

- **Abstraction tier** — the common cases made easy (helpers, wrappers).
- **Possibility floor** — the general case guaranteed *expressible*,
  unaided by helpers.

A missing abstraction costs users convenience. A missing floor is a
designed-in corner. Both tiers are named for every axis below.

### 1.4 The needs

#### Need 2.1 — Maps (intra-component)

A component is one unit of work with inputs and outputs; intermediates
are their own components, composed in systems.

- **Abstractions:** bijective, injective and surjective map helpers
  (reductions, gathers, scatters, region copies, masked ranges). These
  three classes close the abstraction tier.
- **Possibility floor:** the **general map** — arbitrary mixed reads and
  writes in one kernel — guaranteed by atomic element types available on
  every I/O and by indexed access through the data wrappers.
- **Work outside device kernels:** existing user code, file I/O, callbacks, and
  native libraries (BLAS, solvers) must be usable as ordinary components,
  ordered with everything else. Floor: a host body (plain host C++ over
  host-visible I/O, ordered in the stream) for existing user code, file I/O and
  callbacks, and a custom-operation body that receives the native handle
  and raw device pointers for native libraries and does whatever it likes
  on the backend's stream.
- **Shape source:** static, or **data-dependent extent** — an upstream
  output's *shape or value* may size downstream I/O.
- **Write-safety:** atomicity lives in the element type; a kernel cannot
  forget it.
- **Group-level parallelism:** work-group collectives and local memory
  are opt-in per kernel, with wrapper ergonomics, never required.
- **User types on the device:** a type with invariants, a non-trivial
  constructor or virtual functions must be constructible where it lives
  and usable from kernels through a pointer, even though the type itself
  is not device-copyable.
- **Minimal device payload:** what a kernel captures is the data it
  computes on and nothing the framework needs for bookkeeping; host-only
  concerns never ride into a launch.

#### Need 2.2 — Structure (inter-component)

- **Abstractions:**
  - `sequence` — subsystems run in declaration order.
  - `parallel` — independent subsystems run concurrently.
  - `cycle` — a feedback loop solved to convergence by an attached solver.
  - `march` — iterate N times or until a budget; no convergence semantics.
  - `fuse` — several components executed as one launch.
  - **Placement** — any subsystem may run on a device other than its
    owner's, with no change to the subsystems around it.
- **Possibility floor:** custom system types. The extension point and the
  escape hatch are the same mechanism: any orchestration a modeler needs
  — gating on a predicate, running once, running N times, rerunning only
  when inputs changed — is a system type, written without touching the
  components under it or the systems above it.
- **Opacity law:** a subsystem is fully opaque to its owner; returning
  from a forward means *done* (solves solved, fusions fused, parallel
  children joined). Composition legality follows from opacity, not from
  a matrix.
- **Inline composition:** parallelizing, fusing or placing a group of
  components must not require writing a new class.

#### Need 2.3 — Data

- **`connection`** — output→input aliasing; one primitive with a
  spectrum: identity → slice (a region of the source, zero-copy); gathers and
  scatters are provided kernels users compose (§6.6). The only way I/O aliases across the input/output divide.
- **`promote`** — an I/O given a name in an owning system's namespace;
  always allowed, including for I/O that is already connected or
  aliased, so a higher level can always name what it needs.
- **`tie_inputs` / `tie_outputs`** — many same-kind I/O bound to one
  alias. Same kind only. A tie precedes any connection of its members;
  afterwards only the alias connects, and connecting through a member
  name is refused so that the tie is visible at every use site.
- **Index-map-valued I/O** — index maps are *data produced by components*
  (neighbour searches, sort permutations, sparse operator structure) and
  consumed by gathers and scatters. Connectivity is
  computable by the model.
- **Run policy and variable lifetime** — a property a parent gives a child at
  registration (eternal, transient, run_n, do_once): the parent deletes the child
  when its run counter reaches 0, and a variable's storage lives exactly as long
  as some port holds it (§4.4).
- **Chunked execution over bounded scratch** — N independent instances
  whose per-instance intermediate scratch would cap problem size if
  materialized at once (per-node dense solves, operator fits): run in
  chunks over reused scratch, results landing in the
  full-shaped output. A first-class pattern with its own ergonomics;
  touches shapes and lifetimes (§4.4). Specified in §7.7.
- **Initialization** — free input roots receive initial values without a
  component existing to produce them; the value is metadata on the root,
  declared by a component or by any owner above it, and the highest level
  that declares one wins (§4.4). Outputs are always computed fresh and are
  never initialized (§6.0).
- **Host access** — reading any I/O on the host by path, with host-side
  views that behave like the device wrappers; driving a free input
  (design variables and parameters) by giving it a new initial value
  (`Model::init_input`). Outputs are never written from the host.

#### Need 2.4 — Semantics

- **Design variables, constraints, objectives** — marks on I/O with
  bounds and scaling; marking any name of an aliased set marks the data;
  a design variable is a free input, an objective or constraint is an
  output.
- **Derivatives** — exact gradients by automatic differentiation are the
  default; users write forward kernels only. Finite differences,
  complex step and analytical partials remain as the floor of the
  derivative axis (opt-outs for memory or choice, and the guarantee that
  nothing is underivable). Adjoint memory policy (store vs recompute) is
  an execution attribute.

#### Need 2.5 — Deployment

- **Heterogeneous single-binary portability** across vendors and the CPU
  is table stakes: the same model runs on a laptop CPU, a consumer GPU,
  and a datacenter node without source changes.
- **Multi-device** — subsystems placed on different devices in one model
  (need 2.2 Placement), with data movement between them the framework's
  business, not the modeler's.
- **Possibility floor:** explicit device control stays reachable.

#### Need 2.6 — Implicit relations

A cycle is expressed as a **cyclic connection** ("x_rhs = x_lhs") plus "a
cycle needs a solver"; residual vocabulary never surfaces to the modeler,
though residual math runs underneath. Rationale: residual-first
interfaces gatekeep non-specialists; solver experimentation should not
require solver theory.

#### Need 2.7 — Temporal

Temporal components and systems: `march` plus recurrence connections
(state fed across iterations).

---

## 2. DTypes and I/O wrappers

Binding contract; declarations only. Metadata layer: §3. Kernel machinery:
§5. Component base and helpers: §6. System-side execution: §7.

### 2.1 Declarations

```cpp
namespace naga::cospan {

/// The closed set of I/O element types.
/// Atomic variants are distinct enum entries: declaring an I/O with an
/// atomic dtype changes the wrapper's accessor discipline (atomic
/// references) and nothing else about its storage. An atomic variant
/// exists exactly where sycl::atomic_ref is legal (4/8-byte element
/// types); u8 has no atomic form.
/// Why: §1 need 2.1 write-safety.
enum class DType : uint8_t {
    f32, f64, i32, u32, i64, u64, u8,
    atomic_f32, atomic_f64, atomic_i32, atomic_u32, atomic_i64, atomic_u64
};

/// Traits:
///   value_dtype_of<DT>      — underlying value dtype (atomic_f32 -> f32).
///     Connection/tie/promotion compatibility SHALL be judged on
///     value_dtype_of, never DT: the atomic modifier is an access
///     discipline over shared storage, not a storage type.
///     Why: §1 need 2.3 ties (the shared-I/O rules).
///   cpp_type_t<DT>          — C++ element type (through the modifier).
///   is_atomic_v<DT>         — true for atomic entries.
///   is_differentiable_v<DT> — true for f32/f64, plain or atomic.

// -----------------------------------------------------------------------------
// Wrappers — the only device-side data handles in the framework.
// -----------------------------------------------------------------------------
// Common contract:
//   * Trivially copyable; captured by value into kernels
//     (static_assert-enforced per instantiation).
//   * Wrappers carry NO identity — they are pure device payload
//     {shape, stride, data}. Component APIs that take a member reference
//     (e.g. reshape) resolve it to its IOEntry through the member-address
//     map (§3.3); nothing host-only rides into kernel captures.
//     Why: §1 need 2.1 minimal device payload.
//   * Raw pointers are private. The metadata layer (§3.3,
//     detail::IOEntry) is the sole writer of wrapper fields, via
//     friendship. User code never sees a pointer.
//   * `value_type` is the ONLY element-type spelling kernel bodies may
//     use. Automatic differentiation is a value type with the arithmetic
//     operators and math functions defined on it; a body that spells its
//     element type as `value_type` runs at that type as written.
//     Why: §1 need 2.4 derivatives.
//   * Rank is 1..3. Indexing accepts sycl::id<Rank> and flat size_t;
//     flat indexing is row-major over shape().
//   * stride() is the element step per dimension in the OWNING
//     allocation: subarray-aliased views (§8) are a metadata act,
//     never a wrapper variant.

/// Input<DT, Rank> — read-only view. THE INPUT CONTRACT: a kernel SHALL
/// NOT be able to write through an Input — all access is const-reference.
/// Load-bearing twice: ImplicitComponent's no-aliasing design and AD
/// purity both assume it (why: §1 need 2.6, §1 need 2.4 derivatives). Atomic dtypes are ill-formed on
/// Input: atomicity governs write collision, and Inputs cannot write.
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
struct Input {
    using value_type = cpp_type_t<DT>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    const value_type& operator[](sycl::id<Rank> idx) const;
    const value_type& operator[](size_t flat) const;

    sycl::range<Rank> shape() const;
    sycl::range<Rank> stride() const;
    size_t            size() const;   ///< flat element count

private:
    sycl::range<Rank> shape_;
    sycl::range<Rank> stride_;
    const value_type* data_;
    friend class detail::IOEntry;
};

/// Output<DT, Rank> — read-write view with exactly ONE access discipline
/// per instantiation, selected by the dtype:
///   * plain dtypes  -> bare mutable references ONLY. No atomic access
///     exists on the type; a plain Output in a colliding-write role is a
///     model error the tie check (§4) and, in a parallel wave, the
///     tied-outputs check (§7.5) catch.
///   * atomic dtypes -> sycl::atomic_ref (relaxed, device scope) ONLY.
///     No bare reference is obtainable — a kernel CANNOT forget
///     atomicity, by construction. Reads are atomic_ref::load().
template <DType DT, int Rank>
struct Output {
    using value_type = cpp_type_t<DT>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    value_type& operator[](sycl::id<Rank> idx) const requires(!is_atomic_v<DT>);
    value_type& operator[](size_t flat)        const requires(!is_atomic_v<DT>);

    sycl::atomic_ref<value_type, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        operator[](sycl::id<Rank> idx) const requires(is_atomic_v<DT>);
    sycl::atomic_ref<value_type, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        operator[](size_t flat) const requires(is_atomic_v<DT>);

    sycl::range<Rank> shape() const;
    sycl::range<Rank> stride() const;
    size_t            size() const;

private:
    sycl::range<Rank> shape_;
    sycl::range<Rank> stride_;
    value_type*       data_;
    friend class detail::IOEntry;
};

/// Initializer<DT, Rank> — the writable twin of an I/O for a custom
/// initial value: the same accessor discipline as Output (bare references
/// for plain dtypes, sycl::atomic_ref for atomic dtypes), over the free
/// input root's primary storage. Handed to the function
/// of cs::init::custom (§4.4); captured by value into a kernel that
/// function launches on the queue it receives.
template <DType DT, int Rank>
struct Initializer {
    using value_type = cpp_type_t<DT>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    value_type& operator[](sycl::id<Rank> idx) const requires(!is_atomic_v<DT>);
    value_type& operator[](size_t flat)        const requires(!is_atomic_v<DT>);

    sycl::atomic_ref<value_type, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        operator[](sycl::id<Rank> idx) const requires(is_atomic_v<DT>);
    sycl::atomic_ref<value_type, sycl::memory_order::relaxed,
                     sycl::memory_scope::device>
        operator[](size_t flat) const requires(is_atomic_v<DT>);

    sycl::range<Rank> shape() const;
    sycl::range<Rank> stride() const;
    size_t            size() const;

private:
    sycl::range<Rank> shape_;
    sycl::range<Rank> stride_;
    value_type*       data_;
    friend class detail::IOEntry;
};

/// HostView<DT, Rank> — a read-only host copy of one entry, shaped like it,
/// taken at construction: a device-kind root → root.host_view() (SlabKit
/// orders the read after the stream's prior work); a shared- or host-kind
/// root → queue.wait() then get_access(). A peer with a view (§8.1)
/// presents its region only. Host-side only; not a kernel capture.
template <DType DT, int Rank>
class HostView {
public:
    using value_type = cpp_type_t<DT>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    const value_type& operator[](sycl::id<Rank> idx) const;
    const value_type& operator[](size_t flat) const;
    sycl::range<Rank> shape() const;
    size_t            size() const;
    std::span<const value_type> span() const;

private:
    explicit HostView(const detail::IOEntry& entry, sk::queue& q);
    std::vector<value_type> host_;
    sycl::range<Rank>       shape_;
    friend class Model;
};

/// HostViewMut<DT, Rank> — the argument type of the function of
/// cs::init::custom_host (§4.4): a zero-initialised host copy of the free
/// input root, writable. It is built by the framework and committed on return;
/// commit() and destruction write the buffer into the entry's root storage
/// (usm_vector::from_host, which owns its copy of the bytes before
/// returning) and refresh the root's copies at other placements. User code
/// never constructs one.
template <DType DT, int Rank>
class HostViewMut {
public:
    using value_type = cpp_type_t<DT>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    value_type& operator[](sycl::id<Rank> idx);
    value_type& operator[](size_t flat);
    sycl::range<Rank> shape() const;
    size_t            size() const;
    std::span<value_type> span();
    void commit();
    ~HostViewMut();   ///< commit()

private:
    explicit HostViewMut(detail::IOEntry& entry, sk::queue& q);
    detail::IOEntry*        entry_;
    sk::queue*              q_;
    std::vector<value_type> host_;
    friend struct detail::HostViewAccess;
};

static_assert(std::is_trivially_copyable_v<Input<DType::f32, 2>>);
static_assert(std::is_trivially_copyable_v<Output<DType::atomic_f32, 1>>);
static_assert(std::is_trivially_copyable_v<Initializer<DType::f32, 1>>);

// Spelling samples:
//   Input<DType::f32, 1> x;
//   Output<DType::f64, 2> y;
//   Output<DType::atomic_f32, 1> accum;

} // namespace naga::cospan
```

### 2.2 Notes binding this section

- Dtype compatibility across connections, ties and promotions is judged on
  `value_dtype_of`; the atomic modifier never blocks aliasing — it changes
  which accessor discipline each side compiles against.
- `tie_outputs` requires the atomic modifier on member outputs by default
  (overridable) — §4 owns the rule; the dtype system here is what
  makes it checkable.
- DVs, constraints and objectives are metadata marks (§3 / §8);
  nothing in the wrapper layer knows about them.

---

## 3. I/O metadata, shape derivation, declaration API

Binding contract; declarations only. Wrapper types: §2. Kernel machinery:
§5.

### 3.1 Rank is authoritative

A declaration's rank is type-level (the wrapper's `Rank` parameter) and
semantic: rank-3 `{2,1,1}` and rank-1 `{2}` are different declarations with
different API surfaces (`id<3>` vs `id<1>`). The 1-padded `sycl::range<3>`
in metadata exists for exactly one purpose — uniform flat-size arithmetic
with no per-rank conditionals — and SHALL NOT leak into any API surface,
comparison, or user-visible semantics. Why: P6.

### 3.2 Validity timing

Self-side validity — spec count equals Rank, dim indices sane against the
declaring entry — is checked at construction (compile-time where
expressible). Source-side validity — the named source exists, is an input,
has compatible rank/size, integer value dtype where required — CANNOT be
known at construction (the source may itself be dynamic): it is checked at
EVERY shape resolution and throws on violation.

### 3.3 Declarations and metadata records

```cpp
namespace naga::cospan {

enum class IOKind   : uint8_t { input, output };

// -----------------------------------------------------------------------------
// Dimension specification — DimSpec is a CONCEPT, not a type.
// A dimension is specified by anything integral (a fixed extent) or by a
// derivation type. Derivation sources are INPUTS ONLY: an output's shape
// is the result of computation, so it can never determine one.
// Why: §1 need 2.1 shape source.
// -----------------------------------------------------------------------------

/// UNSPECIFIED (shape authority): this dimension is supplied from
/// outside, by whatever supplies the root's VALUE: a connection (or slice), a
/// tie, or, for a free input, its initializer's shape. A SPECIFIED dimension
/// (an integer, FromInputDim, FromInput, PerIndex, FromInputValues) is a
/// CONSTRAINT on the supplier (§3.6).
struct Unspecified {};
inline constexpr Unspecified unspecified{};

/// This dimension = dim `dim` of sibling input `input`'s shape.
struct FromInputDim  { std::string_view input; uint8_t dim; };

/// This dimension's extent = the VALUE of element `index` (flat) of
/// sibling input `input`. The source SHALL have an integer value dtype.
/// Reshaping is a host act, so the value is read on the host: the driving
/// input's value, in the declaring Component's own run_pre_compute
/// (§6.5) — one wait, nothing submitted for the resize. Only
/// entries using FromInput ever pay it. Orchestration is Component-local,
/// always; no producer knows its consumers.
/// Why: §1 need 2.1 shape source · P3 · P5.
struct FromInput { std::string_view input; size_t index; };

template <class T>
concept DimSpec = std::integral<T>
               || std::same_as<T, Unspecified>
               || std::same_as<T, FromInputDim>
               || std::same_as<T, FromInput>;

/// Sugar helpers.
///   unspecified_shape<R>()        — R unspecified dims
///   from_input_shape<R>(input)    — FromInputDim{input,0} .. {input,R-1}
///   from_input_array(input)       — whole-shape value-driven marker:
///     on a rank-R declaration expands to FromInput{input,0}..{input,R-1}.
///     ASSUMES the source is rank-1 with flat size == R; checked at every
///     shape resolution per the validity-timing rule. Accepted ALONE by
///     the declaration overloads; expansion against the member's Rank
///     happens inside the declaration.
template <int Rank> constexpr auto unspecified_shape();
template <int Rank> constexpr auto from_input_shape(std::string_view input);
constexpr auto from_input_array(std::string_view input);

// -----------------------------------------------------------------------------
// Declaration API (declared on the Component/System base — full base:
// §6). Declaration performs ZERO allocation; it records metadata and
// registers the member wrapper's address. The Component maintains TWO
// resolution maps over the same entries: declared name -> IOEntry (path
// resolution, enumeration, connections) and member address -> IOEntry
// (member-reference APIs: reshape and friends). Wrappers themselves carry
// no identity (§2). Rank = the member's Rank; the spec count must equal
// it, at compile time.
// -----------------------------------------------------------------------------

template <DType DT, int Rank, DimSpec... Specs>
    requires(sizeof...(Specs) == Rank)
void declare_input(Input<DT, Rank>& member, std::string name,
                   Specs... specs);
                   // why: §1 need 2.3 connection, §1 need 2.3 initialization · P8

/// An output with an unspecified dim that no tie_outputs owns is a setup error (§3.6).
template <DType DT, int Rank, DimSpec... Specs>
    requires(sizeof...(Specs) == Rank)
void declare_output(Output<DT, Rank>& member, std::string name,
                    Specs... specs);

/// Whole-shape overloads accepting the from_input_array marker alone.
template <DType DT, int Rank>
void declare_input(Input<DT, Rank>& member, std::string name,
                   /* from_input_array marker */);
template <DType DT, int Rank>
void declare_output(Output<DT, Rank>& member, std::string name,
                    /* from_input_array marker */);

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
    enum class How : uint8_t { fixed, unspecified, from_input_dim,
                               from_input, per_index_host, per_index_from_input };
    How      how;
    size_t   extent;        ///< fixed: declared; derived: last resolved
    std::string_view input; ///< from_input_*: sibling input name
    uint8_t  dim;           ///< from_input_dim
    size_t   value_index;   ///< from_input
    bool     per;           ///< dimension >= 1 is per-index (§3.5)
    std::vector<size_t> per_index;  ///< its extents, one per index of dimension 0
};

/// IOEntry — the metadata record behind one declared I/O. One instance
/// per declaration, owned by the declaring Component/System in a
/// name-keyed map. The virtual base is the RTTI anchor: typed retrieval
/// (get_input<DT,Rank>) dynamic_casts the entry; hierarchy resolution
/// (§10.3: get_subsystem chaining) is a chain of ordinary lookups ending
/// here.
class IOEntry {
public:
    virtual ~IOEntry();

    // ---- identity & classification -------------------------------------
    std::string_view name() const;
    IOKind   kind() const;
    DType    dtype() const;          ///< atomic-ness preserved
    int      rank() const;           ///< AUTHORITATIVE (see §3.1)

    // ---- shape ----------------------------------------------------------
    sycl::range<3> extents() const;  ///< 1-padded; flat-size arithmetic ONLY
    sycl::range<3> stride() const;   ///< element step in the owning allocation
    size_t flat_size() const;
    const DimRecord& dim(int d) const;  ///< d < rank()
    bool is_dynamic() const;         ///< any dim not How::fixed

    // ---- marks (semantics: §8) ------------------------------------------
    const std::optional<MarkRecord>& design_var() const;   ///< §8: bounds, scaling, region
    const std::optional<MarkRecord>& constraint() const;
    const std::optional<MarkRecord>& objective()  const;
    /// CONSTRAINT: a design-var entry SHALL NOT be dynamically shaped.

    // ---- the aliasing set (mechanics: §4) -------------------------------
    /// Every entry belongs to one set with one ROOT (the set's identity and, for
    /// outputs, its producer); peers are names for and consumers of the
    /// root's data. There are no source/target links.
    IOEntry* root();                 ///< self iff is_root()
    bool     is_root() const;
    bool     can_be_source() const;  ///< construction-time (§4.2)
    bool     can_be_target() const;
    /// Device placement and per-placement root storage — device(),
    /// placement(), storage_on(), storage_placements() — are declared in §4.

    // ---- registration metadata (ROOTS; recorded by the resolution pass, §4.4) --
    const std::optional<InitSpec>& init_spec() const;  ///< free input roots: the winning initial value, if any
    bool materialized() const;                       ///< some placement has live storage (§4.4)
    const Subsystem* declarer() const;               ///< every entry: the Subsystem that registered it

    // ---- capabilities ---------------------------------------------------

    /// Automatic shape update: applies the DimRecords. Invoked by the
    /// entry's Component in run_pre_compute, BEFORE connection validation.
    /// FromInput dims are resolved there first (the driving input's
    /// value read on the host, resize, point_wrapper — §6.5).
    virtual void update_shape() = 0;

    /// Resize (automatic derivation and user reshape share this path).
    /// REALLOCATION POLICY: growing past the current capacity reallocates
    /// through the root's allocator route (the same route that built it,
    /// §3.3 allocate); shrinking keeps capacity — usm_vector's contract;
    /// extents describe the logical shape. Why: P3.
    virtual void reshape(sycl::range<3> new_extents) = 0;

    /// Connection validation — INPUT PEERS ONLY, against root():
    /// FLAT-SIZE equality (rank and per-dim extents are free to differ
    /// across a connection); slice connections compare the view's size.
    /// Static pairs once, in prepare_exec; dynamic pairs every forward
    /// after update_shape.
    virtual void validate_connection() const = 0;

    /// Allocation: ROOTS ONLY. A variable is materialized on need,
    /// when something accesses it (§4.4).
    /// Builds the root's
    /// slabkit::usm_vector for each placement the set spans. A device
    /// placement takes its device's substrate (§7.1): a device-kind root
    /// that is fully fixed and whose every port is eternal → the arena
    /// route (its bytes are what get_exec_info counted); everything else →
    /// the pool route when the device has a pool, else the OS. A host-kind
    /// placement takes the OS route on its device and counts no arena bytes.
    virtual void allocate(sk::queue&, Substrates&) = 0;

    /// Write {data pointer, shape, stride} into the registered member
    /// Input/Output wrapper — the pointer being
    /// root()->storage_on(placement()) (+ view offset). This is what makes
    /// the user's member variable usable; called at the end of
    /// prepare_exec and after every resize.
    virtual void point_wrapper() = 0;
};

/// Typed leaves. InputEntry holds Input<DT,Rank>* to the member variable;
/// OutputEntry holds Output<DT,Rank>* plus, per placement, the owning data
/// handle slabkit::usm_vector<cpp_type_t<DT>> (§4; an input ROOT — an
/// unconnected input or tie alias — holds the same, since every root is a
/// usm_vector over allocator<T>).
template <DType DT, int Rank> class InputEntry  final : public IOEntry { /* ... */ };
template <DType DT, int Rank> class OutputEntry final : public IOEntry { /* ... */ };

} // namespace detail

// -----------------------------------------------------------------------------
// Enumerability — THE enumeration guarantee.
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

### 3.4 Samples

```cpp
// A 25x100 source consumed as 100x5x5:
Input<DType::f32, 3> reshaped;
declare_input(reshaped, "reshaped_input", unspecified, 5, 5);   // dim 0 from the connection; 5 and 5 are constraints on it

// Whole-shape value-driven: "dims" is a rank-1 u64 input of size 3.
Input<DType::f32, 3> grid;
declare_input(grid, "grid", from_input_array("dims"));
```

---

### 3.5 Ragged dimensions

A ragged port has rank 2 or 3 and, for every index i of dimension 0 (N indices), its own extent on each trailing dimension. Dimension 0 is always scalar.

**Ragged tie aliases.** A tie alias (section 4.3) takes the same per-index rules: `cs::alias::dims` with `per_index`, `output_values`, `output_dim`; `from_output_shape` and an unspecified tie_inputs (the connected output) copy ragged shapes.

**Spec forms.** `cs::PerIndex{values}` / `cs::per_index(vec)` (a host array, one extent per index, fixed at declaration) and `cs::FromInputValues{input, column}` (the extents are the values of an integer sibling input of rank 1 [N] or rank 2 [N, k]; negative values are an error). They are `DimSpec`s on dimensions >= 1 only, mixed freely with scalar specs. `cs::unspecified` and `FromInputDim` on a per-index dimension copy its array; a specified per-index dimension is a constraint on the supplier's array (§3.6).

```cpp
cs::Input<DType::u32, 1>        counts_;
cs::RaggedOutput<DType::f32, 3> pts_;
declare_input(counts_, "counts", cs::unspecified);
declare_output(pts_, "pts", cs::FromInputDim{"counts", 0}, cs::FromInputValues{"counts"}, 3);
// kernel: auto b = out.pts.block(i); b[sycl::id<2>{j, k}] = ...;
```

**Layout.** Blocks are stored contiguously in index order in one allocation; block i has the dense row-major shape (extent_1(i)[, extent_2(i)]) and starts at `offsets[i]`; `offsets` has N + 1 entries, the last is the total. The device holds copies of the offsets and of every per-index array (u64), kept per placement with the root and refreshed whenever N or an array changes. `flat_size()` is the total; `extents()` reports N on dim 0 and 0 on per-index dims; `geometry()` and a Slice are errors.

**Wrappers.** `RaggedInput<DT,R>` / `RaggedOutput<DT,R>`: `size()` is N, `block(i)` a dense `Block<T,R-1>` (`shape()`, `size()`, `[]` by id or flat). Trivially copyable, so they travel in Inputs/Outputs like any wrapper. They also read or write a dense port (block i = row i). **Atomic ragged outputs:** `RaggedOutput<DT,R>` with an atomic DT (e.g. `DType::atomic_f32`) returns an `AtomicBlock<T,R-1>` from `block(i)`: same `shape()` / `size()`, and `[]` (id or flat) gives a relaxed, device-scope `sycl::atomic_ref`, mirroring a dense atomic `Output`; no bare reference is obtainable. A ragged `tie_outputs` of atomic members follows the `require_atomic` default. `RaggedInput` stays non-atomic (read a plain-dtype view of the root).

**Host views.** `model.view_ragged_input<DT,R>(path)` / `view_ragged_output<DT,R>(path)`: `data`, `offsets`, `size()`, `extents(d)`, `is_per_index(d)`, `block(i)`; also on dense entries. `view_input`/`view_output` on a ragged entry throw and point to them.

**Runtime leading ranges.** Framework-only (`detail::LeadingRangeAccess`, no user API): a connected input (a non-root peer) can be pointed, between forwards, at rows / blocks [a, b) of its root along dim 0, and back (`clear`). Dense root: its view becomes the SubarrayRef with dim-0 offset a and extent b-a (composed with a setup-time Slice, which is restored by `clear`). Ragged root: N = b-a, the peer's host per-index arrays are the root's [a, b) slice, its device offsets and per-index pointers are the root's shifted by a, and its data starts at the root's offsets[a]; ragged wrappers address block i at `data + (off[i] - off[0])`, so a shifted pointer works. Setting a range re-runs the peer's shape update and `validate_connection` and re-points its wrapper. `a > b` and `b > N` are errors that leave the previous range in place. `view_ragged_input` of such a peer shows the range (offsets rebased to 0).

**Errors.** A per-index spec on dim 0; a dense wrapper declared with a per-index spec (setup); a dense wrapper whose entry inherits a per-index dim, a per-index length different from N, a negative extent, mismatching per-index extents or totals on a connection (forward); a Slice onto a ragged port (setup).

**Not supported.** Ragged ports in a FusedSystem or ParallelSystem are a setup error.

### 3.6 Shape authority

WHATEVER SUPPLIES A ROOT'S VALUE ALSO SUPPLIES ITS SHAPE: a connection (or a
slice of one), a tie, or an initializer. A component declares its shapes as if
it stood alone.

Per dimension of an entry:

- **Unspecified** (`cs::unspecified`, `cs::unspecified_shape<R>()`): the dim is
  supplied from outside. A connected input takes it from the output (or the
  slice view) it is connected to, per-index arrays included; a tie member
  takes it from the tie root; a free input takes it from its initializer's
  shape.
- **Specified**: the dim is a CONSTRAINT, never silently overridden:
  fixed (checked at setup against a fixed supplier, else at the first
  resolution), inherited (`FromInputDim`, checked once both resolve), and
  value-driven (`FromInput`, `from_input_array`, `FromInputValues`,
  `PerIndex`, checked once the value is read). A mismatch throws
  `std::invalid_argument` naming the entry, the dim, its declared rule and
  value, and the root with what supplies it.
- **Rank** always matches: a connected, sliced or tied entry of another rank than
  its supplier throws.

Setup errors (raised by `resolve()`): an OUTPUT with an unspecified dim that
no `tie_outputs` owns; a FREE INPUT with an unspecified dim and no
initializer that gives a shape. Initializers carry shapes:
`fill(v, shape)`, `zero(shape)`, `from_host(data)` (rank 1: its size),
`from_host(data, shape)`, `custom(fn, shape)`, `custom_host(fn, shape)`; the
shape-less forms stay valid when every dim is specified.

**The shape type.** `cs::init::Shape` is what those calls take. It is written as a
list of extents, `{n, m}`, or a `std::vector<std::size_t>`: both convert to it
implicitly, as the API's ordinary spelling. A RAGGED shape comes from
`cs::init::ragged(n, d1[, d2])`: dim 0 is `n`, and each `di` is either one
extent for every index (a `std::size_t`) or a `std::vector<std::size_t>` of
exactly `n` extents, one per index (a wrong length throws `std::invalid_argument`
when the shape is built; `n = 0` is legal). Dim 0 is always a scalar.
`from_host(data, ragged(...))` requires `data.size()` to equal the SUM of the
block sizes (the blocks end to end; a size-0 leading dim takes no data), else
it throws at the call. For a free input with unspecified dims the init's ragged
shape supplies them: each per-index vector becomes the root's full-range
per-index host array (cut to the chunk inside a ChunkedSystem, as any host
array is), and a ChunkedSystem takes its chunk length from it when the
input is a promoted free input.

**Every dim is checked.** When a free input root's init is applied (at the
full range only: never per chunk, never in a ChunkedSystem's replay) the
init's shape, if it has one, is compared with the resolved root on the rank and
on EVERY dim: fixed, inherited, value-driven, and per-index arrays element
by element. A mismatch throws `std::invalid_argument` naming both sides, e.g.
`init for 'x' has dimension 1 = [3,0,4] but 'x' resolved dimension 1 (PerIndex) = [3,1,4]`.
An init with no shape still needs every dim of the root specified (the setup
error above). Bare data, `from_host(data)`, describes a rank-1 root only: a
rank-2 or rank-3 root takes it by its size.

Inside a ChunkedSystem dim 0 is supplied by the chunked system (§7.7). A
SPECIFIED dim 0 is not an error in itself: a fixed one is a constraint checked
once, against the full length, on the full pass, never per chunk.

## 4. Aliasing

Binding contract; declarations only. Completes §3's IOEntry and gives §7's
structural APIs their mechanics. Storage follows the setup chain in §11.

### 4.1 The model: one relation, root + peers

Every IOEntry belongs to exactly one SET. A set has one ROOT — the entry
that owns the data and, for outputs, the entry that produces it — and any
number of PEERS: names for the same data (promotions, tie aliases) and
consumers of it (connected inputs). Every entry points at its root; the
root holds the peer list. Merging a set into another steals the peer list
and repoints every member — no chains exist and nothing is ever walked
at forward time: a peer reads `root()->data()`, one dereference. Why:
§1 need 2.3 connection (aliasing correctness) · P3.

There are no source/target links. Everything they would carry is a
property of the set: the root is the producer; each input peer is a
consumer; connection validation is each input peer against its root; a
subarray connection is a fact on the input peer ("which part of the root
I see"); later cycle analysis is a graph over components through root
ownership.

### 4.2 Aliasing rules: two-level check, construction-time flags

Each entry carries `can_be_source` and `can_be_target`. An operation on
an entry passes iff BOTH the entry's own flags AND its root's flags allow
it: the entry's flags say "what was I created to support"; the root's
flags say "has aliasing since invalidated that". Flags are set at
construction and NEVER mutated afterwards, with exactly ONE exception
(§4.3, tie_*). Roots need no snapshot logic: a root's own flags are its
set's flags. Why: P1 (a promoted input must never quietly become
a source) · P6.

```cpp
namespace naga::cospan::detail {

class IOEntry {
public:
    // ---- identity, shape, marks: as in §3.3 ----------

    // ---- the set --------------------------------------------------------
    IOEntry*       root();                 ///< self iff is_root()
    const IOEntry* root() const;
    bool           is_root() const;
    std::span<IOEntry* const> peers() const;   ///< root only; empty otherwise
    bool           owns_data() const;      ///< == is_root()

    // ---- aliasing flags (construction-time; see §4.2/§4.3) --------------
    bool can_be_source() const;
    bool can_be_target() const;

    // ---- consumer view (input peers only) -------------------------------
    /// The subarray of the root this peer sees; whole-root when unset.
    std::optional<SubarrayRef> view() const;

    // ---- execution placement (set in get_exec_info; §11) ----------------
    const sycl::device& device() const;        ///< the owning subsystem's device
    Placement           placement() const;     ///< this entry's (device, storage kind)
    /// THE ROOT HOLDS ALL STORAGE, PER PLACEMENT. A Placement is a
    /// (device, kind) pair, declared beside IOKind. A root owns one
    /// allocation at its own placement (the primary, where its producer
    /// writes: the declarer's device and the declarer's default storage
    /// kind — device kind, or host kind for a host component) and one at
    /// every other placement any peer of its set lives at. Keying by
    /// placement lets a root hold a device-kind and a host-kind copy on one
    /// device, as a device producer feeding a host component requires.
    /// Peers own nothing: every peer reaches up to
    /// root()->storage_on(its placement), so peers at the same placement —
    /// however they relate to each other — share one pointer, and
    /// get_output("alias") from a System always yields memory on that
    /// System's device. The root's producer copies primary -> each other
    /// placement's storage in run_post_compute (sync_other_placements: a
    /// copy-assignment of the primary usm_vector to every other slot, the
    /// ordinary usm_vector copy rule). Registered in get_exec_info (register_placement;
    /// set_primary_placement puts the root's own placement first),
    /// materialized on need in a forward (§4.4). Why: §1 need 2.2 placement · P1 (a name
    /// yields memory where the reader is).
    void*  storage_on(const Placement&) const;                 ///< roots only
    std::span<const Placement> storage_placements() const;     ///< roots only; primary first

    // ---- capabilities (§3.3, restated against the set) ------------------
    /// validate_connection: INPUT PEERS ONLY, against root() — flat-size
    /// (or view() subarray size) equality; static pairs once in
    /// prepare_exec, dynamic pairs every forward after update_shape.
    /// allocate(sk::queue&, Substrates&): ROOTS ONLY — one usm_vector per
    /// placement in storage_placements(); a device placement from that
    /// device's substrate, a host placement through the OS route; no-op on
    /// peers.
    /// point_wrapper: root()->storage_on(placement()) (+ view() offset/
    /// stride) into the member wrapper — which is why root regrowth needs
    /// no notification: every peer re-points at its next turn.

private:
    IOEntry*              root_  = this;
    std::vector<IOEntry*> peers_;          // meaningful iff root_ == this
    bool can_be_source_, can_be_target_;
    std::optional<SubarrayRef> view_;
    friend class naga::cospan::System;     // the four structural ops
};

} // namespace naga::cospan::detail
```

### 4.3 The structural operations, as set operations

**Ragged aliases.** A tie alias takes the same per-index shape rules as a port. `cs::alias::dims(spec0, spec1, ...)` gives one spec per dimension (count = Rank): an integer (fixed), `cs::per_index(vec)` (dims >= 1), `cs::alias::output_value(path, index)` (a scalar from an upstream output's value), `cs::alias::output_values(path, column = 0)` (per-index extents from the values of an upstream integer output of rank 1 [N] or rank 2 [N,k]; dims >= 1) and `cs::alias::output_dim(path, d)` (dim d of an upstream output's shape, per-index if that dim is). `from_output_shape` copies the upstream shape including per-index dims, and an unspecified tie_inputs inherits a ragged source wholesale. The same Fused/Parallel refusal applies to every upstream output a shape reads. The declaring System resolves per-index dims on the host, computes the block offsets, resizes and uploads the device copies like any ragged root; members (unspecified dims) copy the arrays from the root and must be declared RaggedInput / RaggedOutput (`declare_*(w, name, cs::unspecified_shape<R>())` overloads exist for them; a dense wrapper is an error). Ragged wrappers are non-atomic, so a ragged tie_outputs takes `TieOutputsPolicy::allow_plain`. Inside a ChunkedSystem an internal ragged alias root is priced like any internal root, dim 0 is set by the framework, a `per_index` host array is sliced per chunk, and a per-index source computed inside is the "computed inside" error. **Alias sources may also be INPUTS** of the declaring System (a promoted input or a tie_inputs alias, connected from upstream or free): `cs::alias::input_value(path, index)`, `input_values(path, column = 0)`, `input_dim(path, d)` (per-index if that dim is) in `alias::dims`, and the whole-shape `cs::alias::from_input_shape(path)` (ragged-aware). Values and shapes are read from the input's ROOT on the host (snapshot), like the output forms. Ordering: a value a shape rule reads must be known when the alias is accessed, else the read throws there (shapes are reactive, §4.5); the Fused/Parallel refusal covers every source.

`merge(into, from)`: append `from`'s peers and `from` itself to `into`'s
peer list; set every moved entry's root to `into`. O(|set|); models are
never large enough for this to matter, and the flatness is what makes
forward free.

**declare_input / declare_output** (§6): a new entry, root of a
singleton set. Inputs: `can_be_source = false, can_be_target = true`.
Outputs: the reverse.

**promote(path, name)** — kind-agnostic, ALWAYS allowed (including on
already-connected or already-aliased entries: promotion exists to give a
higher level a name, e.g. probing an adopted System's internals). Create
the alias, register it in the calling System's namespace, join it to the
resolved entry's set as a PEER (never a root), with its flags a SNAPSHOT
of the root's flags at creation. The snapshot is what keeps a promoted
input from becoming sourceable after its set is later connected. The
alias enumerates under `get_inputs()`/`get_outputs()` by the flags it
snapshotted. `promote<DT, Rank>(path, name)` names the dtype and rank,
which must be the entry's, else a setup error; the alias MIRRORS the entry: its
shape is read live from it, so a view through the promoted name shows the member's
current shape.

**connect(src, dst)** — check `src.can_be_source ∧ root(src).can_be_source
∧ dst.can_be_target ∧ root(dst).can_be_target`, else throw naming the
failing level. Then `merge(root(src), root(dst))`. No flag changes: the
second connection to any name in the set now fails at the ROOT check
(the root is an output root, `can_be_target == false`) — fan-in is
blocked structurally. Fan-out is the same op repeated. Subarray forms
record `view()` on the input peer. Connecting through ANY name of a set
is the same connection. Only an OUTPUT root can be a source: a free
input, and a tie alias root of `tie_inputs`, cannot, because an input has
no producer and "connect from it" could only mean "share its root", which
is what a tie already is. A `Slice` therefore reads an output root.

**tie_inputs({m...}, name)** — the alias is a NEW ROOT with
`can_be_target = true, can_be_source = false`. Each member's root must be
an input root with `can_be_target == true` — a member whose set is
already connected fails here ("tie before connecting"). The
call is `tie_inputs<DT, Rank>({m...}, name[, AliasShape])`. Fewer than two
members is an error ("a tie needs two or more members; use promote for
one"). Every member is an entry (or a promotion of one) of dtype DT and rank Rank;
it may declare any dims: its unspecified dims come from the tie and its
specified dims are constraints on it (§3.6). A tie alias carries a shape, so it cannot be a
member of another tie: pass members up by promotion.) Merge every member's
set into the alias. THE ALIAS OWNS THE SHAPE, given by its AliasShape:
`cs::alias::fixed(e0[, e1, e2])`; an OMITTED shape (tie_inputs only, and the default: the shape of the
output connected into the alias, else of an initializer declared on the tie,
else the free-input setup error of §3.6); `cs::alias::from_output_shape(path)` (the shape of an
upstream output of the alias's rank); `cs::alias::from_output_values(path)`
(the extents are the VALUES of elements 0..Rank-1 of an upstream output of
integer dtype, read on the host as FromInput reads a value). The paths are in
the declaring System's namespace and name an output. If that output has
not been produced when the tie materializes, the read throws there with the
reactive message of §4.5 (FusedSystem and ParallelSystem refuse a source from
the same wave, as for value-driven sizes). The unconnected alias is the root; the declaring System resolves
the shape when the alias is accessed, before materialization. Once an output is
connected into the alias it joins that output's root (the producer's port holds
its storage) and the members take that output's shape; a fixed or
from_output_* AliasShape is checked against it on the first run
(`invalid_argument` on a mismatch). A member's unspecified dims are "from the root"
(`DimRecord::How::unspecified`, resolved in update_shape). THE EXCEPTION: mark each member
`can_be_target = false`, so a later connect *through a member name*
fails at the direct check. Data-wise it would be identical to connecting
through the alias; the mark exists so a tie is visible at every use
site — a discipline rule, not a correctness one, and the only
post-construction flag mutation in the system. Why: P1.

**tie_outputs({m...}, name, policy)** — the alias is a NEW ROOT with
`can_be_source = true, can_be_target = false`; policy `require_atomic`
(default) checks every member's dtype is atomic (§2), `allow_plain`
is the user's assertion. Each member's root must be an output root whose
set contains NO input-kind peer — a downstream connection already exists
otherwise ("tie before connecting, or connect from the alias"); output-
kind peers (promotions) ride along. Merge. THE EXCEPTION, mirrored: mark
each member `can_be_source = false`. SHAPE: the call is
`tie_outputs<DT, Rank>({m...}, name, AliasShape, policy)`; the alias is ALWAYS the root and owns the shape by
its AliasShape (fixed, dims, from_output_shape, from_output_values,
from_input_shape; tie_outputs REQUIRES one: an output with unspecified dims is an
error). Members may declare dims; those are constraints on it (§3.6). The declaring System resolves the shape before materialization, so
the root is sized from it each run. This is what the
zeroing producer plus accumulator of §6.7 (ZeroBins + Histogram) uses: the
bin count is the value of an upstream output.

Path resolution for every operation is from the caller's namespace,
dot-separated, descending children; the last segment resolves an alias
registered in that System or a Component's declared entry.

### 4.4 What owns memory

Two kinds of thing carry the model's data. A VARIABLE says what the data is; PORTS hold its memory. Nothing else owns memory and nothing else decides how long it lives: there are no lifetime rules beyond ownership.

**Variable.** A VARIABLE is one piece of model data, everything joined by `connect`, `promote` or `tie_*` (§4.3). It holds its dtype, rank, shape and dims; at most one source (the output that produces it); its marks (design variable, constraint, objective); the initial-value declarations made on it (below); its write epoch and its shape version; and, for each PLACEMENT (a device plus a storage kind) that has ports on it, a WEAK reference to that placement's storage. A variable holds no memory itself, and a mark keeps nothing alive.

**Port.** A PORT is a Component input or output, or a System's promotion or tie alias. It refers to its variable and holds a SHARED reference to the storage of its own placement. Ports own nothing beyond that share: two ports on one placement share one storage, and the storage is the thing they share. A promotion at the model level is a port of the top System, which is never deleted, so it keeps its placement alive for the model's life.

**Materialization.** A variable is MATERIALIZED on need, implicitly, when something accesses it: a Component's turn, or a System resolving an alias. It allocates the storage of every placement that has ports, at once. Route (§7.1 `MemorySubstrate`): a device placement of a device-kind variable that is static, and whose every port belongs to a subsystem that is never deleted (eternal all the way up), takes its device's arena; any other device placement takes that device's pool when it has one, else the OS; a host-kind placement takes the OS route on its device (a host-kind placement adds no arena bytes, and a host-primary variable has no static bytes) — at materialization and on growth. A placement is LIVE while its storage exists. If the variable is still live on some placement, a placement that comes back (its ports were deleted and rebuilt, say) is filled by COPYING from a live one. The initial value applies only when the variable is live nowhere, because only then is the memory fresh and uninitialized. A live variable is never re-initialized by the framework.

**Freeing is per placement.** When the last port holding a placement's storage goes away, that placement's memory is freed. Ports go away when their subsystem is deleted (run policy, below). A variable that has no port on any placement is not materialized; it survives as metadata and is materialized again when something accesses it.

**Producer push.** After a launch, a producer copies what it wrote to the variable's other placements. Ports on those placements see the copy by the ordering of the queue.

**Ordering without waits.** The kernel events of a launch are registered (SlabKit `usm_vector::register_events`) on the storage of every port the launch touched. Later memory operations on that storage (a resize, a producer push, a copy into a returning placement) are ordered after them by the queue, and a free waits on them itself. The framework never does a queue-wide wait to free.

**Run policy belongs to the PARENT.** A parent gives each child (a Component or a System) a run policy with a property on `add_subsystem` (§7.4). The parent keeps one counter per child:

| property | meaning |
|---|---|
| none | ETERNAL: no counter; the child runs every forward |
| `property::run_n{n}` | the counter starts at n >= 1 (n = 0 throws `invalid_argument`); the child runs n forwards |
| `property::do_once{}` | `run_n{1}` |
| `property::transient{}` | runs every forward; exists for exactly one forward of its parent: BUILT at the start of the parent's forward, DELETED at its end |

`run_n` with `do_once` is an error; so is `transient` with either (`add_subsystem('x'): transient cannot be combined with run_n or do_once`). A `run_n` or `do_once` counter moves after the child's turn (inside a chunked system, once after the last chunk). When it reaches 0 the parent DELETES the child right after that turn: everything the child held goes with it, and variables survive only through other ports. A child does not know its own lifetime. A parent's limits apply to its children by construction: a deleted parent takes its children with it, and a rebuilt parent rebuilds them. A TRANSIENT child exists for exactly one forward of its parent: the parent builds every missing transient child at the START of its forward loop, before any child's turn, and deletes them at the END, after every child's turn (inside a chunked system: built before the first chunk, deleted after the last). So a transient reader's ports exist while its transient producer runs, the ports keep the data, and it is freed at the end of the parent's forward. Rebuilding a transient child every forward is an honest cost and is deliberately not optimized.

**Rearm is a call on the parent, by child name.** `rearm(name)` on a System re-constructs the named child from the constructor arguments the parent stored at `add_subsystem`, runs its setup, replays the parent's own `connect`, `promote`, `tie_*` and `init_input` declarations that touch it, and prepares it for execution. A rebuilt child rejoins the same variables.

**Lookup.** `get_subsystem(path)` walks the dotted path from the top and returns a pointer. It returns null if the subsystem was declared but is currently deleted, and throws if the name was never declared. `add_subsystem` returns nothing.

INITIAL VALUES ARE METADATA ON FREE INPUTS. A FREE INPUT is an input variable with no source. Its initial value is declared during `setup()` and applied by the framework; outputs are never initialized (§6.0).

```cpp
namespace naga::cospan::init {
template <class T> InitSpec fill(T value);                 // every element = value
InitSpec                    zero();                        // every element = 0
template <class T> InitSpec from_host(std::vector<T> v);   // element i = v[i]; size = the variable's flat size
// A shape: {n, m}, a std::vector<std::size_t>, or ragged(n, d1[, d2]) (§3.6). fill, zero, from_host, custom and
// custom_host also take one (fill(v, shape), from_host(v, shape), ...); a ragged from_host needs the sum of the block sizes.
class Shape;
Shape ragged(std::size_t n, /* d1 */ std::size_t or std::vector<std::size_t>, /* d2 */ std::size_t or std::vector<std::size_t> = none);
template <DType DT, int Rank>
InitSpec custom(std::function<void(sk::queue&, Initializer<DT, Rank>)> fn);   // device-side; the function enqueues on the queue
template <DType DT, int Rank>
InitSpec custom_host(std::function<void(HostViewMut<DT, Rank>&)> fn);         // host-side; committed on return
}
// ComponentBase::init_input(std::string_view name, InitSpec)   — on the Component's OWN input
// System::init_input(std::string_view path, InitSpec)          — a child input path, or one of
//                                                                this System's promoted or tied input aliases
```

- **Highest level wins.** The declaring subsystem's depth decides, the top System being depth 0; the smallest depth wins. Two declarations at the same depth on one variable are an error at setup (the resolution pass).
- **A variable with a source ignores its init.** An init declared on an input whose variable ends up with a source (a connected input) is ignored, so a reusable component can carry a default for the case it is left unconnected. An init declared on an output is an error at setup.
- **No init, zeros.** A free input with no init is zero-filled.
- **When it applies.** The init applies at MATERIALIZATION, and only when the variable is live nowhere (above). Nothing is applied at setup, and nothing is applied to a live variable: a value-driven resize preserves the existing content and does not re-init, and a placement that returns is filled by copying, not by the init.

`Model::init_input(path, init)` is the user's explicit host write. It beats the declared inits, and it is applied at the variable's next run even if the variable is live, once per call. An output path throws, and a connected input warns on std::cerr and is never applied. A variable is MATERIALIZED once its storage has been allocated, even at size 0; views of a materialized size-0 variable are empty, views of one that is not materialized throw.

HOST ACCESS. Host views are host copies and hold nothing: a view neither materializes a variable nor keeps one alive. Reading a variable that is not materialized throws, saying it has no data while nothing that uses it exists (a silent empty view would read as valid data). Why: §1 need 2.1 run policy and lifetimes, §1 need 2.3 initialization · P3 · P6 (declaration is one phase; the resolution pass is its end).

### 4.5 From sets to storage

Aliasing is complete the moment each structural call returns. Storage follows in the setup chain (§11) and the first forward: `get_exec_info` registers on each variable the placements its ports span (the primary placement first) and adds the variable's bytes once per such device-kind placement, none for a host-kind one or a variable that is not arena-eligible, which sizes the arena; `prepare_exec` allocates nothing, it only binds the substrates and sets what can be set. A variable is materialized when first needed (§4.4), and the Components and Systems that hold its ports point them at their placement's storage, at their turn. Growth replaces a placement's storage and needs no notification: ports re-point at their next turn. Copies at the non-primary placements are refreshed by the variable's producer after every launch (producer push, §4.4).

`tie_outputs` members MUST share the alias's placement: producers cannot write one storage from two placements. Checked in the members' Components' `get_exec_info` (an output member whose placement differs from the alias's — the alias was processed first by its System, so the placement is known), error naming the tie.

SHAPES ARE REACTIVE. A shape is read when something needs it at run time. If it is not known then, the read throws there, naming what needed it, which dimension, and what it was waiting on (a producer that has not run, an input whose value is unread, a connection that supplies none). Setup rejects only fully declared contradictions (§3.6); it does no ordering or cycle analysis for shapes.

CYCLE WARNING. At setup, a connection whose consumer runs before its producer in registration order (depth-first, which is execution order) is a cycle. CoSpan has no solvers yet, so it prints `naga::cospan: warning: ...` on std::cerr naming both, and never throws. The check is independent of shapes.

### 4.6 Sample

```cpp
// declared: d1.out (root O1), d2.out (root O2), c.in (root I1)
promote<DType::f32, 1>("d1.out", "y");   // y joins O1's set as a peer; flags = O1's
connect("y", "c.in");         // O1 absorbs I1's set: {y, c.in} peers of O1
connect("d2.out", "c.in");    // FAILS at the root check: root(c.in) = O1,
                              //   an output root, can_be_target == false
tie_outputs<DType::f32, 1>({"d1.out","d2.out"}, "sum", alias::fixed(8));   // FAILS: O1's set has an
                              //   input-kind peer (c.in) — connected already
```

---

## 5. Kernel machinery

Binding contract; declarations only. Wrappers: §2. Metadata/shapes: §3.
Component: §6. System-side execution: §7.

**Library shape.** CoSpan is a header-only INTERFACE library. The
placeholders (`kb_entry_rR`), the outer kernels and every stage function
instantiate in the consumer's translation unit: the JIT finds a stage
definition only when it is emitted in the same translation unit as the
outer kernel that calls it. A model is built in one translation unit.

### The map taxonomy is doctrine, not metadata

Bijective, injective, surjective: the completeness argument — each class
being implementable guarantees Components can express any function — and
the organizing scheme for provided kernels (reductions, scatters, scans, sorts).
Kernels are NEVER classified by their authors; no class field exists.
The general case is guaranteed by construction: indexed access plus atomic
dtypes (§2), available in every kernel body. Why: §1 need 2.1 maps · P7.

### 5.1 The KernelBody contract

```cpp
namespace naga::cospan {

/// A kernel is a TYPE with nested argument structs and one call operator:
///
///   struct MyBody {                      // POD config lives as members
///       struct Inputs  { Input<...>  a; Input<...> b; };
///       struct Outputs { Output<...> y; };
///       struct Scratch { Local<...>  tile; };     // OPTIONAL (local mem)
///       void operator()(sycl::id<R> idx, Inputs, Outputs) const;
///          // or (GroupItem<R>, Inputs, Outputs)
///          // or (GroupItem<R>, Inputs, Outputs, Scratch)
///          // or (CustomOp&, Inputs, Outputs)        — CUSTOM OPERATION
///          // or (Host, Inputs, Outputs)             — HOST BODY
///   };
///
/// CUSTOM-OPERATION BODIES are never compiled to a device function or
/// launched: the owner submits them via AdaptiveCpp_enqueue_custom_
/// operation through the same SubmitTarget as kernels (§7.1) — so the
/// stream orders them like any other submission (§7.3) and gating applies
/// unchanged.
/// Inputs/Outputs arrive as the bare wrappers with CURRENT device
/// pointers. A custom operation is for native backend work on the
/// backend's own stream: the body runs when the work is SUBMITTED, not
/// when it executes, so it may only enqueue asynchronous work on the
/// native queue — a vendor library call, a native kernel launch
/// (CustomOp::h is the interop handle). On a backend with no native queue
/// (the OpenMP host device) the body still runs and receives correct
/// device pointers. Not fusable (no device function exists).
///
/// HOST BODIES — `void operator()(Host, Inputs, Outputs) const`, `Host`
/// being an empty tag — are the fourth kind: host work as an ordinary
/// component (existing user code, file I/O, callbacks). They are never compiled
/// to a device function. Every input and output of a host component is
/// HOST-kind USM on the component's device (pinned host memory the body
/// reads and writes directly), so data reaches it and leaves it by the
/// ordinary rule for a producer and a consumer at different placements
/// (§4.2). The owner submits the body as an async host task ordered in
/// the stream: `async_host_in_order` on the stream, or the group's
/// `async_host` under a Parallel owner's group (§7.1, §7.3). A HOST BODY
/// IS PLAIN HOST C++: no SYCL calls except `sycl::free`, no device code,
/// no backend API calls (§11.6). Not fusable.
///
/// THE SIGNATURE IS THE DECLARATION, four ways:
///   * first parameter id<R> vs GroupItem<R>  => group semantics opt-in
///     (a body without the GroupItem cannot touch group abstractions);
///   * first parameter CustomOp& or Host    => custom-operation or host
///     body (no device function; kernel_rank_v is 0);
///   * R                                    => kernel rank, deduced
///     (kernel_rank_v<KB>; no static member);
///   * trailing Scratch parameter           => local-memory opt-in
///     (GroupItem bodies only — local memory without a group is
///     meaningless).
/// Inputs, Outputs, KB itself, and Scratch (if present) SHALL be
/// trivially copyable — concept-enforced; they are copied byte-wise
/// into the launch's device streams.
/// Bodies spell element types as the wrappers' `value_type` (§2); nothing
/// in a body is written for differentiation.
template <class KB> concept PlainBody   = /* operator()(id<R>, In, Out) */;
template <class KB> concept GroupBody   = /* operator()(GroupItem<R>, In, Out [, Scratch]) */;
template <class KB> concept HasScratch  = /* trailing Scratch parameter */;
template <class KB> concept CustomOpBody= /* operator()(CustomOp&, In, Out) */;
template <class KB> concept HostBody    = /* operator()(Host, In, Out) */;
template <class KB> concept KernelBodyC =
    std::is_trivially_copyable_v<KB> &&
    std::is_trivially_copyable_v<typename KB::Inputs> &&
    std::is_trivially_copyable_v<typename KB::Outputs> &&
    (int(PlainBody<KB>) + int(GroupBody<KB>) + int(CustomOpBody<KB>) +
     int(HostBody<KB>) == 1);
/// From the signature; 0 for custom-operation and host bodies.
template <class KB> inline constexpr int kernel_rank_v = /* from signature */;
/// kernel_rank_v, or 1 when that is 0: types the range parameters of
/// create_launch_config.
template <class KB> inline constexpr int launch_rank_v = /* see above */;

/// What a custom-operation body receives: the interop handle (native
/// queue/stream/device of the device this submission targets) and the
/// door to raw device pointers. Anything else (a library call on the
/// native stream, a native kernel launch) is the body's own business, and
/// it only enqueues asynchronous work.
struct CustomOp {
    sycl::interop_handle& h;
    /// THE ONE PLACE a raw device pointer is obtainable (§2: wrappers
    /// never expose one). A custom operation exists to hand memory to
    /// code outside the framework; this is that door, and it opens only
    /// inside a custom-op body. It yields `const value_type*` for an Input
    /// and `value_type*` for an Output or Initializer. Why: §1 need 2.1
    /// work outside device kernels, §1 need 2.1 minimal device payload.
    template <class W> auto device_ptr(const W& io) const;
};

} // namespace naga::cospan
```

### 5.2 Local<DT, Rank> — local memory with wrapper ergonomics

```cpp
namespace naga::cospan {

/// Same accessor discipline as Output (bare references; plain dtypes
/// only in v1), but the storage is GROUP-LOCAL and the wrapper carries a
/// byte OFFSET instead of a pointer — local memory has no host address.
/// The user shapes it host-side in compute_forward() like any other
/// launch parameter:
///     MyBody::Scratch s;  s.tile.reshape({17, 16});
/// create_launch_config() lays out all Locals in a Scratch (aligned,
/// summed -> the stage's scratch_bytes); the device function fixes up
/// each Local's pointer as local_base + offset before the body runs.
/// Per-group instance semantics are local memory's own; the framework
/// adds nothing.
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
struct Local {
    using value_type = cpp_type_t<DT>;

    void reshape(sycl::range<Rank> shape);        // host, compute_forward
    value_type& operator[](sycl::id<Rank> idx) const;  // device, post-fixup
    value_type& operator[](size_t flat) const;
    sycl::range<Rank> shape() const;
    size_t size() const;

private:
    sycl::range<Rank> shape_;
    size_t            offset_;   // into the launch's local allocation
    value_type*       ptr_;      // null until device-side fixup
    friend struct detail::ScratchLayout;
};

} // namespace naga::cospan
```

### 5.3 GroupItem<R> — the group body's item

```cpp
namespace naga::cospan {

/// What a group body receives. The API of sycl::nd_item<R>, with every
/// group-level index STAGE-LOCAL: the body sees the launch it was written
/// for, whether it runs alone or as one stage of a fused wave. Local ids
/// and ranges are the physical ones (a work-group is never split across
/// stages); group and global ids are offset and delinearized over this
/// stage's own group grid.
template <int R>
struct GroupItem {
    sycl::id<R>    get_local_id() const;          size_t get_local_id(int d) const;
    sycl::range<R> get_local_range() const;       size_t get_local_range(int d) const;
    size_t         get_local_linear_id() const;

    sycl::id<R>    get_group_id() const;          size_t get_group(int d) const;   // stage-local
    sycl::range<R> get_group_range() const;       size_t get_group_range(int d) const;
    size_t         get_group_linear_id() const;   // stage-local

    sycl::id<R>    get_global_id() const;         size_t get_global_id(int d) const;
    sycl::range<R> get_global_range() const;      // this stage's padded range
    size_t         get_global_linear_id() const;

    /// The PHYSICAL group, for group functions — sycl::group_barrier(
    /// it.get_group()), reduce_over_group, ... Barriers are group-scoped and
    /// the physical group is exactly the set of work-items executing this
    /// stage together. Its own get_group_id() is physical; index with the
    /// item's.
    sycl::group<R>  get_group() const;
    sycl::sub_group get_sub_group() const;

private:
    sycl::nd_item<R> it_;             // the physical item
    sycl::id<R>      group_id_;       // this stage's group id, stage-local
    sycl::range<R>   group_range_;    // this stage's group grid: ceil(range[d] / wg[d])
    friend struct detail::ItemFixup;  // the sole constructor
};

} // namespace naga::cospan
```

A wave lays its stages' work-groups out along dimension 0 only:
`nd_range<R>{ {G·wg0, wg1, wg2}, {wg0, wg1, wg2} }`, so the physical group
linear id is the group index along that axis. The device function (§5.5)
builds the item from the stage record: the stage-local linear group is the
physical linear group − `group_offset`, delinearized over `group_range`
into `group_id_`; the item stores that id and the grid (`group_range_`),
so every group-level query answers stage-locally. A
Component launched on its own is a one-stage wave with `group_offset = 0`;
the same fixup serves both. Stages in a wave may have any ranges; the only
geometry the wave imposes is the work-group shape. A user barrier inside a
body is safe in either wave mode because the framework skips work in whole
groups only: every work-item of a live group enters the body together.

### 5.4 StageRecord and FnOffsetInfo

```cpp
namespace naga::cospan {

/// One StageRecord per stage per launch, fixed-size, packed into a
/// device array in EXECUTION ORDER. Field ownership:
///   component-bound (create_launch_config): rank, range, flat_extent,
///     stream offsets, scratch_off/scratch_bytes;
///   System-stamped (launch pack): group_offset, group_count, group_range —
///     the GROUP-INDEX slice — and local_off, the stage's byte offset into
///     the launch's local buffer. Slices are counted in whole work-groups, so a
///     group never straddles stages under ANY geometry. Why: §1 need
///     2.2 fuse, §1 need 2.1 group-level parallelism.
struct StageRecord {
    sycl::range<3> range;         // logical iteration space, 1-padded
    size_t   flat_extent;         // product of range
    size_t   group_offset;        // System-stamped
    size_t   group_count;         // System-stamped
    sycl::range<3> group_range;   // System-stamped; group_count is its product
    uint32_t input_off;           // into the inputs stream
    uint32_t output_off;          // into the outputs stream
    uint32_t body_off;            // into the bodies stream
    uint32_t scratch_off;         // into the bodies stream (Scratch bytes)
    uint32_t scratch_bytes;       // this stage's local-memory requirement
    uint32_t local_off;           // System-stamped: byte offset of this stage's slice of the
                                  //   launch's local buffer (0 for an individual launch)
    uint8_t  rank;
};

/// FnOffsetInfo — position-as-identity for the chained executor. Every
/// definition in an acpp call sequence receives IDENTICAL arguments (one
/// call site, expanded), so a stage can only know WHICH stage it is by
/// advancing shared mutable state in call order. This is that state.
struct FnOffsetInfo {
    const StageRecord* records;   // device array, execution order
    uint32_t           i;         // per-work-item cursor, starts at 0
    const StageRecord& next() { return records[i++]; }
};

} // namespace naga::cospan
```

### 5.5 The generated device function — one per KernelBody type, at the body's rank

```cpp
namespace naga::cospan {

/// Component::compile() (§6) emits, for KB at launch rank R:
SYCL_EXTERNAL void kb_fn(sycl::nd_item<R> it, FnOffsetInfo& info,
                         const std::byte* inputs, std::byte* outputs,
                         const std::byte* bodies, std::byte* local_base)
{
    // LAW: THE CURSOR ADVANCE IS THE FIRST STATEMENT AND UNCONDITIONAL.
    // Every stage function consumes exactly one record before any
    // possible return; a masked-out stage still advances. Violating
    // this desynchronizes every stage after it. Why: §1 need 2.2 fuse
    // (the executor exists for fusion; opacity forbids stage ids).
    const StageRecord& r = info.next();

    // Group-index slice check — group-uniform by construction: whole
    // groups skip together; SYCL group barriers are group-scoped, so
    // skipped groups violate nothing (verified: SYCL 2020 §4.17.3 and
    // the execution model's weakly-parallel work-item guarantee).
    const size_t glid = it.get_group_linear_id();
    if (glid < r.group_offset || glid - r.group_offset >= r.group_count)
        return;

    // Stage-local flat index — contiguous per stage under any geometry:
    const size_t flat = (glid - r.group_offset) * it.get_local_range().size()
                      + it.get_local_linear_id();

    // id-only bodies: tail mask HERE. The body sees only its logical
    // range: ZERO guards in user code, fusion invisible (the opacity
    // contract — kernels are never written differently for fusion).
    if constexpr (PlainBody<KB>)
        if (flat >= r.flat_extent) return;
    // GroupItem bodies: receive the full padded slice; ragged-tail
    // handling is the standard obligation of any group-using nd_range
    // kernel. Their freedom is bounded by SYCL 2020 §4.17.3, verbatim:
    // group functions "must be encountered in converged control flow by
    // all work-items in the group ... under the same set of conditions"
    // — skipping one via early exit, or calling it from divergent
    // branches, is UB. Divergence is legal in code free of group
    // functions. Guard the loads, never the participation. We spec to
    // the spec, not to hardware tolerance. Why: §1 need 2.1 group-level
    // parallelism, §1 need 2.2 opacity (kernels are never written for fusion).

    GroupItem<kernel_rank_v<KB>> gi = detail::ItemFixup::make(it, r);   // stage-local view (§5.3)
    const auto idx = detail::delinearize<kernel_rank_v<KB>>(flat, r.range);
    auto in   = *reinterpret_cast<const typename KB::Inputs*>(inputs  + r.input_off);
    auto out  = *reinterpret_cast<const typename KB::Outputs*>(outputs + r.output_off);
    auto body = *reinterpret_cast<const KB*>(bodies + r.body_off);

    if constexpr (HasScratch<KB>) {
        auto s = *reinterpret_cast<const typename KB::Scratch*>(bodies + r.scratch_off);
        detail::fixup_scratch(s, local_base + r.local_off);  // Local.ptr_ = slice base + offset_
        body(gi, in, out, s);
    } else if constexpr (GroupBody<KB>) {
        body(gi, in, out);
    } else {
        body(idx, in, out);
    }
}

} // namespace naga::cospan
```

### 5.6 The chained executor

```cpp
namespace naga::cospan {

/// Placeholders — one per launch rank; declarations only, JIT replaces.
/// Three families cost nothing at rest: SSCP lowers a codepath on first
/// invocation only.
void kb_entry_r1(sycl::nd_item<1>, FnOffsetInfo&, const std::byte*,
                 std::byte*, const std::byte*, std::byte*);
void kb_entry_r2(sycl::nd_item<2>, /* same trailing params */);
void kb_entry_r3(sycl::nd_item<3>, /* same trailing params */);

/// The outer kernel, one per rank — THE single call site:
q.submit([&](sycl::handler& cgh){
    sycl::local_accessor<std::byte, 1> local{launch_scratch_bytes, cgh};
                                 // sequential wave: the sum of the stages' slices;
                                 // parallel wave: the largest stage's; zero when unused
    cgh.parallel_for(sycl::nd_range<R>{padded_global, wg},
        cfg.apply([=](sycl::nd_item<R> it){
            FnOffsetInfo info{records_dev, 0};      // fresh per work-item
            kb_entry_rR(it, info, inputs_dev, outputs_dev, bodies_dev,
                        local.get_pointer().get());
        }));
});

/// MECHANISM. cfg is built ONCE, in the compile step — there is NO
/// runtime config cache: an individual component's config
/// (define(entry_id, fn_id)) never changes, and a FusedSystem's sequence
/// config is built in ITS compile because fuse membership is static.
///   individual:  cfg.define(kb_entry_rR_id, kb_fn_id);
///   fused wave:  cfg.define_as_call_sequence(kb_entry_rR_id,
///                    {fn_A_id, fn_B_id, fn_C_id, ...});
/// At JIT time the ONE kb_entry_rR call expands, inline and in order:
///     kb_fn_A(it, info, inputs, outputs, bodies, local);
///     kb_fn_B(it, info, inputs, outputs, bodies, local);
///     ...
/// Every expanded call receives the SAME arguments — acpp's sequence
/// semantics. `info` is a mutable reference to the work-item's private
/// cursor: A's unconditional first-statement next() returns records[0]
/// and leaves i==1; B's returns records[1]; and so on.
///     CALL ORDER = SEQUENCE ORDER = RECORD ORDER = PACK ORDER
/// — one identity, maintained by the System's packer writing records in
/// execution order and by the first-statement law. No stage ids exist;
/// the same KernelBody type twice in one wave reads two different
/// records correctly by position alone.
///
/// No barriers: the framework inserts none, in either mode, and the
/// sequence holds only stage functions, one record each. Local memory is
/// laid out so that none is needed for scratch:
///   * sequential wave — every scratch stage has its OWN slice of the
///     local buffer, at StageRecord::local_off (the running sum of the
///     earlier stages' scratch bytes, each rounded up to 16 bytes); the
///     launch's local size is the sum, so a stage never touches bytes
///     another stage uses.
///   * parallel wave — each group executes exactly one stage's body, so
///     the stages share one slice at offset 0 and the launch's local
///     size is the largest stage's.
/// A local size above the device's local memory limit throws at forward,
/// naming the scratch stages and their byte counts (§7.5). A read of
/// another work-item's write, inside a stage or across stages of a
/// sequential wave, needs the body's own group barrier (§5.3).
/// Why: §1 need 2.2 fuse, §1 need 2.1 group-level parallelism.

} // namespace naga::cospan
```

### 5.7 Geometry and slices

- All launches are typed by rank (the executor's switch); the default
  geometry is flat: `{{padded_flat}, {default_wg}}` at rank 1, with
  rank-appropriate defaults at ranks 2/3 (implementation-tunable, not
  contract constants). The wg overloads of create_launch_config (§6)
  override per launch; a wave has ONE work-group shape: the
  FusedSystem's own, else the shape its stages request in common (a
  stage asking for a different one throws at forward), else the default
  for the wave's rank; the wave's rank is its children's.
- Rank agreement in a fuse: a stage whose kernel rank differs from the
  wave's rank is a hard error, checked at compile (the children's
  kernel ranks) and on the descriptions each forward (§7.5).
- Padding and extents are counted in WHOLE WORK-GROUPS: individual = the
  stage's own group count; fused-sequential = max stage group count;
  fused-parallel = Σ stage group counts with group_offset the prefix
  sum. A group never straddles stages, under any geometry — this is
  what makes the slice check group-uniform and collectives safe.
- One dynamic_function_config per launch; the config object outlives
  kernel completion; acpp JIT-caches per configuration, persistently
  across runs (acpp laws).

### 5.8 Dynamic shapes and the pipeline

- **Resolution is just-in-time.** A Component resolves shapes in
  run_pre_compute at its own turn in depth-first order — upstream has
  completed by construction. Static models pay nothing anywhere in this
  section.
- **FromInput applies to a Component's own inputs and outputs.** In its
  `run_pre_compute`, the Component reads the driving input's value on the
  host (the one forced sync, so the launch geometry is known at enqueue),
  calls `root.resize(shape)`, then `point_wrapper` — no staging copy,
  nothing submitted for the resize, no reallocation inside a submitted
  task. Nothing is registered anywhere; no producer knows its consumers.
- **FusedSystem: a value-driven size may not come from the same fuse.** A
  wave's geometry needs every stage's extent before the wave launches, and
  a `FromInput` size is read from the driving input's VALUE on the host
  before that launch. An entry of a stage with a `FromInput` dimension
  whose driving input aliases an output of another stage of the same fuse
  is therefore an impossibility, not a trust question: `compile()` throws,
  naming the entry, the driving input and the stage that produces it, in
  both modes. Shape-from-shape links (`FromInputDim`, an unspecified dim,
  through connections) are always allowed: they read an extent, which is
  known when the stage's turn comes, not a value a stage computes.
  USERS SPLIT THEIR FUSEDSYSTEMS WHERE A VALUE MUST BE COMPUTED FIRST.
  Why: §1 need 2.2 fuse, §1 need 2.1 shape source.
- **ParallelSystem: no sibling-shape check.** The framework checks
  nothing about children's shapes; its only wiring rules are those of
  §7.5 (no connections between children; tied outputs need
  `allow_tied_outputs()`). General data-independence of parallel children
  remains the user's responsibility. Why: §1 need 2.2 parallel · P4.

---

## 6. Subsystem and Component

Binding contract; declarations only. Wrappers: §2. Metadata/shapes: §3.
Kernel machinery: §5. System: §7. The end-to-end lifecycle: §11.

### 6.0 The functional model

A model is a function from its free inputs to its outputs, and a component
is one step of it:

- **Outputs are always computed fresh from inputs.** No component may
  assume an initial output value, and outputs are not initializable. Only
  a free input root has an initial value (§4.4).
- **There is no global scratch.** Multi-step work is several components,
  producer to consumer; an intermediate is one component's output and the
  next one's input, given a finite run policy (§4.4) when it should not outlive its last
  user.
- **Accumulation takes one of two forms.** Either group work with local
  memory inside one kernel (§5.2), or a zeroing producer component
  followed by the accumulating components, all joined by `tie_outputs`
  (§4.3), whose default stays `require_atomic` with `allow_plain` as the
  user's opt-out.
- **Atomic accesses are relaxed** (§2). A kernel that needs acquire/release
  ordering uses `sycl::atomic_fence(sycl::memory_order::acq_rel,
  sycl::memory_scope::device)`, which SYCL 2020 guarantees on every device.

Why: §1 need 2.1 possibility floor (the general map) · P1 · P3.

### 6.1 Subsystem — the ONE contract between an owner and what it owns

A System owns Subsystems and never learns which kind they are. Everything
a System calls on a child is declared here; everything a Component does
in response is the private body of its own runner overrides.

```cpp
namespace naga::cospan {

/// Framework verbs (runners). Default implementations forward to the
/// user hooks; framework-provided types (Component, System and its
/// flavors) override runners as they need; users override HOOKS. Runners
/// are virtual so authors of new System types can override them — that
/// is the extensibility contract — but a typical model never does.
/// Why: §1 need 2.2 possibility floor · P5.
class Subsystem {
public:
    virtual ~Subsystem();
    std::string_view name() const;   ///< in the owner's namespace
    std::string      path() const;   ///< dotted from the root
    const System*    owner() const;  ///< null at the root

    // ---- setup chain (Model drives; §11) ----------------------------------
    /// Declare structure and I/O. No device, no allocation.
    virtual void run_setup() = 0;
    /// Decide/inherit device, account static bytes. Returns
    /// device -> bytes for this subtree (merged upward). §6.4 for the
    /// per-entry logic.
    virtual std::unordered_map<sycl::device, size_t>
        get_exec_info(const sycl::device& parent_device) = 0;
    /// Bind the substrates, apply the shape updates that need no value, and point the
    /// wrappers of members whose root has storage; Systems complete their ExecCtx.
    /// Nothing is allocated here: a variable is materialized on need in a forward (§4.4).
    virtual void prepare_exec(sk::queue& queue, Substrates& substrates) = 0;
    /// Compile bodies (Components); recurse (Systems). A FusedSystem also
    /// checks its structure here (children, ranks, value-driven sizes);
    /// the rest of a flavor's legality is checked per forward on the
    /// descriptions (§7.5).
    virtual void compile() = 0;

    // ---- forward (per forward; §6.5) ------------------------------------
    virtual void run_pre_compute(ExecCtx& owner_ctx) = 0;
    /// Components: return a description of their work; they never
    /// submit it. Systems: submit their subtree's work themselves and
    /// return null — which tells the owner "submitted (or chose not to);
    /// nothing for you to do". Why: §1 need 2.2 opacity · P5
    /// (orchestration is a System's; a Component only works).
    virtual std::unique_ptr<LaunchDescription> run_compute_forward() = 0;
    virtual void run_post_compute(ExecCtx& owner_ctx) = 0;
};

/// What a Component's compute turn yields. The owner ROUTES on
/// is_custom_op() and is_host(): kernels take the outer-kernel path (§5.6),
/// custom operations take AdaptiveCpp_enqueue_custom_operation through the
/// same SubmitTarget, host bodies take an async host task (§5.1). A kernel
/// description additionally exposes rank,
/// group-body and scratch facts and its streams/record, for fused owners
/// to validate and pack.
class LaunchDescription {
public:
    virtual ~LaunchDescription();
    virtual bool is_custom_op() const = 0;
    virtual bool is_host() const = 0;
    virtual void submit(SubmitTarget& t) const = 0;   ///< ordering is the stream's (§7.3)
    virtual const KernelParts* kernel() const { return nullptr; }  // null for custom ops and host bodies
};

} // namespace naga::cospan
```

### 6.2 ComponentBase — the user surface

```cpp
namespace naga::cospan {

class ComponentBase : public Subsystem {
public:
    ComponentBase(const ComponentBase&) = delete;
    ComponentBase& operator=(const ComponentBase&) = delete;

    // ---- HOOKS (user-implemented) ----------------------------------------
    /// Declare I/O. Runs once, during the declare pass. ZERO allocation,
    /// no device, no queue.
    virtual void setup() = 0;
    /// There is no init hook: a free input's initial value is declared in
    /// setup() with init_input (below, §4.4), and outputs are never
    /// initialized (§6.0).
    /// Per-forward manual shape control of a subset of entries; runs
    /// inside the automatic pre-compute after automatic derivation.
    /// Default no-op.
    virtual void pre_forward() {}

    // ---- declaration API (semantics: §3; PUBLIC, OpenMDAO-style) ------
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_input(Input<DT, Rank>& m, std::string name, Specs... specs);
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_output(Output<DT, Rank>& m, std::string name, Specs... specs);
    template <DType DT, int Rank>
    void declare_input(Input<DT, Rank>& m, std::string name, from_input_array_t);
    template <DType DT, int Rank>
    void declare_output(Output<DT, Rank>& m, std::string name, from_input_array_t);

    /// pre_forward-slot-only resize; same setter as automatic derivation.
    template <DType DT, int Rank> void reshape(Input<DT, Rank>& m, sycl::range<Rank>);
    template <DType DT, int Rank> void reshape(Output<DT, Rank>& m, sycl::range<Rank>);

    // ---- retrieval & enumeration ------------------------------------------
    template <DType DT, int Rank> Input<DT, Rank>&  get_input(std::string_view name);
    template <DType DT, int Rank> Output<DT, Rank>& get_output(std::string_view name);
    auto get_inputs() const;    ///< range of const detail::IOEntry& (§3.3 guarantee;
                                ///  why: §1 need 2.3 host access)
    auto get_outputs() const;

    // ---- initial values (declared in setup(); else throw; §4.4) -------------
    /// Declare the initial value of one of this Component's OWN inputs, by
    /// name: cs::init::fill(v), zero(), from_host(vector), custom(fn) or
    /// custom_host(fn). Metadata on the input's root, applied by the
    /// framework when the root is allocated from nothing; the highest level
    /// that declares one wins, and it is ignored when the input ends up with
    /// a source — so a reusable component can carry a default for the case
    /// it is left unconnected. An input is the only thing it applies to:
    /// naming an output throws at setup. Components have no init() hook and
    /// no initializer helpers.
    /// Why: §1 need 2.3 initialization · P2 (an adopted subsystem's
    /// defaults must not need editing when its inputs get connected).
    void init_input(std::string_view name, InitSpec init);

    // ---- body facts, read off the body's signature -------------------------
    /// Answered by Component<KB>; a fused owner holds type-erased children
    /// and plans a wave from these without knowing the body types.
    virtual int  kernel_rank() const;      ///< 0 for custom-op and host bodies
    virtual bool is_group_body() const;
    virtual bool has_scratch() const;
    virtual bool is_custom_op() const;
    virtual bool is_host() const;
    /// The type-erased id of the body's stage function; throws for a
    /// custom-op or host body (nothing is compiled).
    virtual detail::JitId definition_id() const;

protected:
    /// The storage kind of this Component's I/O: device, or host for a host body.
    virtual sk::kind default_storage_kind() const;
    std::unordered_map<std::string, std::unique_ptr<detail::IOEntry>> by_name_;
    std::unordered_map<const void*, detail::IOEntry*>                 by_member_;
    // Runner overrides (Subsystem verbs) are implemented by Component<KB>;
    // their bodies are private to it.
};

} // namespace naga::cospan
```

### 6.3 Component&lt;KB&gt; — the typed layer

```cpp
namespace naga::cospan {

/// One KernelBody type per component: the one-kernel rule, by type.
template <KernelBodyC KB>
class Component : public ComponentBase {
public:
    /// USER, every forward: assemble Inputs/Outputs (and body config, and
    /// Scratch shapes) from the already-synced members; state the launch
    /// parameters. Returns the description the owner will submit.
    virtual std::unique_ptr<LaunchDescription> compute_forward() = 0;

protected:
    /// Base-provided staging (launch_rank_v<KB> is kernel_rank_v<KB>, or 1
    /// for a body with no rank). Kernel bodies: range[, wg], Inputs,
    /// Outputs [, Scratch], body. Custom-op and host bodies (§5.1): Inputs,
    /// Outputs, body — no range.
    std::unique_ptr<LaunchDescription> create_launch_config(
        sycl::range<launch_rank_v<KB>> range,
        typename KB::Inputs in, typename KB::Outputs out, KB body = {});
    std::unique_ptr<LaunchDescription> create_launch_config(
        sycl::range<launch_rank_v<KB>> range, sycl::range<launch_rank_v<KB>> wg,
        typename KB::Inputs in, typename KB::Outputs out, KB body = {})
        requires GroupBody<KB>;
    std::unique_ptr<LaunchDescription> create_launch_config(
        sycl::range<launch_rank_v<KB>> range, sycl::range<launch_rank_v<KB>> wg,
        typename KB::Inputs in, typename KB::Outputs out,
        typename KB::Scratch scratch, KB body = {})
        requires(GroupBody<KB> && HasScratch<KB>);
    std::unique_ptr<LaunchDescription> create_launch_config(
        typename KB::Inputs in, typename KB::Outputs out, KB body = {})
        requires CustomOpBody<KB>;
    std::unique_ptr<LaunchDescription> create_launch_config(
        typename KB::Inputs in, typename KB::Outputs out, KB body = {})
        requires HostBody<KB>;
    /// The storage kind of this Component's I/O: device kind, or host kind
    /// for a host body.
    sk::kind default_storage_kind() const override;

private:
    // ---- runner overrides (the private bodies of the Subsystem verbs) ---
    void run_setup() override;                       // -> setup()
    std::unordered_map<sycl::device, size_t>
        get_exec_info(const sycl::device& parent) override;   // §6.4
    void prepare_exec(sk::queue&, Substrates&) override;      // §6.4
    void compile() override;
    void run_pre_compute(ExecCtx&) override;         // §6.5
    std::unique_ptr<LaunchDescription> run_compute_forward() override; // -> compute_forward()
    void run_post_compute(ExecCtx&) override;        // §6.5

    CompiledKernelMeta meta_;   // static after compile (§5)
};

} // namespace naga::cospan
```

### 6.4 A Component in the setup chain

- **`run_setup`**: `setup()`. Entries exist; sets are formed by the owner's
  structural calls as they happen.
- **`get_exec_info(parent_device)`** — a Component has no device choice;
  it records `parent_device`, then for each own entry `e` sets `e`'s
  placement to (`parent_device`, the Component's default storage kind —
  device kind, or host kind for a host body):
  - `e` is root: register this placement as its primary
    (`set_primary_placement`: it goes first even when an input peer's
    owner registered before it); count its static bytes here (zero for a
    root that is not arena-eligible, and zero for a host-kind primary).
  - `e` is an input peer: if `root(e)` already has storage registered at
    this placement → nothing; else register this placement on `root(e)`
    and count the root's static bytes here — a device-kind placement adds
    them once per device, a host-kind one or a root that is not arena-eligible adds none
    (the root owns one copy per placement its set spans — §4.2).
  - `e` is an output peer (only a tied output is one): `root(e)`'s
    placement must equal this placement, else ERROR naming the tie —
    producers cannot write one storage from two placements.
  - Every root is processed before any peer of it: Systems process their alias entries before their children, and producers precede consumers in registration order. A cycle is expressible by registration order: without solvers it produces the setup warning of §4.5 and runs with the consumer reading the producer's previous value; solvers are future work.
  - Return `{device: bytes}`.
- **`prepare_exec`**: capture the queue and substrates, bind the substrates
  to every own root, point the wrapper of every member whose root has
  storage, and validate static input peers against their roots. NOTHING IS
  ALLOCATED here: a variable is materialized on need, in a forward (§4.4, §6.5). A free input with an unspecified dim and no initializer that gives it a shape is the setup error of §3.6, raised by the resolution pass.
- **`compile`**: the device function at the body's own rank (`kernel_rank_v<KB>`);
  the individual config; a custom-op or host body compiles nothing. A
  Component knows nothing of fusing.

### 6.5 A Component's forward turn

Called by its owner in this order; a Component never submits its compute
work and never sees a queue except through the owner's ctx.

1. **`run_pre_compute(ctx)`**: value-driven shapes first, over every own
   entry — inputs, then outputs, with cycle detection between the specs
   (`FromInputDim` reads the sibling input's extent) — for each entry
   carrying a `FromInput` spec, read the driving input's value
   on the host, whichever storage holds it (the
   one forced sync, so this turn's launch geometry is known; integer
    dtypes only, a negative value throws), materialize the entry's variable first if it
    is not yet live, call
   `root.resize(shape)`, then `point_wrapper`; no staging
   copy, nothing submitted for the resize, no reallocation inside a
   submitted task (why: §1 need 2.1 shape source · P3 — only value-driven entries pay
   it). Then shape-derived dims, `pre_forward()` hook (per-entry
   override, last write wins), dynamic connection validation, wrapper
   sync (a root that grew was reallocated by its own allocator route).
    Then every variable this Component has a port on is MATERIALIZED if it is not yet live
    (§4.4): the storage of every placement with ports is allocated at once (route and current
    extents as in §4.4), filled by copying from a live placement when the variable is live
    elsewhere, and otherwise initialized from its init when it is a free input (zeros when
    none). The Component's ports share their placement's storage, and it points its wrappers at
    it. A variable that is already live is left alone, so a value-driven resize keeps its
    content; a pending `Model::init_input` is applied to the live storage at this variable's
    next run. The queue orders the init before this turn's launch. A shape that is needed and
    not yet known throws here, naming the entry, the dimension and what it was waiting on
    (§4.5). Specified dims of connected and tied entries are checked against their supplier
    (§3.6).
2. **`run_compute_forward()`**: `compute_forward()` — returns the
   description. The owner submits it (§7.3).
3. **`run_post_compute(ctx)`**: through `ctx`, after the owner's
   submission and after the launch's kernel events have been registered on the
   storage of every port this Component touched (§4.4): the producer push — for each
   output variable with storage at other placements, for each other placement
   `copy_d = root` — one assignment resizes to match and copies, routed by SlabKit.
   Ordered after the work it depends on by the queue and the registered events
   (§7.3). Nothing is released here: storage goes when the last port holding it is
   deleted (§4.4).

### 6.6 Provided kernels — v1

```cpp
namespace naga::cospan::kernels {
template <DType DT, class Op = sycl::plus<cpp_type_t<DT>>>
struct GroupReduce {      // fast; reduce_over_group + leader atomic; its `out` is tied (tie_outputs) with a
                          // zeroing producer that runs before it (§6.0), because outputs are never initialized
    struct Inputs  { Input<DT, 1> in; };
    struct Outputs { Output<to_atomic_v<DT>, 1> out; };
    void operator()(GroupItem<1>, Inputs, Outputs) const;
};
template <DType DT, class Op = sycl::plus<cpp_type_t<DT>>>
struct OrderedReduce {    // deterministic; one agent, ascending order; fuses anywhere
    struct Inputs  { Input<DT, 1> in; };
    struct Outputs { Output<DT, 1> out; };
    void operator()(sycl::id<1>, Inputs, Outputs) const;
};
/// dst[offset + i] = src[map[i]]. Launch range = map's range. `map` is any integer-dtype I/O.
template <DType DT, DType IDX, int Rank>
struct GatherInto {
    struct Inputs  { Input<DT, 1> src; Input<IDX, Rank> map; };
    struct Outputs { Output<DT, Rank> out; };
    sycl::id<Rank> dst_offset;
    void operator()(sycl::id<Rank> i, Inputs a, Outputs o) const
        { o.out[i + dst_offset] = a.src[a.map[i]]; }
};

/// Scatter — the additive reverse map, out[map[i]] += in[i]; output
/// atomic BY TYPE (the general-case floor). A provided kernel users compose.
template <DType DT, DType IDX, int Rank>
struct Scatter {
    struct Inputs  { Input<DT, Rank> in; Input<IDX, Rank> map; };
    struct Outputs { Output<to_atomic_v<DT>, 1> out; };
    void operator()(sycl::id<Rank> i, Inputs a, Outputs o) const
        { o.out[a.map[i]] += a.in[i]; }
};
}
```

### 6.7 End-to-end components

```cpp
// (a) Dynamic + value-driven + atomic; id-only body.
struct HistBody {
    struct Inputs  { Input<DType::f32,1> values; Input<DType::f32,1> weights;
                     Input<DType::u32,1> n_bins; };
    struct Outputs { Output<DType::atomic_f32,1> bins; };
    float lo, hi;
    void operator()(sycl::id<1> i, Inputs in, Outputs out) const {
        auto b = static_cast<size_t>((in.values[i] - lo) / (hi - lo)
                                     * static_cast<float>(in.n_bins[0]));
        out.bins[b] += in.weights[i];
    }
};
class WeightedHistogram : public Component<HistBody> {
    Input<DType::f32,1> values_, weights_;  Input<DType::u32,1> n_bins_;
    Output<DType::atomic_f32,1> bins_;  float lo_, hi_;
public:
    WeightedHistogram(float lo, float hi) : lo_{lo}, hi_{hi} {}
    void setup() override {
        declare_input (values_, "values",  unspecified);
        declare_input (weights_,"weights", from_input_shape<1>("values"));
        declare_input (n_bins_, "n_bins",  unspecified_shape<1>());   // connected from NBins
        declare_output(bins_,   "bins",    unspecified_shape<1>());   // a tie_outputs member: the tie supplies the shape
    }
    std::unique_ptr<LaunchDescription> compute_forward() override {
        return create_launch_config(values_.shape(),
            {values_, weights_, n_bins_}, {bins_}, {lo_, hi_});
    }
};

// (a') The zeroing producer. Outputs are never initialized (§6.0), so what a
// histogram accumulates onto is the output of a component that computes zero.
struct ZeroBody {
    struct Inputs  {};
    struct Outputs { Output<DType::atomic_f32,1> bins; };
    void operator()(sycl::id<1> i, Inputs, Outputs out) const { out.bins[i].store(0.0f); }
};
class ZeroBins : public Component<ZeroBody> {
    Output<DType::atomic_f32,1> bins_;
public:
    void setup() override { declare_output(bins_, "bins", unspecified_shape<1>()); }   // the tie alias owns its shape
    std::unique_ptr<LaunchDescription> compute_forward() override
        { return create_launch_config(bins_.shape(), {}, {bins_}); }
};

// NBins: the bin count as an OUTPUT, n[0] = v[0] (v is the free input a user drives).
class NBins : public Component<NBinsBody> { /* declare_input(v_, "v", 1); declare_output(n_, "n", 1); */ };

class Histogram : public System {
public:
    void setup() override {
        add_subsystem<NBins>("nb");                                // BEFORE the tie's members
        add_subsystem<ZeroBins>("zero");                           // BEFORE hist: registration order is execution order
        add_subsystem<WeightedHistogram>("hist", 0.0f, 1.0f);
        connect("nb.n", "hist.n_bins");
        // bins' shape is the VALUE of nb.n[0], read each run by this System when the tie is accessed, just before zero; atomic by default
        tie_outputs<DType::atomic_f32, 1>({"zero.bins", "hist.bins"}, "bins", alias::from_output_values("nb.n"));
        promote<DType::u32, 1>("nb.v", "n_bins");
    }
};
```

The zeroing producer runs first in registration order, the histogram
accumulates atomically onto the tied root, and nothing is initialized.

```cpp

// (b) Group + wg override + Local scratch; GroupItem body.
struct TransposeBody {
    struct Inputs  { Input<DType::f32,2> in; };
    struct Outputs { Output<DType::f32,2> out; };
    struct Scratch { Local<DType::f32,2> tile; };
    void operator()(GroupItem<2> it, Inputs in, Outputs out, Scratch s) const {
        const auto l = it.get_local_id(), g = it.get_global_id();
        const bool live = g[0] < in.in.shape()[0] && g[1] < in.in.shape()[1];
        if (live) s.tile[{l[0], l[1]}] = in.in[{g[0], g[1]}];   // guard loads
        sycl::group_barrier(it.get_group());                    // never participation
        const auto gt = sycl::id<2>{ it.get_group(1) * it.get_local_range(1) + l[0],
                                     it.get_group(0) * it.get_local_range(0) + l[1] };
        if (gt[0] < out.out.shape()[0] && gt[1] < out.out.shape()[1])
            out.out[gt] = s.tile[{l[1], l[0]}];
    }
};
class TiledTranspose : public Component<TransposeBody> {
    Input<DType::f32,2> in_;  Output<DType::f32,2> out_;  static constexpr size_t WG = 16;
public:
    void setup() override {
        declare_input (in_,  "in",  unspecified_shape<2>());
        declare_output(out_, "out", FromInputDim{"in",1}, FromInputDim{"in",0});
    }
    std::unique_ptr<LaunchDescription> compute_forward() override {
        TransposeBody::Scratch s;  s.tile.reshape({WG, WG + 1});
        return create_launch_config(in_.shape(), {WG, WG}, {in_}, {out_}, s, {});
    }
};

// (c) A custom operation: hand device pointers to a native library.
struct GemvBody {
    struct Inputs  { Input<DType::f32,2> A; Input<DType::f32,1> x; };
    struct Outputs { Output<DType::f32,1> y; };
    void operator()(CustomOp& op, Inputs in, Outputs out) const {
        native_gemv(op.h /* interop handle: native stream */,
                    op.device_ptr(in.A), op.device_ptr(in.x), op.device_ptr(out.y),
                    in.A.shape()[0], in.A.shape()[1]);
    }
};
class Gemv : public Component<GemvBody> {
    Input<DType::f32,2> A_;  Input<DType::f32,1> x_;  Output<DType::f32,1> y_;
public:
    void setup() override {
        declare_input (A_, "A", unspecified_shape<2>());
        declare_input (x_, "x", FromInputDim{"A", 1});
        declare_output(y_, "y", FromInputDim{"A", 0});
    }
    std::unique_ptr<LaunchDescription> compute_forward() override
        { return create_launch_config({A_, x_}, {y_}); }
};
```

None of these contains a guard for machinery, a queue, a submission, an
allocation, or a pointer the user manages. `setup()`, `compute_forward()` —
and physics.

---

## 7. System: execution context, orchestration, structural APIs

Binding contract; declarations only. Subsystem verbs and Component: §6.
Aliasing mechanics: §4. Topology semantics: §8. Flavors beyond
Parallel/Fused: §10. The end-to-end lifecycle: §11.

### 7.1 ExecCtx — System-owned execution state

```cpp
namespace naga::cospan {

/// Every System owns exactly one. Populated across the setup chain
/// (device in get_exec_info; queue, group and substrate in prepare_exec)
/// and never handed to anyone: children receive a reference only to USE
/// it — to allocate, to look up the device, to submit copies in
/// post-compute. Nothing in the tree ever reads another System's ctx.
/// ExecCtx carries no events: ordering is the stream's (§7.3).
/// Why: P5 (ordering is the stream's, never passed) · P3.
struct ExecCtx {
    sycl::device       device;     ///< this System's choice (own or inherited)
    sk::queue*         queue;      ///< the Model's ONE queue
    sk::queue::group*  group;      ///< non-null only under a Parallel owner
    MemorySubstrate*   substrate;  ///< this device's arena (+ pool)

    /// SubmitTarget view: what a LaunchDescription::submit consumes.
    SubmitTarget target();
};

/// One per device (SlabKit spec): the device's arena (present iff it has static bytes) and pool (present iff configured),
/// built from (queue, device) and held for the model's life. Every
/// allocation a subsystem makes goes through its device's substrate.
struct MemorySubstrate { std::optional<slabkit::DeviceArena> arena;   // present iff the device has static bytes
                         std::optional<slabkit::DevicePool>  pool; };  // present iff configured (§9)
using  Substrates = std::unordered_map<sycl::device, MemorySubstrate>;

/// What a submission consumes (§6.1 LaunchDescription::submit): the
/// queue, the group when the submission belongs to a Parallel owner's
/// group, and the device it targets.
struct SubmitTarget { sk::queue* queue; sk::queue::group* group;
                      sycl::device device; };
/// What a fused owner reads off a kernel description (§6.1): the facts it
/// validates and the parts it packs.
struct KernelParts { int rank; bool group_body; bool has_scratch; StageRecord record;
                     std::span<const std::byte> inputs, outputs, body, scratch;
                     std::optional<sycl::range<3>> wg; };
// the trivially-copyable structs as bytes; group_* fields stamped by the fused owner

} // namespace naga::cospan
```

### 7.2 System — concrete, orchestration only

```cpp
namespace naga::cospan {

/// A System never does work: it orders, submits and gates the work its
/// children describe. It has no I/O of its own beyond alias entries
/// (§4). `System` is a complete type: a System with no overrides is a
/// plain sequential container (the property-list add_subsystem overload
/// creates exactly that).
class System : public Subsystem {
public:
    // ---- HOOKS ------------------------------------------------------------
    virtual void setup() {}                     ///< structure & wiring (§7.4); there is no init hook (§4.4)

    // ---- Model's entries ------------------------------------------------------
    /// The root System's end-of-declare verb: resolve(), the resolution pass
    /// (§4.4, §11.2.1). Model::forward runs the root's run_pre_compute,
    /// run_compute_forward and run_post_compute (§11.3). A variable's init is
    /// applied at materialization (§4.4).
    void resolve();

    // ---- retrieval & enumeration (hierarchy-aware; §3.3 guarantee) ------------
    template <DType DT, int Rank> Input<DT, Rank>&  get_input (std::string_view path);
    template <DType DT, int Rank> Output<DT, Rank>& get_output(std::string_view path);
    auto get_inputs()  const;   ///< this System's alias entries
    auto get_outputs() const;
    Subsystem* get_subsystem(std::string_view path);          ///< dotted path from the top; null if declared but currently deleted, throws if never declared
    std::span<Subsystem* const> children() const;             ///< declaration order = exec order; Model's walks use this and nothing else

    // ---- runners (Subsystem verbs) — the DEFAULT orchestration ----------------
protected:
    /// setup(); children's run_setup fire inside add_subsystem (§7.4).
    void run_setup() override;
    /// ctx.device = own property::device or parent_device. Process OWN
    /// alias entries first (§6.4 logic — a tie alias root counts its
    /// bytes here), then children in order; merge maps (sum on key).
    std::unordered_map<sycl::device, size_t>
        get_exec_info(const sycl::device& parent_device) override;
    /// Complete ctx (queue = the Model's; group = null — only a Parallel
    /// owner sets one; substrate = &substrates[device]); bind the
    /// substrates to its alias roots and point the wrappers of those with storage; then children. Nothing is allocated (§4.4).
    void prepare_exec(sk::queue&, Substrates&) override;
    /// children.compile() in order.
    void compile() override;
    void run_pre_compute(ExecCtx&) override {}     ///< Systems do no work
    void run_post_compute(ExecCtx&) override {}
    /// THE DEFAULT LOOP (§7.3).
    std::unique_ptr<LaunchDescription> run_compute_forward() override;

    /// Run policy is registration metadata this System keeps for each child (§4.4):
    /// the property from add_subsystem, the stored constructor arguments (for the
    /// rebuild of a transient child and for rearm) and a run counter. This System
    /// builds every missing transient child at the start of its forward, runs each
    /// existing child, moves a run_n / do_once counter after the child's turn and deletes
    /// the child when it reaches 0, and deletes every transient child at the end.
    /// The Components under it materialize the variables they access in their own
    /// turns (§6.5); a System materializes the aliases it declares when it reads them.

    ExecCtx ctx_;
    std::vector<std::unique_ptr<Subsystem>> children_;   ///< declared order = exec order; a deleted child keeps its slot (null) and its stored arguments
    std::unordered_map<std::string, std::unique_ptr<detail::IOEntry>> aliases_;
};

} // namespace naga::cospan
```

### 7.3 The ordering contract

The stream is an abstract in-order sequence of work items. A kernel is an
item (a custom operation, submitted through the same `SubmitTarget`, is
an item the same way; a host body is submitted with `async_host_in_order`,
which joins the stream's order as an item — under a Parallel owner's
group, with the group's `async_host`). A group (`queue.submit_group`) is
one item:
unordered inside, ordered both ways outside. Async host and memory tasks
are concurrent with each other but ordered after every kernel/group item
before them and before every one after. Placement never costs the order.
Nothing in CoSpan threads dependencies.

The default loop is per child:
`run_pre_compute → d = run_compute_forward → if d, d->submit(target) →
run_post_compute`; a child that submitted itself needs no owner action.

```
for (child : children_)                       // at the START, before any child's turn
    if (child is transient and deleted) build it;   // re-constructed from its stored arguments (§4.4)
for (child : children_) {
    if (child is deleted) continue;           // declared, currently deleted (§4.4)
    child->run_pre_compute(ctx_);             // a Component materializes the variables it accesses here (§4.4, §6.5)
    auto d = child->run_compute_forward();
    if (d) d->submit(ctx_.target());          // WE submit, routed on d->is_custom_op() / d->is_host()
    register the launch's kernel events on the storage of every port the child touched   // §4.4
    child->run_post_compute(ctx_);            // copies AFTER the work they depend on —
                                              //   ordered by the queue and the registered events
    if (child is run_n or do_once) { move its counter; if (it reaches 0) delete child; }   // right after its turn (§4.4)
}
for (child : children_)                       // at the END, after the last child's turn
    if (child is transient) delete child;     // a transient child exists for one forward of its parent (§4.4)
return nullptr;                               // "I submitted"
```

- **Ordering is the stream's, never passed.** Which queue a System submits
  to is the Model's one queue (`ctx.queue`); the ordering contract above
  is what every submission, host task and copy relies on. The only events
  CoSpan keeps are a launch's kernel events, registered on the storage of
  the ports it touched so that later memory operations and frees are
  ordered after them (§4.4); no wait list or dependency snapshot passes
  between items; a child that submitted for itself, or is deleted, leaves
  the parent nothing to refresh.
- **Devices are per submission.** Every submission carries its
  `SubmitTarget::device`; the stream keeps its order across devices.
  Nobody creates queues but the Model.
  Why: §1 need 2.2 placement, §1 need 2.5 multi-device · P5.
- **Waiting is the caller's choice**: `Model::wait()` is
  `queue.wait_and_throw()`; a loop of forwards need not wait between
  them (P3).

### 7.4 Structural APIs — all inside `setup()`, else throw (why: P6 — declaration is one phase)

```cpp
namespace naga::cospan {

/// Options are SYCL-style properties: unordered, self-naming, type-checked.
namespace property {
    struct device        { sycl::device dev; };            // placement; inherited when absent
    struct parallel      {};                               // flavor
    struct transient     {};   // run policy: runs every forward; built at the start of its parent's forward, deleted at its end (§4.4)
    struct run_n         { std::size_t n; };   // run policy n >= 1 (n == 0 throws std::invalid_argument)
    struct do_once       {};   // run policy 1; with run_n an error; transient with either is an error
    struct fused         { FusedSystem::Mode mode; std::optional<sycl::range<3>> wg; };
}
// Gating and change detection are custom System TYPES (§10),
// added with the templated overload — not properties.
class property_list { /* SYCL idiom: constructed from any subset; has_property<T>(), get_property<T>() */ };

class System {   // ...continued
public:
    /// A REALIZED System or Component type, optionally with properties
    /// (device).
    template <class T, class... Args> requires std::derived_from<T, Subsystem>
    void add_subsystem(std::string name, Args&&... ctor_args);
    template <class T, class... Args> requires std::derived_from<T, Subsystem>
    void add_subsystem(std::string name, const property_list& props, Args&&... ctor_args);
    /// A child Component to add through the helper form: type, name, ctor args.
    template <class T, class... Args> requires std::derived_from<T, ComponentBase>
    auto child(std::string name, Args&&... ctor_args);
    /// The HELPER FORM: a System configured by properties (device and/or ONE
    /// flavor — parallel | fused; both on one call is an error, and/or a run policy: transient | run_n | do_once), populated
    /// with the given Components in order. No class is written for it. The
    /// children are Components by construction (child<T> requires
    /// ComponentBase), which is what Parallel and Fused require; a helper-
    /// built plain System takes Components too — nest Systems through the
    /// templated overload or a class. Returns nothing: connect/
    /// promote reach the System (typed Parallel/Fused underneath) by name. Why: §1 need 2.2
    /// inline composition · P2 (placement without editing).
    template <class... Children>
    void add_subsystem(std::string name, const property_list& props, Children&&... children);
    void add_subsystem(std::string name, const property_list& props = {});   ///< empty; add children by the templated overload inside setup()
    // add_subsystem stores its constructor arguments (for the rebuild of a
    // transient child and for rearm), constructs the child, registers the name,
    // and runs the child's run_setup BEFORE returning, so later structural calls
    // in this same setup() see it fully declared.
    /// REARM (§4.4): re-construct the named child from the arguments stored at
    /// add_subsystem, run its setup, replay this System's connect, promote, tie_*
    /// and init_input declarations that touch it, and prepare it for execution.
    void rearm(std::string_view child);

    void connect(std::string_view src, std::string_view dst);                 // identity
    void connect(std::string_view src, std::string_view dst, Slice view);   // §8.1: dst sees a rectangular region of src's root
    template <DType DT, int Rank> void promote(std::string_view child_path, std::string promoted_name);
    template <DType DT, int Rank>
    void tie_inputs (std::initializer_list<std::string_view> members, std::string name, AliasShape shape = unspecified);   // unspecified: the shape comes from what connects into it, else the tie's initializer
    template <DType DT, int Rank>
    void tie_outputs(std::initializer_list<std::string_view> members, std::string name, AliasShape shape,
                     TieOutputsPolicy policy = TieOutputsPolicy::require_atomic);
    /// Declare the initial value of a free input root (§4.4): `path` is a
    /// child input path, or one of this System's promoted or tied input
    /// aliases. A System's declaration is at its own depth, so it wins over
    /// any declaration below it and loses to any above it; two at one depth
    /// on one root are an error at setup.
    void init_input(std::string_view path, InitSpec init);
    void add_design_var(...);  void add_constraint(...);  void add_objective(...);   // §8
};

} // namespace naga::cospan
```

Rules (mechanics in §4; semantics in §8): output → input only; an input
has at most one source, enforced at the ROOT's flag after the first
connection; connecting through any name of a set is the same connection;
`promote` is kind-agnostic and always allowed; ties must precede
connections ("tie before connecting, or connect from the alias");
`tie_outputs` requires atomic member dtypes unless `allow_plain`; both
ties take members whose unspecified dims the tie supplies, and an AliasShape (optional for tie_inputs) that owns the
shape (§4.3); `init_input` names an input (an init on an output is an error at setup) and
is ignored when the input's root ends up with a source.

### 7.5 ParallelSystem and FusedSystem — the two policy flavors

```cpp
namespace naga::cospan {

/// CHILDREN ARE COMPONENTS ONLY — enforced by the type system at the add,
/// with the same constraint FusedSystem uses (a System cannot discern
/// children at runtime; why: §1 need 2.2 parallel · P5):
///   template <class T, class... A> requires std::derived_from<T, ComponentBase>
///   void add_subsystem(std::string, A&&...);
/// (property-list overload with `property::device` only: each child may sit
/// on its own device; the group is one item of the stream whichever devices
/// its members target). Nested Parallel is not a concept.
/// run_compute_forward: pre_compute ALL children → queue.submit_group {
/// each child's description submitted with its own device (the Parallel's
/// unless placed) through a SubmitTarget carrying the group } →
/// post_compute ALL after the group returns (so a child's materialization is in
/// its pre_compute and its copies in its post_compute,
/// both outside the group, §4.4). The group is ONE item of the
/// stream (§7.3): its children are unordered among themselves and ordered
/// both ways against everything outside. ctx.group is set for the extent
/// of that submit_group and null otherwise. Data independence of children
/// is the USER's assertion (why: P4); the framework checks two wirings,
/// each forward, on the children's metadata, both `invalid_argument`
/// naming the children involved: NO CONNECTIONS between children (an
/// input whose aliasing set holds another child's output; no override,
/// since the children run unordered there is nothing to read yet), and no
/// tied outputs (two children whose outputs share one root, e.g. through
/// `tie_outputs`) unless the owner calls `allow_tied_outputs()`, the user's
/// assertion that the writers do not race. Children's shapes are not
/// checked against each other.
class ParallelSystem : public System {
public:
    void allow_tied_outputs();
    /* runner overrides as described */
};

/// COMPONENTS ONLY — enforced by the type system at the add, since a
/// System cannot discern children at runtime (why: §1 need 2.2 fuse · P5):
///   template <class T, class... A> requires std::derived_from<T, ComponentBase>
///   void add_subsystem(std::string, A&&...);
/// (no property-list overload). The wave has one work-group shape: the
/// fuse's `wg`, else the stages' common request, else the rank's default.
/// The wave's rank is its children's.
///
/// compile(): every child compiled, then the structure checked, each
/// `invalid_argument` naming the child: at least one child; no custom-op or
/// host child (neither has a stage function); every child on the fuse's
/// device; every child's kernel rank (`kernel_rank_v`, from the body's
/// signature) equal; and no value-driven size (`FromInput`) whose driving
/// input aliases another stage's output in the same fuse, in either mode
/// (shape-from-shape links, `FromInputDim` and an unspecified dim, are always
/// allowed, §5.8). The sequence config is then built once, at the wave's
/// rank, from the children's definition ids in child order.
/// run_compute_forward: pre-compute ALL children → compute ALL (collect
/// descriptions) → VALIDATE on the descriptions, every forward, host-
/// side and cheap: none null, rank agreement, one work-group shape;
/// PARALLEL mode adds: no connections between stages (no override) and
/// no tied outputs unless the fuse calls `allow_tied_outputs()`; SEQUENTIAL
/// mode makes no inter-stage checks — stage ranges may differ, and the
/// access pattern between stages is the user's responsibility (§5.3) →
/// pack streams + records, stamp group slices and local slices, check the
/// wave's local memory against the device's limit, one staging, ONE
/// submission with the sequence config → post-compute ALL (materialization
/// belongs to the children's own pre-compute, §4.4). The
/// framework inserts no barriers, in either mode.
/// Local memory (§5.6): sequential mode gives every scratch stage its own
/// slice (`StageRecord::local_off`, the running sum, each stage's bytes
/// rounded up to 16) and the launch needs the sum; parallel mode shares
/// one slice at offset 0 and needs the largest stage's. A wave over the
/// device's limit throws `invalid_argument` at forward, naming the scratch
/// stages and their byte counts.
/// The order is forced: post-compute copies follow the wave in the stream.
/// Why: §1 need 2.2 fuse, §1 need 2.2 opacity.
class FusedSystem : public System {
public:
    enum class Mode : uint8_t { sequential, parallel };
    explicit FusedSystem(Mode = Mode::sequential, std::optional<sycl::range<3>> wg = {});
    /// Parallel mode: the user's assertion that stages whose outputs share
    /// one root do not race (atomic writers, say); skips the tied-outputs
    /// check. The no-connections rule has no override. No effect in
    /// sequential mode.
    void allow_tied_outputs();
};

} // namespace naga::cospan
```

### 7.6 Sample — Sellar with inline Systems

```cpp
class SellarMDA : public System {
public:
    void setup() override {
        add_subsystem<Discipline1>("d1");
        add_subsystem<Discipline2>("d2");
        connect("d1.y1", "d2.y1_in");
        connect("d2.y2", "d1.y2_in");            // the cycle: the setup warning (§4.5); a solver is future work
        tie_inputs<DType::f64, 1>({"d1.z", "d2.z"}, "z", alias::fixed(2));   // d1.z, d2.z declare unspecified dims
        promote<DType::f64, 1>("d1.y1", "y1");  promote<DType::f64, 1>("d2.y2", "y2");
        init_input("z", init::fill(1.0));        // this level is above d1 and d2: it wins over their defaults
    }
};

class Top : public System {
    void setup() override {
        add_subsystem<SellarMDA>("mda", property_list{property::device{gpu0}});
        add_subsystem<Objective>("obj");
        connect("mda.y1", "obj.y1");
        connect("mda.y2", "obj.y2_head", Slice{subarray<1>({0}, {1})});

        add_subsystem("post", property_list{property::parallel{}}, child<Report>("r1"), child<Report>("r2"));
    }
};
```

---

### 7.7 Chunked systems

`class ChunkedSystem : public System`, constructor `explicit ChunkedSystem(std::size_t budget_bytes)` (0 throws `std::invalid_argument`); users derive and override `setup()` like any System. It runs its subtree once per CHUNK of the leading dimension, so the memory of its internal ports is bounded by the budget. `last_chunk_count()` reports the chunks of the last forward.

**The leading-dimension rule.** The chunked dimension is the leading dimension of every port and is never named. INSIDE a chunked system the FRAMEWORK sets dim 0 of EVERY port of every inside component (inputs and outputs, roots and peers) to the current chunk length (N in the full-range pass, b - a per chunk), whatever rule the component declared for dim 0 (fixed, FromInput, FromInputDim, unspecified; a fixed one is checked once against the full length): the declared rule still applies when the component is used outside a chunked system, inside one it is overridden (`IOEntry::dim0_override_`, consulted by `update_shape` and by the Component's `resolve`, which does not even evaluate a declared FromInput / FromInputDim for dim 0) and the component never knows. N is known before the pass: it is the leading length of the boundary roots that exist outside, or the declared (fixed) leading length of a promoted free input; they must all agree ("boundary roots ... disagree"), and at least one must exist. Dims >= 1 are unchanged: they are resolved from their declared rules, and every dim >= 1 must be resolvable before the chunked system runs, from things crossing its boundary: only VALUE reads from roots computed inside are refused: a FromInput / FromInputValues / alias `output_value(s)` / `input_value(s)` read whose driving root is computed inside the chunked system (an INTERNAL root, or an inside output re-rooted at this level) is an error ("'x' is sized by a value computed inside chunked system '...'; chunked sizes must come from outside"). A value read from a BOUNDARY root (one that exists outside, including through a promoted input of the chunked subtree) is allowed; a per-index array that covers the whole run is cut to the chunk by the internal alias that reads it. SHAPE inheritance from internal entries is allowed (FromInputDim, unspecified dims, connection inheritance, alias `output_dim` / `from_output_shape` / `input_dim` / `from_input_shape`): pricing evaluates those rules, since no value is read. A per-index array driving a dim >= 1 still has length = dim 0, since it is read over the same range.

**Boundary vs internal (classified once, at the end of setup, in `System::resolve`).** For every root touched by an entry declared in the subtree: INTERNAL when every member of its aliasing set is declared inside (a strict descendant) and no promotion or connection reaches it from outside; every port on it is inside, so it lives exactly as long as they do (§4.4). BOUNDARY when it has members inside and outside (a producer outside, a consumer outside, a promotion at this level or above).

**Re-rooting.** A boundary root whose root entry is itself inside (an inside output consumed outside; an inside free input promoted out) is re-rooted: a new alias root is inserted at this system's level (the way `tie_*` aliases are made: `make_alias_entry`, then `IOEntry::merge`), so every inside member becomes a non-root peer and the outside members are peers of the new root. Its dims are inert markers, so it is dynamic; its shape is adopted (`adopt_shape`: dense extents, or N and the per-index arrays) from the inside entry it re-rooted, each forward, after that entry's Component has resolved over the full range. The ChunkedSystem holds its port (not a Component): it materializes the variable at full size before the first chunk, and a free input's init applies there when the variable is live nowhere (`Model::init_input` on the promoted name is applied at the next run, once per call). Marks move to the new root. A boundary root that is already outside needs nothing.

**Per forward.** (1) Full-range pass, host only, nothing materialized, no `pre_forward`: the inside peers of outside boundary roots see their whole root (`LeadingRangeAccess::set(0, N)`), and every inside Component runs `resolve_shapes()` (the shape resolution factored out of `run_pre_compute`) in depth-first registration order. (2) N is the common leading length; every inside entry must agree. (3) Pricing: cost[i] = sum over INTERNAL roots of `sizeof(dtype)` times the block size at i (a ragged root's block, or the product of the dims >= 1), from the declared shape rules only. A greedy scan cuts maximal contiguous chunks with sum <= budget; one index over budget is an error naming the index and its bytes; N = 0 runs nothing (re-rooted outputs are allocated empty). `pre_forward` reshapes inside a chunked system are NOT priced (pre_forward is not run for pricing) and should not be done. (4) Re-rooted roots are materialized (and free-input inits applied when the variable is live nowhere). (5) Per chunk [a, b): every inside peer of every boundary root, inputs and the output peers of re-rooted roots, is pointed at [a, b), then the default loop runs the subtree once: children never see a or b; internal roots are kept across the chunks (Lifetimes, below), and the run counters of everything inside move once, after the last chunk. (6) The ranges are cleared and the shapes restored to the full range, so views from the host see whole roots.

**Lifetimes.** An internal variable holds ONE CHUNK's slice: each chunk is new data in reused memory, so the variable is materialized fresh for every chunk, and an internal FREE INPUT is initialized at the start of every chunk (each chunk is a fresh computation; an init must therefore not depend on the variable's full-run shape). This does not conflict with §4.4's rule that a live variable is never re-initialized: the next chunk's slice is a different piece of data. The memory is reused, not reallocated: before the first chunk every INTERNAL root's storage is allocated ONCE, with capacity for its largest chunk (the maximum over chunks of the sum of its block sizes), and the ports inside hold it, so it is kept across the chunks and across forwards for as long as the chunked system exists; nothing is freed at the end of a forward. Per chunk the root is resized to that chunk's size within the capacity (a `usm_vector` resize within its capacity, up or down, never reallocates; growing zero-fills the new tail on the stream), and a ragged root's small device offset / extent copies are rewritten per chunk in the same storage. On the next forward an internal root whose capacity covers the largest chunk is kept, and a smaller one is replaced by a larger allocation. The run counters of everything inside the chunked system move once per forward, after the last chunk; a child that reaches 0 is deleted by its parent then (§4.4), and when the chunked system itself is deleted its parent takes everything inside with it, and the storage goes with the last port. Ordering is as everywhere (§4.4): the kernel events of each chunk are registered on the storage of every port it touched, the queue orders the next chunk's memory operations after them, and a free waits on them itself; there is no queue-wide wait between chunks or at the end. `IOEntry::allocation_count()` (a test hook) counts a root's storage allocations.

**Opacity.** A chunked system's only interface is its own PROMOTIONS. Inside / outside is decided by where something is DECLARED: everything declared by the chunked system itself or any descendant is inside (its ties, the ties in its own `setup()` included, are internal; its promotions are the boundary ports: promoted inputs are its inputs, each chunk reading its [a, b) slice of whatever is connected or tied outside; promoted outputs are full-size roots written chunk by chunk, still read by their inside consumers as chunk slices); everything declared above it is outside. The single path resolver (`System::entry_at` / `locate`) refuses a path that descends THROUGH a chunked system X into X's children unless X is the caller resolving its own subtree: `'<full path>' lives inside chunked system '<X path>'; only its promotions are visible outside it; promote it to use it here`. This covers connect, tie member paths, alias source paths, promote, init_input, marks and the Model's verbs (a Model resolves from its root as an OUTSIDE caller, via `System::entry_at_from_outside`, so a Model whose root is a chunked system sees its promotions only). A tie declared ABOVE the chunked system whose members are promotions of inside ports with unspecified dims is an outside boundary root: the chunked system evaluates its alias rule (outside sources by construction) before the chunk length and the pass, records the reads as topology dependencies, and its inside members see [a, b). After recording, every dependency of the cache is checked to be an internal root or a boundary root (internal error otherwise). An outside boundary root with an unresolved dimension >= 1 is a setup error.

**Errors.** budget 0 (construction); a nested ChunkedSystem ("nested chunked systems are not supported", at setup); boundary roots of different N; a shape driven by a value computed inside; one index over budget; a chunked system with no port crossing its boundary.

## 8. Topology semantics and optimization marks

Binding contract; declarations only. Aliasing mechanics: §4. Structural
APIs: §7.

### 8.1 The connection spectrum

Two kinds of "dst sees src's data", and they are NOT one mechanism:

- **identity** — `connect(src, dst)`: dst joins the root's set as a peer
  reading the whole root. Zero-copy.
- **slice** — a rectangular region of the root whose flat
  size equals dst's: dst is a peer whose `view()` is that region, in the
  ROOT's rank; the wrapper gets `root.data + offset` and the root's
  strides (§2). Zero-copy. This is what the stride metadata exists for.
  Why (for the whole spectrum): §1 need 2.3 connection.

```cpp
namespace naga::cospan {

/// What `connect` accepts beyond identity.
struct Slice { SubarrayRef src; };   // a view

class System {   // ...continued from §7.4
public:
    void connect(std::string_view src, std::string_view dst);                // identity
    void connect(std::string_view src, std::string_view dst, Slice view);
};

} // namespace naga::cospan
```

### 8.2 View composition

Views compose by offset addition with containment: connecting from a
name whose peer carries view `B`, with a `Slice{A}`, yields
`{B.offset + A.offset, A.extents}` and requires `A ⊆ B` (error naming
both; why: §1 need 2.3 connection · P1). Promoting a peer that carries a view produces an alias that
COPIES the view — the alias presents what the promoted entry sees. All
views are in the root's rank; there is no rank reinterpretation through
a view (rank is authoritative, §3.1).

### 8.3 Cross-level ties

Nothing new: members resolve from the caller's namespace at any depth,
and a member whose root is already a tie alias (a child-level tie) is a
legal member when that alias is unconnected — the merge makes the child
alias a peer of the new root, and the `tie_*` exception marks it like
any named member. Ties layer upward without special rules.

### 8.4 Optimization marks

```cpp
namespace naga::cospan {

/// Per-element or scalar bounds/scaling; host-side vectors sized to the
/// marked region's flat size (scalar forms broadcast).
struct Bounds  { std::optional<std::vector<double>> lower, upper; };
struct Scaling { std::vector<double> ref{1.0}, ref0{0.0}; };   ///< semantics: with the optimizers

class System {   // ...continued
public:
    /// MARKS RIDE ON THE ROOT: marking any name marks the data. A later
    /// mark of the same kind on the same root replaces the earlier one.
    /// Connecting an output into a marked input moves the input into the
    /// output's set, and the mark is gone with the old root: a consumed
    /// System's marks never bind the System that consumes it. Recorded
    /// on the root's IOEntry (§3.3 design_var()/…), enumerated by
    /// Model. A marked root is eternal (§4.4): a
    /// marked root must outlive a forward. `region` restricts a mark to a subarray (root rank) via
    /// the same view machinery. Why: §1 need
    /// 2.4 design variables, constraints, objectives.

    /// Root must be an INPUT root that is not connected — a free input;
    /// the optimizer is its producer. Dynamically shaped DVs are
    /// ill-formed (§3.3). A connected input errors: "connected inputs
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
/// Collected once, at the end of Model::setup(), from every root entry (§9).
class Model {   // ...continued from §9
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

### 8.5 Samples

```cpp
// Sellar's marks.
add_design_var("mda.z", Bounds{.lower = {{-10, 0}}, .upper = {{10, 10}}});
add_design_var("mda.x", Bounds{.lower = {{0}},      .upper = {{10}}});
add_objective ("obj.f");
add_constraint("obj.g1", Bounds{.upper = {{0}}});
add_constraint("obj.g2", Bounds{.upper = {{0}}});
```

---

## 9. Model

Binding contract; declarations only. Memory objects: SlabKit's spec.
Aliasing: §4. Subsystem verbs: §6. System: §7. The full end-to-end walk:
§11.

### 9.1 The Model

```cpp
namespace naga::cospan {

/// The Model owns exactly four kinds of thing: one sk::context, one
/// sk::queue, the per-device memory substrates ({arena, optional pool}),
/// and the root System. It drives the setup chain (declare, exec info,
/// substrates, prepare, compile, and collecting the marks of §8.4) and
/// the forward; it
/// never touches a Subsystem except through the Subsystem verbs.
struct ModelConfig {
    /// The platform context every device of the model lives in. A
    /// device-list constructor builds one; the default constructor builds
    /// one over the runtime's default-selector device.
    sk::context ctx;
    ModelConfig();
    explicit ModelConfig(sk::context c);
    explicit ModelConfig(std::span<const sycl::device> devices);

    /// Per-device dynamic-pool sizes; a device not listed gets
    /// `default_pool_bytes`. 0 = no pool = every dynamic allocation takes
    /// the OS route. Pre-allocation is a profiling-discovered
    /// optimization; memory_report() reports the pool's use.
    /// Why: P3.
    std::unordered_map<sycl::device, size_t> pool_bytes;
    size_t default_pool_bytes = 0;
    size_t pool_node_size     = 1024;
    /// The default device (root System's device unless it overrides;
    /// what every System inherits by default). Defaults to the runtime's
    /// default selector. Validated at Model construction:
    /// ctx.contains(*default_device), else throw.
    std::optional<sycl::device> default_device;
    /// Handed to the queue at construction. Asynchronous errors reach it
    /// when wait() runs queue.wait_and_throw(); absent → the runtime's
    /// default behaviour. SYCL's own mechanism; the Model adds nothing.
    std::optional<sycl::async_handler> async_handler;
};

/// THE factory. Constructs the root System with `args...`, the Model
/// around it, and returns the Model by value (prvalue; Model is movable,
/// so this is also merely ordinary).
template <class Sys, class... Args> requires std::derived_from<Sys, System>
Model create_model(ModelConfig cfg, Args&&... args);

class Model {
public:
    Model(Model&&) noexcept;  Model& operator=(Model&&) noexcept;
    Model(const Model&) = delete;

    // ---- lifecycle -------------------------------------------------------------
    /// THE SETUP CHAIN. Runs once. Every step is a Subsystem verb driven
    /// from the root; the Model adds only the two things nobody else
    /// owns (the queue, the substrates):
    ///   0. at Model construction: the context is cfg.ctx; the queue is
    ///      queue_ = sk::queue{ctx_, default_device, cfg.async_handler} — ONE queue,
    ///      one in-order stream (§7.3), existing for everything below
    ///   1. root.run_setup(); root.resolve()       — declare (§6/§7), ending with the resolution pass
    ///      that records, per variable, its ports and init (§4.4)
    ///   2. bytes = root.get_exec_info(default_device)   — device -> bytes (arena bytes: static device roots whose every port is eternal)
    ///   3. substrates[d] = { DeviceArena{queue_, d, kind::device, bytes[d] + slack} iff bytes[d] > 0,
    ///      DevicePool iff configured for d } — every subsystem allocates
    ///      through its device's substrate. The arena's capacity is the static
///      bytes plus 4096*64 bytes of slack, the bytes measured exactly (the
///      sum over the device's distinct roots), so capacity = sum + 262144
    ///   4. root.prepare_exec(queue_, substrates_) — capture queue and substrates, bind them to the roots, shape updates; NO allocation; ExecCtx complete
    ///   5. root.compile()                        — bodies
    ///   6. marks: walk children() depth-first in declaration order and read design_var()/constraint()/objective() off every root entry; cached (§8.4)
    /// Runs once: a second setup(), or forward() / host access before setup(), throws.
    /// After setup the model's STATE is not yet meaningful: it holds
    /// initial values. State is defined after the first forward.
    /// Why: §1 need 2.3 initialization.
    void setup();

    /// The root's run_pre_compute, run_compute_forward, run_post_compute: enqueue one forward and return. Loop it freely — the stream orders (§7.3).
    void forward();
    /// queue_.wait_and_throw(). The caller's choice, never implicit.
    /// Why: P3.
    void wait();

    // ---- host access (between runs) ---------------------------------------------
    /// Views (HostView, §2): shape(), operator[] with
    /// id<Rank> and flat, size(), span(). On a device-kind root a view is
    /// root.host_view(); on a shared- or host-kind root it is
    /// queue_.wait() then root.get_access(). Either way the cost is
    /// honest and paid only where asked; the read is ordered after the
    /// stream's prior work by SlabKit (§9.4).
    template <DType DT, int Rank> HostView<DT, Rank>    view_input (std::string_view path) const;
    template <DType DT, int Rank> HostView<DT, Rank>    view_output(std::string_view path) const;   // read-only
    template <DType DT, int Rank> HostRagged<DT, Rank> view_ragged_input (std::string_view path) const;   // §3.5
    template <DType DT, int Rank> HostRagged<DT, Rank> view_ragged_output(std::string_view path) const;
    /// The only way to give an input data from the host: set the initial value of
    /// the input's variable, any time after setup(). It wins over the declared inits and is
    /// applied at the variable's next run, even if it is live (once per call). An output path throws
    /// std::invalid_argument; a connected input stores it, warns on std::cerr, and
    /// it is never applied.
    void init_input(std::string_view path, InitSpec init);

    // ---- retrieval & observation ------------------------------------------------
    template <DType DT, int Rank> Input<DT, Rank>&  get_input (std::string_view path);
    template <DType DT, int Rank> Output<DT, Rank>& get_output(std::string_view path);
    System& root();
    Subsystem& get_subsystem(std::string_view path);           // root.get_subsystem

    struct MemoryReport {
        /// A snapshot at the time of the call, between runs. `arena_bytes` is the
        /// arena's capacity, slack included. The OS route is not counted:
        /// `os_allocations` and `os_bytes_in_use` are always 0.
        struct PerDevice { size_t arena_bytes, arena_bytes_in_use, pool_bytes, pool_bytes_in_use,
                                  os_allocations, os_bytes_in_use; };
        std::unordered_map<sycl::device, PerDevice> devices;
    };
    MemoryReport memory_report() const;

    // ---- marks (§8.4) — collected at the end of setup() ----
    std::span<const MarkRecord> design_vars() const;
    std::span<const MarkRecord> constraints() const;
    std::span<const MarkRecord> objectives()  const;

    /// Does not wait: call wait() before destroying or reassigning.
    ~Model();

private:
    sk::context ctx_;
    sk::queue   queue_;
    Substrates  substrates_;
    std::unique_ptr<System> root_;
    ModelConfig cfg_;
    sycl::device            default_device_;
    std::vector<MarkRecord> design_vars_, constraints_, objectives_;
};

} // namespace naga::cospan
```

### 9.2 Host access

- `view_*` on a device-kind root = `root.host_view()`; on a shared- or
  host-kind root = `queue_.wait()` then `root.get_access()`.
- `init_input(path, init)` gives a free input a new initial value, applied at the variable's next run, even if it is live (§4.4).
- `forward()` enqueues; `wait()` = `queue_.wait_and_throw()`.

### 9.3 Sample — the whole user-side lifecycle

```cpp
auto model = create_model<Top>(ModelConfig{});          // default device, no pools
model.setup();                                          // declare → exec-info → substrates
                                                        //   → prepare → compile → marks (no allocation yet)
model.init_input("mda.z", init::from_host(std::vector<double>{1.0, 2.0}));  // drive a free input
model.forward();  model.wait();                         // enqueue; wait when results are wanted

auto y1 = model.view_output<DType::f64, 1>("mda.y1");   // host_view, read
auto r  = model.memory_report();                        // r.devices[gpu0].pool_bytes_in_use: how much pool did the run use?
                                                        //   -> size pool_bytes from it

// Two devices in one model:
ModelConfig cfg{std::array{gpu0, gpu1}};                // builds the sk::context
cfg.default_device = gpu0;                              // validated with ctx.contains()
auto m2 = create_model<Top>(std::move(cfg));
```

### 9.4 The layering rule

Each layer orders what it submits: SlabKit its tasks, CoSpan the kernels it
launches, the user anything they launch over a root outside the framework.
Teardown relies on AdaptiveCpp queue destruction; waiting before exit is
the user's obligation. `~Model` does not wait; call `wait()` before destroying or reassigning a Model.
Members are destroyed in reverse declaration order: `cfg_`, `root_` (every
root's `usm_vector` releases), `substrates_` (arenas and pools), `queue_`,
`ctx_` — no destruction order is relied on beyond that (P9).

---

## 10. System types: the extension contract, by example

Binding contract; declarations only. The base System, Parallel and
Fused: §7. Systems never do work — a System type alters ORCHESTRATION
(what runs, when, on which device), never compute. Work is a Component's,
including custom operations and host bodies (§5.1).

### 10.1 The principle

Wondering where some unique behaviour goes? It is a custom System type.
The runners are virtual precisely so that a System subclass can override
the orchestration while every Component under it stays untouched and
every owner above it stays ignorant. **No framework primitive exists for
gating or change detection**: those are ordinary System
subclasses — provided in the library where common, written by users
where not — and their entire mechanism is an early return from
`run_compute_forward`. Running a subtree for its first N forwards is a run policy, not a type: `property::run_n{n}` and
`property::do_once{}` on `add_subsystem` (§4.4, §7.4).
Why: §1 need 2.2 possibility floor (do once / run N / run when changed /
run if) · P5.

```cpp
namespace naga::cospan {

// A run-N subtree: add_subsystem<Operators>("operators", property_list{property::run_n{n}}) runs it
// on the first n forwards; do_once is n = 1. A deleted subtree materializes
// nothing; its variables live as long as some port holds them.

/// Predicate-gated on COMPUTED state: the System names one of its own
/// (promoted) inputs; each forward it reads that input's root on the
/// host and decides — the flag root's host_view() for a device-kind
/// root, ctx_.queue->wait() then get_access() for any other kind. The
/// read is the honest cost of a host-side decision on device data, paid
/// only by Systems that make one. The untaken case submits nothing;
/// outputs of the subtree simply hold their previous values.
class ConditionalSystem : public System {
    std::string flag_path_;
public:
    explicit ConditionalSystem(std::string flag_input_path);
protected:
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        if (!read_scalar<bool>(flag_path_)) return nullptr;    // one scalar, host-side
        return System::run_compute_forward();
    }
    // read_scalar<bool>: flag root's host_view()[0] (device kind);
    //   ctx_.queue->wait() then get_access()[0] (other kinds).
};

/// Host-decided gating: the model's own logic (schedules, counters).
class HostConditionalSystem : public System {
    std::function<bool()> pred_;
public:
    explicit HostConditionalSystem(std::function<bool()> p) : pred_{std::move(p)} {}
protected:
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        return pred_() ? System::run_compute_forward() : nullptr;
    }
};

} // namespace naga::cospan
```

These are added like any realized type: `add_subsystem<Operators>("operators")` (add it with `property::run_n{n}` for a run-N subtree),
`add_subsystem<ConditionalSystem>("regime", "regime_flag")`. They compose by
nesting.

### 10.2 What the contract guarantees a custom System type

- **Runners are virtual, hooks are for users.** A System type overrides
  `run_compute_forward` (and, when it must, `prepare_exec`, `compile`);
  its users override `setup` as with any System.
- **Returning `nullptr` means "I submitted (or chose not to)".** The owner
  carries on; there is nothing for it to refresh (§7.3). Returning a
  description means "submit this for me" — a System type may also do
  that (a fused wave is exactly that).
- **The queue, device, group and substrate are in `ctx_`**, complete after
  `prepare_exec`. A System type that groups its children's submissions
  (ParallelSystem) does so with `queue->submit_group` (§7.5).
- **Ordering never needs cooperation from children or owners.** It is the
  stream's (§7.3); a System type that submits on its own inherits
  correctness by construction.
- **Nothing above or below can tell.** The opacity contract (§7) is
  what makes a custom System type composable anywhere in a tree.

### 10.3 Path resolution and enumeration

Observation is path resolution and enumeration, nothing more (why:
§1 need 2.3 host access):
`model.get_input("subs1.subs2.x")` resolves as
`get_subsystem("subs1") → get_subsystem("subs2") → get_input("x")`, with
`get_subsystem("subs1.subs2")` as the chaining convenience; every
Subsystem exposes `get_inputs()` / `get_outputs()` (§3.3 enumerability
guarantee) and a System exposes its children (`children()`). Recorders, listings and
renderers are built over these calls and nothing else.

---

## 11. Lifecycle — construct, set up, run

The spine the other sections hang from: every framework verb, in the
order it fires, with the state that exists after each. Binding contract.
Cast: Model (§9) · Subsystem verbs (§6.1) · System and ExecCtx (§7) ·
Component (§6) · IOEntry sets (§4) · SlabKit memory (SlabKit's spec).

**Assumption stated once:** the model is ACYCLIC; cycles are not supported.

### 11.1 Construct

```cpp
auto model = create_model<Top>(ModelConfig{...}, top_args...);
```

1. Model validates its config (`default_device` against `ctx.contains()`),
   holds the context (`cfg.ctx`), then constructs its one queue from it:
   `queue_ = sk::queue{ctx_, default_device, cfg.async_handler}` — one in-order
   stream. Every submission names the device it targets.
2. Model constructs `Top(top_args...)`. A constructor is plain C++ state.
   Structural calls and declarations outside `setup()` throw.

**State:** one context, one idle queue, one root object, nothing else.

### 11.2 `model.setup()` — the setup chain

`run_setup + resolve → get_exec_info → substrates → prepare_exec → compile →
marks`.
Every step is a Subsystem verb driven from the root (the resolution pass is
the root's own); Systems recurse in
declared order (`children_` IS the exec order); nobody asks what kind a
child is. The order is the one of §9.1: (0 construct) 1 run_setup and the
resolution pass · 2 get_exec_info · 3 substrates · 4 prepare_exec ·
5 compile · 6 marks.

#### 11.2.1 Declare — `root.run_setup()`

`System::run_setup` → `setup()`. Inside it: `add_subsystem` constructs the
child, registers the name, sets its owner, **runs the child's `run_setup`
before returning** (why: P6 — no deferred resolution pass);
`connect` / `promote` / `tie_*` resolve paths in this
namespace and apply the set operation (§4); marks record on roots;
`init_input` records an init declaration at its depth.

`Component::run_setup` → `setup()` → `declare_*` creates entries (roots of
singleton sets; flags by kind; DimRecords); `init_input` records an init
declaration on one of its own inputs. Zero
allocation; no device anywhere in this step.

**The resolution pass.** Once the root System's `setup()` returns, the root runs
`resolve()` over the final tree and sets (§4.4). Per variable it records its ports and
its `init` (highest declaring level wins; a tie at one depth is an error; an init on an
output is an error; an init on a variable with a source is ignored; none means zeros).
It checks the shape-authority setup errors of §3.6, which are fully declared
contradictions only (no ordering or cycle analysis for shapes, §4.5), and prints the
cycle warning of §4.5 for each connection whose consumer runs before its producer.
Nothing is materialized here, and no run counter is derived: each parent keeps the
counter of each child from its `add_subsystem` property (§4.4). A mark does not affect
lifetime.

**State:** final tree and exec order; each parent's run counters; final sets; marks; each
variable's ports and init. No storage.

#### 11.2.2 Exec info — `bytes = root.get_exec_info(default_device)`

`System`: `ctx.device` = own `property::device` else `parent_device`; seed
`{device: 0}`; process **own alias entries first** (a tie alias root
counts its static bytes here); then each child in order, merging returned
maps (sum on key).

`Component`: device = `parent_device`; every own entry's placement is
(`parent_device`, the Component's default storage kind — host kind for a
host body); per own entry `e`:
- root → register this placement as `e`'s primary; count its static bytes
  here (zero unless every port is eternal, and for a host-kind primary);
- input peer, `root(e)` already has storage registered at this placement →
  nothing;
- input peer, it does not → register this placement on `root(e)` and count
  the root's static bytes here when the placement is device kind and the
  root is arena-eligible (the root will own a copy at this placement; §4.2);
- output peer (only a tied output is one) whose root's placement ≠ this
  placement → ERROR naming the tie: producers cannot write one storage
  from two placements.

Every root is processed before any peer of it: alias roots before
children, producers before consumers in registration order. A cycle is expressible by
registration order: without solvers it produces the setup warning of §4.5 and runs with
the consumer reading the producer's previous value; solvers are future work. A System processes its
own alias entries the same way (a tie alias root registers the System's
device as primary; a promotion alias is a peer and registers the
System's device on its root — so `get_output("alias")` from that System
yields memory on that System's device). Returns `{device: bytes}`.
Why: §1 need 2.2 placement · P1 · P5.

#### 11.2.3 Substrates

Model: for each `(device, bytes)`: `substrates[device] =
{ DeviceArena{queue_, device, kind::device, bytes} iff bytes > 0, DevicePool iff
configured for that device }` — built from `(queue, device)`, with no
context of its own; the one context is the Model's. Every
allocation for a device goes through its substrate.

#### 11.2.4 Prepare — `root.prepare_exec(queue, substrates)`

`System`: complete `ctx` — `queue` = the Model's, `group` = null (only a
Parallel owner sets one, for the extent of its `submit_group`),
`substrate = &substrates[device]`; bind the substrates to its alias roots;
recurse. NOTHING IS ALLOCATED here: a variable is materialized on need,
in a forward (§4.4, §6.5). The arena
is created before this step (§11.2.3), sized by `get_exec_info`, and stays
empty until something is materialized.

`Component`: capture the queue and substrates, bind the substrates to each
own root (so the one allocation routine, §4.4, can draw on them), apply the
shape updates that need no value (an unspecified dim of a connected input),
point the wrapper of every member whose root happens to have storage, and
validate static input peers against their roots, and check the fixed dims of connected and tied
entries against fixed suppliers (§3.6). A free input with an unspecified dim and no shape-giving
initializer is a setup error (raised by resolve()).

#### 11.2.5 Compile — `root.compile()`

`Component`: device function(s) + id at the body's own rank
(`kernel_rank_v`); individual config; a custom-op or host body compiles
nothing. `System`:
children in order; a FusedSystem also checks its structure and builds its
sequence config: it has children, none a custom-op or host body, all on its
device and of one kernel rank, and no value-driven size (`FromInput`) whose
driving input aliases another stage's output in the same fuse
(`FromInputDim` and an unspecified dim are always allowed; §5.8, §7.5). The
rest of the Fused and Parallel legality is checked each forward on the
descriptions the children return (cheap, host-side, and it is where the
information is).

**State:** nothing is materialized yet: each variable is materialized on need in the
first forward (§4.4); wrappers of variables without storage are
unset; kernels compiled. The model can run. Its state is not meaningful
until the first forward.

#### 11.2.6 Marks

Model walks `children()` depth-first in declaration order and reads
`design_var()`/`constraint()`/`objective()` off every root entry into its
three cached lists (§8.4). **State:** the marks an optimizer reads.

### 11.3 `model.forward()`

`Model::forward()` runs the root System's `run_pre_compute`, `run_compute_forward` and `run_post_compute`.
Enqueue only. `model.wait()` (`queue_.wait_and_throw()`) waits the stream
when the caller wants the results on the host; a caller looping N forwards
need not wait between them — the stream orders (§7.3).

#### 11.3.1 A System's turn — the ordering contract (§7.3)

```
for (child : children_)                       // at the START, before any child's turn
    if (child is transient and deleted) build it;   // re-constructed from its stored arguments (§4.4)
for (child : children_) {
    if (child is deleted) continue;           // declared, currently deleted (§4.4)
    child->run_pre_compute(ctx_);             // a Component materializes the variables it accesses here (§4.4, §6.5)
    auto d = child->run_compute_forward();
    if (d) d->submit(ctx_.target());          // WE submit, routed on d->is_custom_op() / d->is_host()
    register the launch's kernel events on the storage of every port the child touched   // §4.4
    child->run_post_compute(ctx_);            // copies AFTER the work they depend on —
                                              //   ordered by the queue and the registered events
    if (child is run_n or do_once) { move its counter; if (it reaches 0) delete child; }   // right after its turn (§4.4)
}
for (child : children_)                       // at the END, after the last child's turn
    if (child is transient) delete child;     // a transient child exists for one forward of its parent (§4.4)
return null
```

The stream is an abstract in-order sequence of items: a kernel is an item;
a group is one item (unordered inside, ordered both ways outside); async
host and memory tasks are concurrent with each other and ordered after
every kernel/group item before them and before every one after. Placement
never costs the order; nothing in CoSpan threads dependencies. Kernel
descriptions take the outer-kernel path, custom operations take
`AdaptiveCpp_enqueue_custom_operation`, through the same `SubmitTarget`;
host bodies take `async_host_in_order` (the group's `async_host` under a
Parallel owner).
**Parallel** (Components only): pre-compute ALL → `queue.submit_group {
each child's description submitted with its own device }` → post-compute ALL
after the group returns. **Fused**: pre-compute ALL → compute ALL (collect;
a null description is an error) → validate the wave on the descriptions
(none null, rank agreement, one work-group shape; parallel mode: no
connections between stages, and tied outputs only with
`allow_tied_outputs()`; sequential mode: no inter-stage checks) → pack,
stamp group and local slices, check the wave's local memory against the
device's limit, one staging, ONE submission with the sequence config (the
framework inserts no barriers) → post-compute ALL. **Custom System types** (§10): early-return from
`run_compute_forward`; the components under an early return materialize and submit nothing. A System materializes only the aliases it declares, when it reads them (§4.4); every other variable is materialized by the Components that access it (§11.3.2).

#### 11.3.2 A Component's turn (§6.5)

1. `run_pre_compute(ctx)`: shape resolution — value-driven entries first (read the driving input's value on the host, the one forced sync, so the launch geometry is known; `root.resize(shape)`, `point_wrapper`; nothing submitted for the resize, no reallocation inside a submitted task), then shape-derived dims. Then `pre_forward()` → MATERIALIZE every variable this Component has a port on that is not yet live (its init applied when it is a free input and live nowhere; copied from a live placement otherwise) → constraint checks against suppliers (§3.6) → dynamic validation → wrapper sync (§4.4).
2. `run_compute_forward()` → `compute_forward()` → description (kernel,
   custom op or host body). The owner submits it.
3. `run_post_compute(ctx)`: for each output root with storage at other
   placements, for each other placement `copy_d = root` — one assignment
   resizes to match and copies, routed by SlabKit — after the owner's
   submission and the registration of its kernel events on the storage of every
   port this Component touched (§4.4), ordered by the queue. Nothing is released
   here: storage goes when the last port holding it is deleted.

**State:** every output root holds this forward's values; dynamic roots
sized to this forward; a variable is materialized when first accessed and lives as long as
some port holds it (§4.4).

### 11.4 Host access

`view_*` (device-kind root: `root.host_view()`; shared/host-kind root: `queue_.wait()` then `root.get_access()`), `init_input(path, init)` (§9.2), `memory_report()`, `get_subsystem(path)` chaining and `get_inputs()/get_outputs()` enumeration (§10.3). Host access is §9.2. Host access to a variable that is not materialized (§4.4) throws, saying it has no data while nothing that uses it exists.

### 11.5 End of life

The layering rule (§9.4): each layer orders what it submits; teardown
relies on AdaptiveCpp queue destruction, and waiting before exit is the
user's obligation. `~Model` does not wait; call `wait()` before destroying or reassigning a Model. Everything
else is RAII, in reverse declaration order, identical to §9.4: `cfg_`,
`root_` (roots' `usm_vector`s release), `substrates_` (arenas, pools),
`queue_`, `ctx_` (P9).

### 11.6 Platform and toolchain notes

- Builds are Release without `-g`: the toolchain's clang 21.1.8 crashes in
  the Debug Variable Analysis pass on these translation units, so Debug
  and `-g` builds are not supported.
- On CUDA and HIP, work that depends on `async_host` work is held back on
  the host until that work completes: a kernel that follows a host body
  waits in the host, not in the stream. A host body may therefore call
  `sycl::free` (the case the hold-back exists for) and nothing else from
  SYCL.
- A host body is plain host C++: no SYCL calls except `sycl::free`, no
  device code, no backend API calls (§5.1).

