# CoSpan specification — r01, 2026-09-28: the design as agreed

**What this is.** The CoSpan design as built. This file describes the design the code
implements and is kept in step with the code. Discussion decisions live in
the project knowledge; this document holds only what is specified. Notation: `sk::` is SlabKit's
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
map-valued I/O · transient subtrees · initialization · host access ·
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
- **Work outside device kernels:** legacy code, file I/O, callbacks, and
  native libraries (BLAS, solvers) must be usable as ordinary components,
  ordered with everything else. Floor: a host body (plain host C++ over
  host-visible I/O, ordered in the stream) for legacy code, file I/O and
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
- **Transient subtrees** — a property an owner gives a subsystem at
  registration: the roots that live only inside it are allocated by the
  components that touch them and released after their last user (§4.4).
- **Chunked execution over bounded scratch** — N independent instances
  whose per-instance intermediate scratch would cap problem size if
  materialized at once (per-node dense solves, operator fits): run in
  chunks over reused transient scratch, results landing in the
  full-shaped output. A first-class pattern with its own ergonomics;
  touches shapes and transient subtrees (§4.4). (Not specified in this revision;
  see the roadmap.)
- **Initialization** — free input roots receive initial values without a
  component existing to produce them; the value is metadata on the root,
  declared by a component or by any owner above it, and the highest level
  that declares one wins (§4.4). Outputs are always computed fresh and are
  never initialized (§6.0).
- **Host access** — reading any I/O on the host by path; writing free
  inputs (driving design variables and parameters) and outputs
  (injecting external results), with host-side views that behave like
  the device wrappers.

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

/// The closed set of I/O element types. Extension is a spec revision.
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

/// SUPERSEDED by D139: there are no mutable host views; HostViewMut survives only as
/// the argument type of cs::init::custom_host (built by the framework, committed on return).
/// HostViewMut<DT, Rank> — the same host copy, writable: commit() and
/// destruction write the buffer into the entry's root storage
/// (usm_vector::from_host, which owns its copy of the bytes before
/// returning) and refresh the root's copies at other placements. Obtained
/// from Model::view_output_mut (§9) and, as the argument of the function of
/// cs::init::custom_host, by an initial-value declaration (§4.4).
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
    friend class Model;
    friend class ComponentBase;
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

/// This dimension = dim `dim` of the upstream connection's shape.
/// Inputs only (outputs have no upstream).
struct FromSourceDim { uint8_t dim; };

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

/// FromSourceDim is ill-formed for outputs.
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
    /// Every entry belongs to one set with one ROOT (data owner and, for
    /// outputs, producer); peers are names for and consumers of the
    /// root's data. There are no source/target links.
    IOEntry* root();                 ///< self iff is_root()
    bool     is_root() const;
    bool     can_be_source() const;  ///< construction-time (§4.2)
    bool     can_be_target() const;
    /// Device placement and per-placement root storage — device(),
    /// placement(), storage_on(), storage_placements() — are declared in §4.

    // ---- registration metadata (ROOTS; recorded by the resolution pass, §4.4) --
    bool transient() const;                          ///< every registration in the set is transient
    const std::optional<InitSpec>& init_spec() const;  ///< free input roots: the winning initial value, if any
    const Subsystem* last_user() const;              ///< transient roots: the last component that touches it
    bool released() const;                           ///< transient roots: storage currently dropped
    const Subsystem* declarer() const;               ///< every entry: the Subsystem that registered it

    // ---- capabilities ---------------------------------------------------

    /// Automatic shape update: applies the DimRecords. Invoked by the
    /// OWNING COMPONENT in run_pre_compute, BEFORE connection validation.
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

    /// Allocation: ROOTS ONLY. Every root, durable or transient, is allocated
    /// by its OWNER Component in run_pre_compute (§4.4), or by a host write.
    /// Builds the root's
    /// slabkit::usm_vector for each placement the set spans. A device
    /// placement takes its device's substrate (§7.1): a device-kind root
    /// that is fully fixed and not transient → the arena
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
declare_input(reshaped, "reshaped_input", FromSourceDim{1}, 5, 5);

// Whole-shape value-driven: "dims" is a rank-1 u64 input of size 3.
Input<DType::f32, 3> grid;
declare_input(grid, "grid", from_input_array("dims"));
```

---

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
    /// ordinary usm_vector copy rule); whoever writes a root from the host
    /// does the same. Registered in get_exec_info (register_placement;
    /// set_primary_placement puts the root's own placement first),
    /// allocated by its owner Component in a forward (§4.4). Why: §1 need 2.2 placement · P1 (a name
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
snapshotted. (D139 part 2: `promote<DT, Rank>(path, name)` names the dtype and rank,
which must be the entry's, else a setup error; the alias MIRRORS the entry: its
shape is read live from it, so a view through the promoted name shows the member's
current shape. There are no inert alias extents.)

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
already connected fails here ("tie before connecting"). (D139 part 2: the
call is `tie_inputs<DT, Rank>({m...}, name, AliasShape)`. Fewer than two
members is an error ("a tie needs two or more members; use promote for
one"). Every member is a RANK-ONLY entry (declared `cs::tied`, or a
promotion of one) of dtype DT and rank Rank; a member that declares any
shape is an error naming it. A tie alias carries a shape, so it cannot be a
member of another tie: pass members up by promotion.) Merge every member's
set into the alias. THE ALIAS OWNS THE SHAPE, given by its AliasShape:
`cs::alias::fixed(e0[, e1, e2])`; `cs::alias::from_connection()` (tie_inputs
only: the shape of the output connected into the alias; an error at setup
if nothing is); `cs::alias::from_output_shape(path)` (the shape of an
upstream output of the alias's rank); `cs::alias::from_output_values(path)`
(the extents are the VALUES of elements 0..Rank-1 of an upstream output of
integer dtype, read on the host as FromInput reads a value). The paths are in
the declaring System's namespace and name an output whose producing
component is registered BEFORE the tie's first member (checked in resolve();
FusedSystem and ParallelSystem refuse a source from the same wave, as for
value-driven sizes). The unconnected alias is the root; its owner resolves
the shape in `run_pre_compute` before allocation. Once an output is
connected into the alias it joins that output's root (the producer owns and
allocates it) and the members take that output's shape; a fixed or
from_output_* AliasShape is checked against it on the first run
(`invalid_argument` on a mismatch). Members' shapes are "from the root"
(`DimRecord::How::from_root`, resolved in update_shape). After resolve() every
rank-only entry must belong to a tie's set. THE EXCEPTION: mark each member
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
each member `can_be_source = false`. SHAPE (D139 part 2): the call is
`tie_outputs<DT, Rank>({m...}, name, AliasShape, policy)`; the members are
rank-only outputs, and the alias is ALWAYS the root and owns the shape by
its AliasShape (fixed, from_output_shape, from_output_values;
`from_connection` is an error, "an output alias defines the shape of its
connections"). The root's owner resolves the shape before allocation, so
the root is sized and (re)allocated from it each run. This is what the
zeroing producer plus accumulator of §6.7 (ZeroBins + Histogram) uses: the
bin count is the value of an upstream output. (Removed: shape_source, the
"first fixed member" rule, inert alias dims and the D138 member-to-root
reshape propagation.)

Path resolution for every operation is from the caller's namespace,
dot-separated, descending children; the last segment resolves an alias
registered in that System or a Component's declared entry.

### 4.4 What owns memory

Every entry that is a root once declaration completes owns storage —
including unconnected inputs and unconnected `tie_inputs` aliases: that
is how top-level design variables and shared inputs get memory with no
IndepVarComp (why: §1 need 2.3 initialization, §1 need 2.4 design
variables). Route (§7.1 `MemorySubstrate`): a device placement of a device-kind root
that is static and not transient takes its device's
arena; any other device placement takes that device's pool when it has one,
else the OS; a host-kind placement takes the OS route on its device (a
host-kind slot counts no arena bytes, and a host-primary root has no
static bytes) — all at first need and on growth. Peers own nothing, ever:
a root holds one allocation per placement its set spans (§4.2).

TRANSIENTNESS IS REGISTRATION METADATA. Only an OWNER declares a subsystem
transient, by passing `property::transient` to `add_subsystem` (§7.4); the
subsystem may be a Component or a System. (A ParallelSystem's children
still take only `property::device`, §7.5.) A REGISTRATION is a Component's
entry, or a promoted or tied alias made by a System; every registration
records the Subsystem that made it (`declarer()`). A registration is
transient if its registering subsystem, or any ancestor of it, was added
transient: the highest level wins, and an inner level cannot undo an outer
transient. A ROOT is transient if and only if EVERY registration in its
alias set is transient. A promotion or tie made at a non-transient level,
or any non-transient component touching the root, makes it durable — which
is how an output consumed from outside a transient subtree, or promoted by
an owner above it, keeps its storage. The decision is recorded once per
root by the RESOLUTION PASS, at the END of the ROOT System's setup (the
System with no owner), not per System, because registrations made later by
an outer System decide whether a root is transient. The pass also records,
per root, its initial value (below), its OWNER (the first Component, in
depth-first registration order, that touches the root) and its LAST USER
(the last such Component). A transient root is never arena-eligible and
counts no static bytes: it takes the pool route, or the OS route when the
device has no pool. A mark (§8.4) on a transient root is an error at setup.

INITIAL VALUES ARE METADATA ON FREE INPUT ROOTS. A FREE INPUT ROOT is an
input root with no source. Its initial value is declared during `setup()`
and applied by the framework; outputs are never initialized (§6.0).

```cpp
namespace naga::cospan::init {
template <class T> InitSpec fill(T value);                 // every element = value
InitSpec                    zero();                        // every element = 0
template <class T> InitSpec from_host(std::vector<T> v);   // element i = v[i]; size = the root's flat size
template <DType DT, int Rank>
InitSpec custom(std::function<void(sk::queue&, Initializer<DT, Rank>)> fn);   // device-side; the function enqueues on the queue
template <DType DT, int Rank>
InitSpec custom_host(std::function<void(HostViewMut<DT, Rank>&)> fn);         // host-side; committed on return
}
// ComponentBase::init_input(std::string_view name, InitSpec)   — on the Component's OWN input
// System::init_input(std::string_view path, InitSpec)          — a child input path, or one of
//                                                                this System's promoted or tied input aliases
```

- **Highest level wins.** The declaring subsystem's depth decides, the
  root System being depth 0; the smallest depth wins. Two declarations at
  the same depth on one root are an error at setup (the resolution pass).
- **A root with a source ignores its init.** An init declared on an input
  whose root ends up with a source (a connected input) is ignored, so a
  reusable component can carry a default for the case it is left
  unconnected. An init declared on an output is an error at setup.
- **No init, zeros.** A free input root with no init is zero-filled.
- **When it applies.** The framework applies a root's init when the root's
  OWNER allocates it from nothing, in its `run_pre_compute`: on the owner's
  first run for a durable root, at each (re)allocation for a transient one.
  Nothing is applied at setup. A value-driven resize preserves the existing
  content and does not re-init. (Superseded by D139: a host write no longer
  exists; `Model::init_input` leaves a pending init that the owner applies
  at its next run, once per call.)

THE LIFECYCLE IS THE COMPONENT'S, and the component is always the leaf, and
ALLOCATION AND INIT HAVE ONE PATH, the owning component's, for durable and
transient roots alike. Every root has an OWNER, recorded by the resolution
pass: the FIRST Component, in depth-first registration order, that touches
it. In `run_pre_compute`, after shape resolution and `pre_forward`, a
Component goes over the roots it OWNS (inputs and outputs); any owned root
without storage is allocated (the route above), and if it is a free input
root its init from metadata is applied (zeros when none). A Component never
allocates or inits a root it does not own; non-owners only point their
wrappers, as peers do. A DURABLE root is allocated once, on its owner's
first run, and never released. A TRANSIENT root is released after its last
user and allocated again on its owner's next run. In `run_post_compute`,
after the other-placement sync, a Component releases the transient roots
for which it is the LAST USER: each root waits on the stream, then is freed
(synchronous frees, D135), with one stream wait for the component's batch.
Children of a ParallelSystem or FusedSystem are ordered within their flavor
(for owner and last user alike), and their `post_compute` runs after the
group or wave returns, so a root's release follows the work of every child
in it. A Component that does not run in a forward (a RunNSystem that is
exhausted, a gate not taken) neither allocates nor releases; its roots
simply wait. An exception thrown mid-forward leaves roots unreleased (D84).

SUPERSEDED by D139 (the host-write part of this paragraph): there is no host
write. `Model::init_input(path, init)` sets the root's init (it beats the
declared ones), marks it pending, and the owner applies it at its next run;
an output path throws, and a connected input warns on std::cerr and is never
applied. A root is PRESENT once its owner's allocation path has run, even at
size 0; views of a present size-0 root are empty, views of a root that is not
present throw.

HOST ACCESS. A host WRITE to a root with no storage (`Model::set_input`, or
a `HostViewMut` write-back or commit) allocates the root through the SAME
allocation routine the owner uses (one function, called by both), and the
written value is its initial value: when the owner later runs it finds
storage and does nothing, so `set_input` before the first forward beats the
init. Reading a root that has no storage (`view_*`) throws, saying it has no
data until its owner runs; for a RELEASED transient root the error names the
root and the transient subsystem whose registration made it transient, and
says the storage exists only while that subsystem's components run (a
silent empty view would read as valid data). A host write to a transient
root throws the same way. The one cost is a host stream wait per releasing
component per forward. Why: §1 need 2.1 transient subtrees, §1 need 2.3
initialization · P3 · P6 (declaration is one phase; the resolution pass is
its end).

### 4.5 From sets to storage

Aliasing is complete the moment each structural call returns. Storage
follows in the setup chain (§11) and the first forward: `get_exec_info`
registers on each root the placements its set spans (the root's own
placement first) and counts the root's bytes once per such device-kind
placement, none for a host-kind one or a transient root, which sizes the
arena; `prepare_exec` allocates nothing, it only binds the substrates and
sets what can be set. Each root is allocated by its OWNER in the owner's
`run_pre_compute` (§4.4), and the wrappers of every member of an allocated
root are pointed at `root()->storage_on(own placement)` by the Components
that touch it, at their turn. Regrowth of a root needs no notification:
peers re-point at their next turn. Copies at the non-primary placements are
refreshed by the root's producer after every launch.

`tie_outputs` members MUST share the alias's placement: producers cannot
write one storage from two placements. Checked in the members' Components'
`get_exec_info` (a non-root output whose root's placement differs from the
Component's own — the alias was processed first by its System, so the
placement is known), error naming the tie.

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
the roadmap for provided kernels (reductions, scatters, scans, sorts).
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
/// component (legacy code, file I/O, callbacks). They are never compiled
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
  both modes. Shape-from-shape links (`FromInputDim`, `FromSourceDim`,
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
  next one's input, transient (§4.4) when it should not outlive its last
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
    /// Allocate the DURABLE roots that can be allocated now and point their
    /// wrappers at the storage (own allocation, else root()->data() — roots
    /// are allocated before their peers); Systems complete their ExecCtx.
    /// Transient roots are the components' to allocate, in a forward (§4.4).
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
    transient root, and zero for a host-kind primary).
  - `e` is an input peer: if `root(e)` already has storage registered at
    this placement → nothing; else register this placement on `root(e)`
    and count the root's static bytes here — a device-kind placement adds
    them once per device, a host-kind one or a transient root adds none
    (the root owns one copy per placement its set spans — §4.2).
  - `e` is an output peer (only a tied output is one): `root(e)`'s
    placement must equal this placement, else ERROR naming the tie —
    producers cannot write one storage from two placements.
  (Every root is processed before any peer of it in an acyclic model —
  Systems process their alias entries before their children, and
  producers precede consumers. Cycles are on the roadmap; their design
  owns the exception.)
  - Return `{device: bytes}`.
- **`prepare_exec`**: capture the queue and substrates, bind the substrates
  to every own root, point the wrapper of every member whose root has
  storage, and validate static input peers against their roots. NOTHING IS
  ALLOCATED here: the owner of each root allocates it in its own
  `run_pre_compute` (§6.5). An unconnected input with a `FromSourceDim`
  dimension errors here.
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
    dtypes only, a negative value throws), allocate the entry''s root first if this
    Component owns it and it is still unallocated, call
   `root.resize(shape)`, then `point_wrapper`; no staging
   copy, nothing submitted for the resize, no reallocation inside a
   submitted task (why: §1 need 2.1 shape source · P3 — only value-driven entries pay
   it). Then shape-derived dims, `pre_forward()` hook (per-entry
   override, last write wins), dynamic connection validation, wrapper
   sync (a root that grew was reallocated by its own allocator route).
    Then ALLOCATE every root this Component OWNS (inputs and outputs) that
    has no storage (§4.4: arena for a static durable device-kind root,
    otherwise the pool, then the OS; current extents) and, for a free input
    root, apply that root's init (zeros when none) — the stream orders the
    init before this turn's launch. A root it does not own is only pointed
    at (a peer's wrapper). A root that already has storage — allocated by an
    earlier run, or by a host write — is left alone, so a value-driven
    resize keeps its content and a `set_input` before the first forward
    beats the init. (Superseded by D139: no host write; a pending
    `Model::init_input` is applied to the existing storage.)
2. **`run_compute_forward()`**: `compute_forward()` — returns the
   description. The owner submits it (§7.3).
3. **`run_post_compute(ctx)`**: through `ctx`, after the owner's
   submission: for each output root with storage at other placements, for
   each other placement `copy_d = root` — one assignment resizes to match
   and copies, routed by SlabKit. Ordered after the work it depends on by
   the stream (§7.3). Then RELEASE every transient root for which this
   Component is the last user (§4.4): each root waits on the stream, then
   is freed (synchronous frees, D135), with one stream wait for this
   Component's batch.

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
        declare_input (values_, "values",  FromSourceDim{0});
        declare_input (weights_,"weights", from_input_shape<1>("values"));
        declare_input (n_bins_, "n_bins",  from_source_shape<1>());   // connected from NBins
        declare_output(bins_,   "bins",    tied);                      // a tie_outputs member (D139 part 2)
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
    void setup() override { declare_output(bins_, "bins", tied); }   // the tie alias owns its shape
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
        // bins' shape is the VALUE of nb.n[0], read by the root's owner (zero) each run; atomic by default
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
        declare_input (in_,  "in",  from_source_shape<2>());
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
        declare_input (A_, "A", from_source_shape<2>());
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
    /// Model knows its root is a System and has no ctx of its own, so the
    /// root supplies its own. The root's end-of-declare verb:
    /// resolve() — the resolution pass that ends declare (§4.4, §11.2.1).
    /// There is no apply_inits: every root's init is applied by its owner
    /// Component when it allocates the root (§4.4).
    /// forward() == run_compute_forward() (no pre/post-compute exists for
    /// the root).
    void resolve();
    void forward();

    // ---- retrieval & enumeration (hierarchy-aware; §3.3 guarantee) ------------
    template <DType DT, int Rank> Input<DT, Rank>&  get_input (std::string_view path);
    template <DType DT, int Rank> Output<DT, Rank>& get_output(std::string_view path);
    auto get_inputs()  const;   ///< this System's alias entries
    auto get_outputs() const;
    Subsystem& get_subsystem(std::string_view path);          ///< chains through children
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
    /// owner sets one; substrate = &substrates[device]); allocate own
    /// durable alias roots and point their wrappers; then children.
    void prepare_exec(sk::queue&, Substrates&) override;
    /// children.compile() in order.
    void compile() override;
    void run_pre_compute(ExecCtx&) override {}     ///< Systems do no work
    void run_post_compute(ExecCtx&) override {}
    /// THE DEFAULT LOOP (§7.3).
    std::unique_ptr<LaunchDescription> run_compute_forward() override;

    /// Transientness is registration metadata, not System state (§4.4): a
    /// System neither revives nor releases anything. The Components under
    /// it allocate and release their own roots in their own turns (§6.5), so
    /// every System flavor and any custom type runs its children's turns
    /// and nothing more.
    bool transient_ = false;   ///< a Subsystem field (Components carry it too): set by add_subsystem from property::transient; read by the resolution pass

    ExecCtx ctx_;
    std::vector<std::unique_ptr<Subsystem>> children_;   ///< declared order = exec order
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
for (child : children_) {
    child->run_pre_compute(ctx_);             // a Component allocates its roots here (§6.5)
    auto d = child->run_compute_forward();
    if (d) d->submit(ctx_.target());          // WE submit, routed on d->is_custom_op() / d->is_host()
    child->run_post_compute(ctx_);            // copies AFTER the work they depend on —
                                              //   by the stream, not by an event; a Component
                                              //   releases the roots it is the last user of here
}
return nullptr;                               // "I submitted"
```

- **Ordering is the stream's, never passed.** Which queue a System submits
  to is the Model's one queue (`ctx.queue`); the ordering contract above
  is what every submission, host task and copy relies on. No event, wait
  list or dependency snapshot exists in CoSpan; a child that submitted
  for itself, or skipped, leaves the owner nothing to refresh.
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
    struct transient     {};   // given by an OWNER at registration (a Component or a System): the subsystem and its descendants are transient registrations (§4.4)
    struct fused         { FusedSystem::Mode mode; std::optional<sycl::range<3>> wg; };
}
// Gating, counting and change detection are custom System TYPES (§10),
// added with the templated overload — not properties.
class property_list { /* SYCL idiom: constructed from any subset; has_property<T>(), get_property<T>() */ };

class System {   // ...continued
public:
    /// A REALIZED System or Component type, optionally with properties
    /// (device).
    template <class T, class... Args> requires std::derived_from<T, Subsystem>
    T& add_subsystem(std::string name, Args&&... ctor_args);
    template <class T, class... Args> requires std::derived_from<T, Subsystem>
    T& add_subsystem(std::string name, const property_list& props, Args&&... ctor_args);
    /// A child Component to add through the helper form: type, name, ctor args.
    template <class T, class... Args> requires std::derived_from<T, ComponentBase>
    auto child(std::string name, Args&&... ctor_args);
    /// The HELPER FORM: a System configured by properties (device and/or ONE
    /// flavor — parallel | fused; both on one call is an error, and/or transient), populated
    /// with the given Components in order. No class is written for it. The
    /// children are Components by construction (child<T> requires
    /// ComponentBase), which is what Parallel and Fused require; a helper-
    /// built plain System takes Components too — nest Systems through the
    /// templated overload or a class. Returns the System (typed Parallel/
    /// Fused underneath) for connect/promote by name. Why: §1 need 2.2
    /// inline composition · P2 (placement without editing).
    template <class... Children>
    System& add_subsystem(std::string name, const property_list& props, Children&&... children);
    System& add_subsystem(std::string name, const property_list& props = {});   ///< empty; add children by the templated overload inside setup()
    // add_subsystem constructs the child, registers the name, and runs the
    // child's run_setup BEFORE returning, so later structural calls in this
    // same setup() see it fully declared.

    void connect(std::string_view src, std::string_view dst);                 // identity
    void connect(std::string_view src, std::string_view dst, Slice view);   // §8.1: dst sees a rectangular region of src's root
    template <DType DT, int Rank> void promote(std::string_view child_path, std::string promoted_name);   // D139 part 2
    template <DType DT, int Rank>
    void tie_inputs (std::initializer_list<std::string_view> members, std::string name, AliasShape shape);
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
ties take rank-only (`cs::tied`) members and an AliasShape that owns the
shape (§4.3); `init_input` names an input (an init on an output is an error at setup) and
is ignored when the input's root ends up with a source.

### 7.5 ParallelSystem and FusedSystem — the two policy flavors

```cpp
namespace naga::cospan {

/// CHILDREN ARE COMPONENTS ONLY — enforced by the type system at the add,
/// with the same constraint FusedSystem uses (a System cannot discern
/// children at runtime; why: §1 need 2.2 parallel · P5):
///   template <class T, class... A> requires std::derived_from<T, ComponentBase>
///   T& add_subsystem(std::string, A&&...);
/// (property-list overload with `property::device` only: each child may sit
/// on its own device; the group is one item of the stream whichever devices
/// its members target). Nested Parallel is not a concept.
/// run_compute_forward: pre_compute ALL children → queue.submit_group {
/// each child's description submitted with its own device (the Parallel's
/// unless placed) through a SubmitTarget carrying the group } →
/// post_compute ALL after the group returns (so a child's allocation is in
/// its pre_compute and its release, as a last user, in its post_compute,
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
///   T& add_subsystem(std::string, A&&...);
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
/// (shape-from-shape links, `FromInputDim` and `FromSourceDim`, are always
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
/// submission with the sequence config → post-compute ALL (allocation and
/// release belong to the children's own pre- and post-compute, §4.4). The
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
        auto& d1 = add_subsystem<Discipline1>("d1");
        auto& d2 = add_subsystem<Discipline2>("d2");
        connect("d1.y1", "d2.y1_in");
        connect("d2.y2", "d1.y2_in");            // the cycle: needs a solver
        tie_inputs<DType::f64, 1>({"d1.z", "d2.z"}, "z", alias::fixed(2));   // d1.z, d2.z are declared `tied`
        promote<DType::f64, 1>("d1.y1", "y1");  promote<DType::f64, 1>("d2.y2", "y2");
        init_input("z", init::fill(1.0));        // this level is above d1 and d2: it wins over their defaults
    }
};

class Top : public System {
    void setup() override {
        auto& mda = add_subsystem<SellarMDA>("mda", property_list{property::device{gpu0}});
        auto& obj = add_subsystem<Objective>("obj");
        connect("mda.y1", "obj.y1");
        connect("mda.y2", "obj.y2_head", Slice{subarray<1>({0}, {1})});

        add_subsystem("post", property_list{property::parallel{}}, child<Report>("r1"), child<Report>("r2"));
    }
};
```

---

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
    /// Model. A mark on a transient root is an error at setup (§4.4): a
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
/// substrates, prepare, compile, applying the inits of durable free input
/// roots, and collecting the marks of §8.4) and
/// the forward; it
/// never touches a Subsystem except through the Subsystem verbs and
/// System::forward().
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
    ///      that records, per root, transient, init and last_user (§4.4)
    ///   2. bytes = root.get_exec_info(default_device)   — device -> bytes (arena bytes: durable static device roots only)
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

    /// root.forward(): enqueue one forward. Returns immediately. Loop it
    /// freely — the stream orders (§7.3).
    void forward();
    /// queue_.wait_and_throw(). The caller's choice, never implicit.
    /// Why: P3.
    void wait();

    // ---- host access (between runs) ---------------------------------------------
    /// Views (HostView / HostViewMut, §2): shape(), operator[] with
    /// id<Rank> and flat, size(), span(). On a device-kind root a view is
    /// root.host_view(); on a shared- or host-kind root it is
    /// queue_.wait() then root.get_access(). Either way the cost is
    /// honest and paid only where asked; the read is ordered after the
    /// stream's prior work by SlabKit (§9.4). Read-only variants cost no
    /// writeback.
    template <DType DT, int Rank> HostView<DT, Rank>    view_input (std::string_view path) const;
    template <DType DT, int Rank> HostView<DT, Rank>    view_output(std::string_view path) const;   // read-only
    /// SUPERSEDED by D139: view_output_mut, set_input and set_output are removed;
    /// Model::init_input(std::string_view path, InitSpec) replaces them.
    /// Writable output view: writes land in host staging; written back
    /// (root.from_host, then `copy_d = root` per other placement) on
    /// destruction AND on commit() (both; the destructor is the
    /// protection, commit() the mid-scope flush).
    /// Why: §1 need 2.3 host access · P3 (a read-only view costs no
    /// submission).
    template <DType DT, int Rank> HostViewMut<DT, Rank> view_output_mut(std::string_view path);
    /// Setters: root.from_host(span) into the root's primary storage,
    /// then `copy_d = root` per other placement, like any root write.
    /// Inputs: meaningful for free inputs (a connected one is overwritten
    /// at the next forward — harmless). A released root throws (§4.4).
    template <DType DT, int Rank> void set_input (std::string_view path, std::span<const cpp_type_t<DT>>);
    template <DType DT, int Rank> void set_output(std::string_view path, std::span<const cpp_type_t<DT>>);

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

    /// Does not wait (D135): call wait() before destroying or reassigning.
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
- (Superseded by D139: `set_*` and `view_output_mut` are removed; initial
  values go through `init_input`.)
- `set_*` = `root.from_host(span)` then `copy_d = root` per other placement.
- `view_output_mut` stages on the host and does `from_host` on commit and
  on destruction.
- `forward()` enqueues; `wait()` = `queue_.wait_and_throw()`.

### 9.3 Sample — the whole user-side lifecycle

```cpp
auto model = create_model<Top>(ModelConfig{});          // default device, no pools
model.setup();                                          // declare → exec-info → substrates
                                                        //   → prepare → compile → marks (no allocation yet)
model.init_input("mda.z", init::from_host(std::vector<double>{1.0, 2.0}));  // drive a free input (D139)
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
the user's obligation. `~Model` does not wait; call `wait()` before destroying or reassigning a Model (D135).
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
gating, counting, or change detection**: those are ordinary System
subclasses — provided in the library where common, written by users
where not — and their entire mechanism is an early return from
`run_compute_forward`. RunNSystem, the setup-step pattern, lives inside
the framework; do-once is RunNSystem with n = 1, not a separate type.
Why: §1 need 2.2 possibility floor (do once / run N / run when changed /
run if) · P5.

```cpp
namespace naga::cospan {

/// Run the subtree on the first n forwards, then never (n >= 1; n == 0
/// throws std::invalid_argument). Do-once is n = 1: construction-time
/// operators (LBVH, LRPIM, nonlocal-operator assembly). Users derive from it,
/// pass n to its constructor and override setup() like any System. An early
/// return means its components neither allocate nor release (§4.4): when the
/// subtree is transient, its roots are allocated and released in the runs,
/// and after the n-th run its transient roots stay released. Roots kept by a
/// consumer outside persist.
class RunNSystem : public System {
    std::size_t remaining_;
public:
    explicit RunNSystem(std::size_t n);
protected:
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        if (remaining_ == 0) return nullptr;   // nothing submitted
        --remaining_;
        return System::run_compute_forward();  // the default loop
    }
};

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

These are added like any realized type: `add_subsystem<Operators>("operators")` (Operators derives from RunNSystem and passes n),
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

**Assumption stated once:** the model is ACYCLIC. Cycles (registration
order; implicit components) are on the roadmap; their design is where any
ordering exception below lives. Why: §1 need 2.2 cycle.

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

**The resolution pass.** Once the root System's `setup()` returns, the root
runs `resolve()` over the final tree and sets (§4.4). Per root it records
its `transient` flag (a root is transient iff EVERY registration in its
alias set is), its `init` (highest declaring level wins; a tie at one depth
is an error; an init on an output is an error; an init on a root with a
source is ignored; none means zeros), and, for a transient root, its
`last_user` (the last Component, in depth-first registration order, that
touches it). A mark on a transient root is an error here. It replaces the
transient marking of earlier drafts.

**State:** final tree and exec order; final sets; marks; each root's
transient flag, init and last user. No storage.

#### 11.2.2 Exec info — `bytes = root.get_exec_info(default_device)`

`System`: `ctx.device` = own `property::device` else `parent_device`; seed
`{device: 0}`; process **own alias entries first** (a tie alias root
counts its static bytes here); then each child in order, merging returned
maps (sum on key).

`Component`: device = `parent_device`; every own entry's placement is
(`parent_device`, the Component's default storage kind — host kind for a
host body); per own entry `e`:
- root → register this placement as `e`'s primary; count its static bytes
  here (zero for a transient root, and for a host-kind primary);
- input peer, `root(e)` already has storage registered at this placement →
  nothing;
- input peer, it does not → register this placement on `root(e)` and count
  the root's static bytes here when the placement is device kind and the
  root is not transient (the root will own a copy at this placement; §4.2);
- output peer (only a tied output is one) whose root's placement ≠ this
  placement → ERROR naming the tie: producers cannot write one storage
  from two placements.

Every root is processed before any peer of it: alias roots before
children, producers before consumers (acyclic). A System processes its
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
recurse. NOTHING IS ALLOCATED here: every root is allocated by its OWNER
Component in `run_pre_compute` (§6.5), or by a host write (§4.4). The arena
is created before this step (§11.2.3), sized by `get_exec_info`, and stays
empty until the owners run.

`Component`: capture the queue and substrates, bind the substrates to each
own root (so the one allocation routine, §4.4, can draw on them), apply the
shape updates that need no value (`FromSourceDim` of a connected input),
point the wrapper of every member whose root happens to have storage, and
validate static input peers against their roots. An unconnected input
carrying a `FromSourceDim` dimension has no source to derive from: ERROR here
("source-derived dimension on an unconnected input").

#### 11.2.5 Compile — `root.compile()`

`Component`: device function(s) + id at the body's own rank
(`kernel_rank_v`); individual config; a custom-op or host body compiles
nothing. `System`:
children in order; a FusedSystem also checks its structure and builds its
sequence config: it has children, none a custom-op or host body, all on its
device and of one kernel rank, and no value-driven size (`FromInput`) whose
driving input aliases another stage's output in the same fuse
(`FromInputDim` and `FromSourceDim` are always allowed; §5.8, §7.5). The
rest of the Fused and Parallel legality is checked each forward on the
descriptions the children return (cheap, host-side, and it is where the
information is).

**State:** no root has storage yet: every root is allocated by its OWNER in its
first forward (or by a host write, §4.4); wrappers of unallocated roots are
unset; kernels compiled. The model can run. Its state is not meaningful
until the first forward.

#### 11.2.6 Marks

Model walks `children()` depth-first in declaration order and reads
`design_var()`/`constraint()`/`objective()` off every root entry into its
three cached lists (§8.4). **State:** the marks an optimizer reads.

### 11.3 `model.forward()`

`Model::forward()` → `root.forward()` → `root.run_compute_forward()`.
Enqueue only. `model.wait()` (`queue_.wait_and_throw()`) waits the stream
when the caller wants the results on the host; a caller looping N forwards
need not wait between them — the stream orders (§7.3).

#### 11.3.1 A System's turn — the ordering contract (§7.3)

```
for child in children:
    child.run_pre_compute(ctx)                // a Component: shapes, then allocate its roots + apply inits
    d = child.run_compute_forward()
    if d:  d->submit(ctx.target())            // this System submits;
                                              //   routed on d->is_custom_op() / d->is_host()
    child.run_post_compute(ctx)               // a Component: other-placement sync, then release the
                                              //   transient roots it is the last user of
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
`run_compute_forward`; the components under an early return neither
allocate nor release. No System flavor revives or releases anything: that
is the Components' own, in their pre- and post-compute (§11.3.2).

#### 11.3.2 A Component's turn (§6.5)

1. `run_pre_compute(ctx)`: for each own
   entry (input or output) whose shape a `FromInput` spec drives — read the driving input's value on the host (the one
   forced sync, so the launch geometry below is known), `root.resize(shape)`,
   `point_wrapper`; nothing submitted for the resize, no reallocation
   inside a submitted task. Then `pre_forward()` → dynamic validation →
   wrapper sync → ALLOCATE every root this Component touches that has no
   storage, and apply its init when it is a free input root (§4.4).
2. `run_compute_forward()` → `compute_forward()` → description (kernel,
   custom op or host body). The owner submits it.
3. `run_post_compute(ctx)`: for each output root with storage at other
   placements, for each other placement `copy_d = root` — one assignment
   resizes to match and copies, routed by SlabKit — after the owner's
   submission, ordered by the stream. Then RELEASE the transient roots for
   which this Component is the last user: each waits on the stream and is
   freed (D135), with one stream wait for the batch.

**State:** every output root holds this forward's values; dynamic roots
sized to this forward; a transient root exists only between its first
user's pre-compute and its last user's post-compute.

### 11.4 Host access

`view_*` (device-kind root: `root.host_view()`; shared/host-kind root:
`queue_.wait()` then `root.get_access()`; the writable output view stages
on the host and does `from_host` on destruction and on `commit()`),
`set_*` (`root.from_host(span)`, then `copy_d = root` per other placement),
`memory_report()`, `get_subsystem(path)` chaining and
`get_inputs()/get_outputs()` enumeration (§10.3). Host access is §9.2. Host
access to a released root (§4.4) is an error naming the root and the
transient subsystem whose registration made it transient.

### 11.5 End of life

The layering rule (§9.4): each layer orders what it submits; teardown
relies on AdaptiveCpp queue destruction, and waiting before exit is the
user's obligation. `~Model` does not wait; call `wait()` before destroying or reassigning a Model (D135). Everything
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

