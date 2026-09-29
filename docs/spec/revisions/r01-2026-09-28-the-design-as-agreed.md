# CoSpan specification — r01, 2026-09-28: the design as agreed

**What this is.** The CoSpan design as agreed on 2026-09-28. It is
unverified by code. When the code disagrees, a new revision is cut and this
file is never edited. Discussion decisions live in the project knowledge;
this document holds only what was agreed. Notation: `sk::` is SlabKit's
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
aliasing correctness · slice and assembly connections · marks · index-
map-valued I/O · non-durable I/O · initialization · host access ·
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
  ordered with everything else. Floor: a component body that receives
  the native handle and raw device pointers and does whatever it likes.
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
  spectrum: identity → slice (a region of the source, zero-copy) →
  **assembly** (several producers' regions into one input, disjoint,
  with coverage checked by default) → general index map (computed
  gather). The only way I/O aliases across the input/output divide.
- **`promote`** — an I/O given a name in an owning system's namespace;
  always allowed, including for I/O that is already connected or
  aliased, so a higher level can always name what it needs.
- **`tie_inputs` / `tie_outputs`** — many same-kind I/O bound to one
  alias. Same kind only. A tie precedes any connection of its members;
  afterwards only the alias connects, and connecting through a member
  name is refused so that the tie is visible at every use site.
- **Index-map-valued I/O** — index maps are *data produced by components*
  (neighbour searches, sort permutations, sparse operator structure) and
  consumed by gathers, scatters and connections. Connectivity is
  computable by the model.
- **Non-durable I/O** — a lifetime attribute on declarations: durable, or
  scoped to a forward.
- **Chunked execution over bounded scratch** — N independent instances
  whose per-instance intermediate scratch would cap problem size if
  materialized at once (per-node dense solves, operator fits): run in
  chunks over reused transient scratch, results landing in the
  full-shaped output. A first-class pattern with its own ergonomics;
  touches shapes and transient lifetime. (Not specified in this revision;
  see the roadmap.)
- **Initialization** — free inputs and outputs receive initial values
  without a component existing to produce them; an owner can override
  initialization anywhere in its subtree by path.
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
///     model error the tie check (§4) and the fused-parallel
///     producer-consumer check (§7.5) catch.
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

/// Initializer<DT, Rank> — the writable twin of an I/O for init(): the
/// same accessor discipline as Output (bare references for plain dtypes,
/// sycl::atomic_ref for atomic dtypes), over the entry's root storage on
/// the Component's device. Obtained inside init() only (§6.2); captured
/// by value into a kernel the user launches on the queue init() receives.
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

/// HostInitializer<DT, Rank> — a host buffer shaped like the entry.
/// Element access reads and writes the buffer, never the device.
/// commit() and destruction write the buffer into the entry's root
/// storage (usm_vector::from_host, which owns its copy of the bytes
/// before returning) and refresh the root's copies on other devices.
/// Host-side only; not a kernel capture. Obtained inside init() only
/// (§6.2).
template <DType DT, int Rank>
class HostInitializer {
public:
    using value_type = cpp_type_t<DT>;
    static constexpr DType dtype = DT;
    static constexpr int   rank  = Rank;

    value_type& operator[](sycl::id<Rank> idx);
    value_type& operator[](size_t flat);
    sycl::range<Rank> shape() const;
    size_t            size() const;
    void commit();
    ~HostInitializer();   ///< commit()

private:
    explicit HostInitializer(detail::IOEntry& entry);
    detail::IOEntry*        entry_;
    std::vector<value_type> host_;
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
enum class Lifetime : uint8_t { durable, transient };
    ///< transient = valid within the owning System's forward invocation;
    ///  storage comes from the dynamic chain, released at scope end.

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
/// input's host_view() in the declaring Component's own run_pre_compute
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
                   Specs... specs, Lifetime life = Lifetime::durable,
                   Coverage cov = Coverage::required);   // §8.2.4;
                   // why: §1 need 2.3 assembly, §1 need 2.3 initialization · P8

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
    Lifetime lifetime() const;

    // ---- shape ----------------------------------------------------------
    sycl::range<3> extents() const;  ///< 1-padded; flat-size arithmetic ONLY
    sycl::range<3> stride() const;   ///< element step in the owning allocation
    size_t flat_size() const;
    const DimRecord& dim(int d) const;  ///< d < rank()
    bool is_dynamic() const;         ///< any dim not How::fixed

    // ---- marks (semantics: §8) ------------------------------------------
    bool is_design_var() const;
    bool is_constraint() const;
    bool is_objective() const;
    /// CONSTRAINT: a design-var entry SHALL NOT be dynamically shaped.

    // ---- the aliasing set (mechanics: §4) -------------------------------
    /// Every entry belongs to one set with one ROOT (data owner and, for
    /// outputs, producer); peers are names for and consumers of the
    /// root's data. There are no source/target links.
    IOEntry* root();                 ///< self iff is_root()
    bool     is_root() const;
    bool     can_be_source() const;  ///< construction-time (§4.2)
    bool     can_be_target() const;
    /// Device placement and per-device root storage — device(),
    /// storage_on(), storage_devices() — are declared in §4.

    // ---- capabilities ---------------------------------------------------

    /// Automatic shape update: applies the DimRecords. Invoked by the
    /// OWNING COMPONENT in run_pre_compute, BEFORE connection validation.
    /// FromInput dims are resolved there first (the driving input's
    /// host_view(), resize, point_wrapper — §6.5).
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

    /// Allocation, in prepare_exec: ROOTS ONLY. Builds the root's
    /// slabkit::usm_vector for each device the set spans, each from that
    /// device's substrate (§7.1): durable ∧ fully-fixed → the arena route
    /// (its bytes are what get_exec_info counted); everything else → the
    /// pool route, with the OS as fallback, at first need.
    virtual void allocate(sk::queue&, Substrates&) = 0;

    /// Write {data pointer, shape, stride} into the registered member
    /// Input/Output wrapper — the pointer being
    /// root()->storage_on(device()) (+ view offset). This is what makes
    /// the user's member variable usable; called at the end of
    /// prepare_exec and after every resize.
    virtual void point_wrapper() = 0;
};

/// Typed leaves. InputEntry holds Input<DT,Rank>* to the member variable;
/// OutputEntry holds Output<DT,Rank>* plus, per device, the owning data
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
    // ---- identity, shape, marks, lifetime: unchanged from §3.3 ----------

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
    /// THE ROOT HOLDS ALL STORAGE, PER DEVICE. A root owns one allocation
    /// on its own device (the primary, where its producer writes) and one
    /// on every other device any peer of its set lives on. Peers own
    /// nothing: every peer reaches up to root()->storage_on(its device),
    /// so peers on the same device — however they relate to each other —
    /// share one pointer, and get_output("alias") from a System always
    /// yields memory on that System's device. The root's producer copies
    /// primary -> each other device's storage in run_post_compute (and
    /// whoever writes a root from the host does the same). Registered in
    /// get_exec_info, allocated in prepare_exec. Why: §1 need 2.2
    /// placement · P1 (a name yields memory where the reader is).
    void*  storage_on(const sycl::device&) const;            ///< roots only
    std::span<const sycl::device> storage_devices() const;   ///< roots only; primary first

    // ---- capabilities (§3.3, restated against the set) ------------------
    /// validate_connection: INPUT PEERS ONLY, against root() — flat-size
    /// (or view() subarray size) equality; static pairs once in
    /// prepare_exec, dynamic pairs every forward after update_shape.
    /// allocate(sk::queue&, Substrates&): ROOTS ONLY — one usm_vector per
    /// device in storage_devices(), each from that device's substrate;
    /// no-op on peers.
    /// point_wrapper: root()->storage_on(device()) (+ view() offset/
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
snapshotted.

**connect(src, dst)** — check `src.can_be_source ∧ root(src).can_be_source
∧ dst.can_be_target ∧ root(dst).can_be_target`, else throw naming the
failing level. Then `merge(root(src), root(dst))`. No flag changes: the
second connection to any name in the set now fails at the ROOT check
(the root is an output root, `can_be_target == false`) — fan-in is
blocked structurally. Fan-out is the same op repeated. Subarray forms
record `view()` on the input peer. Connecting through ANY name of a set
is the same connection.

**tie_inputs({m...}, name)** — the alias is a NEW ROOT with
`can_be_target = true, can_be_source = false`. Each member's root must be
an input root with `can_be_target == true` — a member whose set is
already connected fails here ("tie before connecting"). Merge every
member's set into the alias. THE EXCEPTION: mark each member
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
each member `can_be_source = false`.

Path resolution for every operation is from the caller's namespace,
dot-separated, descending children; the last segment resolves an alias
registered in that System or a Component's declared entry.

### 4.4 What owns memory

Every entry that is a root once declaration completes owns storage —
including unconnected inputs and unconnected `tie_inputs` aliases: that
is how top-level design variables and shared inputs get memory with no
IndepVarComp (why: §1 need 2.3 initialization, §1 need 2.4 design
variables). Route by lifetime (§7.1 `MemorySubstrate`): durable ∧ static
roots from the arena; all other roots via the pool route (OS fallback) at
first need and on growth. Peers own nothing, ever: a root holds one
allocation per device its set spans (§4.2).

### 4.5 From sets to storage

Aliasing is complete the moment each structural call returns. Storage
follows in the setup chain (§11): `get_exec_info` registers on
each root the devices its set spans and counts the root's bytes once per
such device; `prepare_exec` allocates each root's per-device storage from
the substrates and points every wrapper at
`root()->storage_on(own device)`. Regrowth of a root needs no
notification: peers re-point at their next turn. Non-primary device
copies are refreshed by the root's producer after every launch.

`tie_outputs` members MUST share the alias's device: producers cannot
write one storage from two devices. Checked in the members' Components'
`get_exec_info` (a non-root output whose root's device differs from the
Component's own — the alias was processed first by its System, so the
device is known), error naming the tie.

### 4.6 Sample

```cpp
// declared: d1.out (root O1), d2.out (root O2), c.in (root I1)
promote("d1.out", "y");       // y joins O1's set as a peer; flags = O1's
connect("y", "c.in");         // O1 absorbs I1's set: {y, c.in} peers of O1
connect("d2.out", "c.in");    // FAILS at the root check: root(c.in) = O1,
                              //   an output root, can_be_target == false
tie_outputs({"d1.out","d2.out"}, "sum");   // FAILS: O1's set has an
                              //   input-kind peer (c.in) — connected already
```

---

## 5. Kernel machinery

Binding contract; declarations only. Wrappers: §2. Metadata/shapes: §3.
Component: §6. System-side execution: §7.

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
///          // or (sycl::nd_item<R>, Inputs, Outputs)
///          // or (sycl::nd_item<R>, Inputs, Outputs, Scratch)
///          // or (CustomOp&, Inputs, Outputs)        — CUSTOM OPERATION
///   };
///
/// CUSTOM-OPERATION BODIES are never compiled to a device function or
/// launched: the owner submits them via AdaptiveCpp_enqueue_custom_
/// operation through the same SubmitTarget as kernels (§7.1) — so the
/// stream orders them like any other submission (§7.3) and gating applies
/// unchanged.
/// Inputs/Outputs arrive as the bare wrappers with CURRENT device
/// pointers. What the body does is open: host work, or handing the
/// pointers to a native library on the native stream (CustomOp::h is
/// the interop handle; CustomOp::to_host/from_host move data through
/// the native backend for pure host work). Not fusable (no device
/// function exists).
///
/// THE SIGNATURE IS THE DECLARATION, four ways:
///   * first parameter id<R> vs nd_item<R>  => group semantics opt-in
///     (a body without the nd_item cannot touch group abstractions);
///   * R                                    => kernel rank, deduced
///     (kernel_rank_v<KB>; no static member);
///   * trailing Scratch parameter           => local-memory opt-in
///     (nd_item bodies only — local memory without a group is
///     meaningless).
/// Inputs, Outputs, KB itself, and Scratch (if present) SHALL be
/// trivially copyable — concept-enforced; they are copied byte-wise
/// into the launch's device streams.
/// Bodies spell element types as the wrappers' `value_type` (§2); nothing
/// in a body is written for differentiation.
template <class KB> concept PlainBody   = /* operator()(id<R>, In, Out) */;
template <class KB> concept GroupBody   = /* operator()(nd_item<R>, In, Out [, Scratch]) */;
template <class KB> concept HasScratch  = /* trailing Scratch parameter */;
template <class KB> concept CustomOpBody= /* operator()(CustomOp&, In, Out) */;
template <class KB> concept KernelBodyC =
    std::is_trivially_copyable_v<KB> &&
    std::is_trivially_copyable_v<typename KB::Inputs> &&
    std::is_trivially_copyable_v<typename KB::Outputs> &&
    (int(PlainBody<KB>) + int(GroupBody<KB>) + int(CustomOpBody<KB>) == 1);
template <class KB> inline constexpr int kernel_rank_v = /* from signature */;

/// What a custom-operation body receives: the interop handle (native
/// queue/stream/device of the device this submission targets)
/// and two helpers for pure host work — device USM is not host-
/// dereferenceable, so data moves through the native backend on h's
/// stream. Anything else (a library call on the native stream, a native
/// kernel launch) is the body's own business.
struct CustomOp {
    sycl::interop_handle& h;
    /// THE ONE PLACE a raw device pointer is obtainable (§2: wrappers
    /// never expose one). A custom operation exists to hand memory to
    /// code outside the framework; this is that door, and it opens only
    /// inside a custom-op body. Why: §1 need 2.1 work outside device
    /// kernels, §1 need 2.1 minimal device payload.
    template <class W> auto device_ptr(const W& io) -> typename W::value_type*;
    template <class W> auto to_host(const W& in) -> std::vector<typename W::value_type>;
    template <class W> void from_host(W& out, std::span<const typename W::value_type>);
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

### 5.3 StageRecord and FnOffsetInfo

```cpp
namespace naga::cospan {

/// One StageRecord per stage per launch, fixed-size, packed into a
/// device array in EXECUTION ORDER. Field ownership:
///   component-bound (create_launch_config): rank, range, flat_extent,
///     stream offsets, scratch_off/scratch_bytes;
///   System-stamped (launch pack): group_offset, group_count — the
///     GROUP-INDEX slice: slices are counted in whole work-groups, so a
///     group never straddles stages under ANY geometry. Why: §1 need
///     2.2 fuse, §1 need 2.1 group-level parallelism.
struct StageRecord {
    sycl::range<3> range;         // logical iteration space, 1-padded
    size_t   flat_extent;         // product of range
    size_t   group_offset;        // System-stamped
    size_t   group_count;         // System-stamped
    uint32_t input_off;           // into the inputs stream
    uint32_t output_off;          // into the outputs stream
    uint32_t body_off;            // into the bodies stream
    uint32_t scratch_off;         // into the bodies stream (Scratch bytes)
    uint32_t scratch_bytes;       // this stage's local-memory requirement
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

### 5.4 The generated device function — one per (KernelBody type, rank)

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
    // nd_item bodies: receive the full padded slice; ragged-tail
    // handling is the standard obligation of any group-using nd_range
    // kernel. Their freedom is bounded by SYCL 2020 §4.17.3, verbatim:
    // group functions "must be encountered in converged control flow by
    // all work-items in the group ... under the same set of conditions"
    // — skipping one via early exit, or calling it from divergent
    // branches, is UB. Divergence is legal in code free of group
    // functions. Guard the loads, never the participation. We spec to
    // the spec, not to hardware tolerance. Why: §1 need 2.1 group-level
    // parallelism, §1 need 2.2 opacity (kernels are never written for fusion).

    const auto idx = detail::delinearize<kernel_rank_v<KB>>(flat, r.range);
    auto in   = *reinterpret_cast<const typename KB::Inputs*>(inputs  + r.input_off);
    auto out  = *reinterpret_cast<const typename KB::Outputs*>(outputs + r.output_off);
    auto body = *reinterpret_cast<const KB*>(bodies + r.body_off);

    if constexpr (HasScratch<KB>) {
        auto s = *reinterpret_cast<const typename KB::Scratch*>(bodies + r.scratch_off);
        detail::fixup_scratch(s, local_base);  // Local.ptr_ = base + offset_
        body(it, in, out, s);
    } else if constexpr (GroupBody<KB>) {
        body(it, in, out);
    } else {
        body(idx, in, out);
    }
}

} // namespace naga::cospan
```

### 5.5 The chained executor

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
                                 // max over stages; zero when unused
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
///                    {fn_A_id, fn_B_id, barrier_id, fn_C_id, ...});
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
/// Barrier stages: when any stage of a SEQUENTIAL-mode wave declares
/// Scratch, the System interleaves the provided
///     void kb_barrier(...same signature...) {
///         sycl::group_barrier(it.get_group());   // does NOT call next()
///     }
/// between consecutive scratch-using stages: the call site is
/// unconditional, so every work-item reaches it in converged control
/// flow (legal), and stage N+1 cannot touch local bytes stage N still
/// reads (necessary). kb_barrier consumes NO record — the one deliberate
/// exception to one-call-one-record; the packer writes records only for
/// real stages. Parallel-mode waves need no barriers: each group
/// executes exactly one stage's body, so scratch never crosses stages.
/// Why: §1 need 2.2 fuse, §1 need 2.1 group-level parallelism.

} // namespace naga::cospan
```

### 5.6 Geometry and slices

- All launches are typed by rank (the executor's switch); the default
  geometry is flat: `{{padded_flat}, {default_wg}}` at rank 1, with
  rank-appropriate defaults at ranks 2/3 (implementation-tunable, not
  contract constants). The wg overloads of create_launch_config (§6)
  override per launch; FUSED rank and wg are set at SYSTEM level,
  component wg intent ignored inside a fuse.
- Rank agreement in a fuse: an nd_item stage whose rank differs from the
  wave's rank is a hard error, checked on the descriptions each forward
  (§7.5); id-only stages are rank-exempt (their device function is
  generated for the wave's rank; SSCP lowers only what is invoked).
- Padding and extents are counted in WHOLE WORK-GROUPS: individual = the
  stage's own group count; fused-sequential = max stage group count;
  fused-parallel = Σ stage group counts with group_offset the prefix
  sum. A group never straddles stages, under any geometry — this is
  what makes the slice check group-uniform and collectives safe.
- One dynamic_function_config per launch; the config object outlives
  kernel completion; acpp JIT-caches per configuration, persistently
  across runs (acpp laws).

### 5.7 Dynamic shapes and the pipeline

- **Resolution is just-in-time.** A Component resolves shapes in
  run_pre_compute at its own turn in depth-first order — upstream has
  completed by construction. Static models pay nothing anywhere in this
  section.
- **FromInput resizes are the consumer's own business.** In its
  `run_pre_compute`, the consuming Component reads the driving input
  root's `host_view()` (the one forced sync, so the launch geometry is
  known at enqueue), calls `root.resize(shape)`, then `point_wrapper` —
  no staging copy, nothing submitted for the resize, no reallocation
  inside a submitted task. Nothing is registered anywhere; no producer knows its consumers.
- **FusedSystem: no intra-fuse shape dependencies — hard error.** A
  wave's geometry needs every stage's extent before the wave launches; a
  stage whose shape derives, through any determiner-or-connection chain,
  from data computed by another stage of the same fuse is an
  impossibility, not a trust question. Detected from metadata each
  forward alongside the other wave checks (§7.5); the error names the
  crossing chain. USERS SPLIT THEIR FUSEDSYSTEMS AT DYNAMIC-SHAPE
  BOUNDARIES. Why: §1 need 2.2 fuse, §1 need 2.1 shape source.
- **ParallelSystem: no cross-sibling shape dependencies — hard error.**
  A child's shape reading a sibling's output (by value or shape) is a
  data dependency, contradicting the independence the user asserted by
  wrapping in ParallelSystem. Detected from the same metadata chains.
  General data-independence of parallel children remains the user's
  responsibility, unchanged. Why: §1 need 2.2 parallel · P4.

---

## 6. Subsystem and Component

Binding contract; declarations only. Wrappers: §2. Metadata/shapes: §3.
Kernel machinery: §5. System: §7. The end-to-end lifecycle: §11.

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
    /// Allocate what can be allocated now and point every wrapper at its
    /// storage (own allocation, else root()->data() — roots are allocated
    /// before their peers); Systems complete their ExecCtx.
    virtual void prepare_exec(sk::queue& queue, Substrates& substrates) = 0;
    /// Compile bodies (Components); recurse (Systems). Flavor legality is
    /// checked per forward on the descriptions, not here.
    virtual void compile() = 0;
    /// Initialize: children before self (Systems), so an owner's init
    /// has the last word by path.
    virtual void run_init(const ExecCtx& owner_ctx) = 0;

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
/// is_custom_op(): kernels take the outer-kernel path (§5.5), custom
/// operations take AdaptiveCpp_enqueue_custom_operation through the same
/// SubmitTarget. A kernel description additionally exposes rank,
/// group-body and scratch facts and its streams/record, for fused owners
/// to validate and pack.
class LaunchDescription {
public:
    virtual ~LaunchDescription();
    virtual bool is_custom_op() const = 0;
    virtual void submit(SubmitTarget& t) const = 0;   ///< ordering is the stream's (§7.3)
    virtual const KernelParts* kernel() const { return nullptr; }  // null for custom ops
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
    /// Initialize own I/O. Runs once, after allocation and compilation,
    /// children before owners. Not limited beyond what `q` permits:
    /// write with an initializer inside your own kernel, fill, copy —
    /// anything (anything launched over a root outside the framework is
    /// ordered by its launcher, §9.4). Initializing a connected input is
    /// harmless: the producer overwrites it at the first forward. Default
    /// no-op.
    /// Why: §1 need 2.3 initialization · P2 (an adopted subsystem's
    /// inits must not need editing when its inputs get connected).
    virtual void init(sk::queue& q, const sycl::device& dev) {}
    /// Per-forward manual shape control of a subset of entries; runs
    /// inside the automatic pre-compute after automatic derivation.
    /// Default no-op.
    virtual void pre_forward() {}

    // ---- declaration API (semantics: §3; PUBLIC, OpenMDAO-style) ------
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_input(Input<DT, Rank>& m, std::string name, Specs... specs,
                       Lifetime life = Lifetime::durable,
                       Coverage cov = Coverage::required);
    template <DType DT, int Rank, DimSpec... Specs>
        requires(sizeof...(Specs) == Rank)
    void declare_output(Output<DT, Rank>& m, std::string name, Specs... specs,
                        Lifetime life = Lifetime::durable);
    template <DType DT, int Rank>
    void declare_input(Input<DT, Rank>& m, std::string name, from_input_array_t,
                       Lifetime life = Lifetime::durable,
                       Coverage cov = Coverage::required);
    template <DType DT, int Rank>
    void declare_output(Output<DT, Rank>& m, std::string name, from_input_array_t,
                        Lifetime life = Lifetime::durable);

    /// pre_forward-slot-only resize; same setter as automatic derivation.
    template <DType DT, int Rank> void reshape(Input<DT, Rank>& m, sycl::range<Rank>);
    template <DType DT, int Rank> void reshape(Output<DT, Rank>& m, sycl::range<Rank>);

    // ---- retrieval & enumeration ------------------------------------------
    template <DType DT, int Rank> Input<DT, Rank>&  get_input(std::string_view name);
    template <DType DT, int Rank> Output<DT, Rank>& get_output(std::string_view name);
    auto get_inputs() const;    ///< range of const detail::IOEntry& (§3.3 guarantee;
                                ///  why: §1 need 2.3 host access)
    auto get_outputs() const;

    // ---- initializers (legal inside init() only; else throw) --------------
    /// Device initializer: the writable twin of Input/Output with Output's
    /// accessor discipline (§2) — usable directly inside a kernel the user
    /// writes on `q`.
    /// Why: §1 need 2.3 initialization (not limited beyond the queue).
    template <DType DT, int Rank> Initializer<DT, Rank> get_input_initializer(std::string_view name);
    template <DType DT, int Rank> Initializer<DT, Rank> get_output_initializer(std::string_view name);
    /// Host initializer: a host buffer shaped like the entry (§2); commit()
    /// and destruction write it to the root and refresh the root's
    /// other-device copies.
    template <DType DT, int Rank> HostInitializer<DT, Rank> get_input_host_initializer(std::string_view name);
    template <DType DT, int Rank> HostInitializer<DT, Rank> get_output_host_initializer(std::string_view name);
    /// Common-case helpers, by name.
    /// Every helper writes the ROOT's storage on this Component's device,
    /// then refreshes the root's other-device copies (`copy_d = root`,
    /// §6.5 step 3).
    template <class T> void fill_input (std::string_view name, T value);
    template <class T> void fill_output(std::string_view name, T value);
    void zero_input (std::string_view name);
    void zero_output(std::string_view name);
    template <class T> void set_input_from_host (std::string_view name, std::span<const T>);
    template <class T> void set_output_from_host(std::string_view name, std::span<const T>);

protected:
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
    /// Base-provided staging. Kernel bodies: range[, wg], Inputs, Outputs
    /// [, Scratch], body. Custom-op bodies (§5.1): Inputs, Outputs,
    /// body — no range.
    std::unique_ptr<LaunchDescription> create_launch_config(
        sycl::range<kernel_rank_v<KB>> range,
        typename KB::Inputs in, typename KB::Outputs out, KB body = {});
    std::unique_ptr<LaunchDescription> create_launch_config(
        sycl::range<kernel_rank_v<KB>> range, sycl::range<kernel_rank_v<KB>> wg,
        typename KB::Inputs in, typename KB::Outputs out, KB body = {})
        requires GroupBody<KB>;
    std::unique_ptr<LaunchDescription> create_launch_config(
        sycl::range<kernel_rank_v<KB>> range, sycl::range<kernel_rank_v<KB>> wg,
        typename KB::Inputs in, typename KB::Outputs out,
        typename KB::Scratch scratch, KB body = {})
        requires(GroupBody<KB> && HasScratch<KB>);
    std::unique_ptr<LaunchDescription> create_launch_config(
        typename KB::Inputs in, typename KB::Outputs out, KB body = {})
        requires CustomOpBody<KB>;

private:
    // ---- runner overrides (the private bodies of the Subsystem verbs) ---
    void run_setup() override;                       // -> setup()
    std::unordered_map<sycl::device, size_t>
        get_exec_info(const sycl::device& parent) override;   // §6.4
    void prepare_exec(sk::queue&, Substrates&) override;      // §6.4
    void compile() override;
    void run_init(const ExecCtx& c) override;        // -> init(*c.queue, c.device)
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
  it records `parent_device`, then for each own entry `e`:
  - `e` is root: register this device as its primary; count its static
    bytes here.
  - `e` is an input peer: if `root(e)` already has storage registered on
    this device → nothing; else register this device on `root(e)` and
    count the root's static bytes here (the root owns one copy per
    device its set spans — §4.2).
  - `e` is an output peer (only a tied output is one): `root(e)`'s device
    must equal this device, else ERROR naming the tie — producers cannot
    write one storage from two devices.
  (Every root is processed before any peer of it in an acyclic model —
  Systems process their alias entries before their children, and
  producers precede consumers. Cycles are on the roadmap; their design
  owns the exception.)
  - Return `{device: bytes}`.
- **`prepare_exec`**: for every own ROOT allocatable now, build its
  `usm_vector` per registered device from that device's substrate
  (durable∧static → arena route; durable∧dynamic with known shape → pool
  route with OS fallback); then point every member wrapper at
  `root()->storage_on(this device)` and validate static input peers
  against their roots. An unconnected input with a `FromSourceDim`
  dimension errors here.
- **`compile`**: the device function at the owner's fuse rank if any,
  else native; the individual config; custom-op bodies recorded.
- **`run_init`**: `init(queue, device)` with the Model's queue and this
  Component's device.

### 6.5 A Component's forward turn

Called by its owner in this order; a Component never submits its compute
work and never sees a queue except through the owner's ctx.

1. **`run_pre_compute(ctx)`**: value-driven shapes first — for each own
   input carrying a `FromInput` spec, read the driving input's value
   through `host_view()` — a host read, whichever storage holds it (the
   one forced sync, so this turn's launch geometry is known), call
   `root.resize(shape)`, then `point_wrapper`; no staging
   copy, nothing submitted for the resize, no reallocation inside a
   submitted task (why: §1 need 2.1 shape source · P3 — only value-driven entries pay
   it). Then shape-derived dims, `pre_forward()` hook (per-entry
   override, last write wins), dynamic connection validation, wrapper
   sync (a root that grew was reallocated by its own allocator route).
2. **`run_compute_forward()`**: `compute_forward()` — returns the
   description. The owner submits it (§7.3).
3. **`run_post_compute(ctx)`**: through `ctx`, after the owner's
   submission: for each output root with storage on other devices, for
   each other device `copy_d = root` — one assignment resizes to match
   and copies, routed by SlabKit. Ordered after the work it depends on by
   the stream (§7.3).

### 6.6 Provided kernels — v1

```cpp
namespace naga::cospan::kernels {
template <DType DT, class Op = sycl::plus<cpp_type_t<DT>>>
struct GroupReduce {      // fast; reduce_over_group + leader atomic; accumulates ONTO out
    struct Inputs  { Input<DT, 1> in; };
    struct Outputs { Output<to_atomic_v<DT>, 1> out; };
    void operator()(sycl::nd_item<1>, Inputs, Outputs) const;
};
template <DType DT, class Op = sycl::plus<cpp_type_t<DT>>>
struct OrderedReduce {    // deterministic; one agent, ascending order; fuses anywhere
    struct Inputs  { Input<DT, 1> in; };
    struct Outputs { Output<DT, 1> out; };
    void operator()(sycl::id<1>, Inputs, Outputs) const;
};
// RegionCopy, GatherInto, Scatter: §8.
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
        declare_input (n_bins_, "n_bins",  1);
        declare_output(bins_,   "bins",    FromInput{"n_bins", 0});
    }
    void init(sk::queue&, const sycl::device&) override { zero_output("bins"); }   // accumulate-onto needs a zero
    std::unique_ptr<LaunchDescription> compute_forward() override {
        return create_launch_config(values_.shape(),
            {values_, weights_, n_bins_}, {bins_}, {lo_, hi_});
    }
};

// (b) Group + wg override + Local scratch; nd_item body.
struct TransposeBody {
    struct Inputs  { Input<DType::f32,2> in; };
    struct Outputs { Output<DType::f32,2> out; };
    struct Scratch { Local<DType::f32,2> tile; };
    void operator()(sycl::nd_item<2> it, Inputs in, Outputs out, Scratch s) const {
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
allocation, or a pointer the user manages. `setup()`, `compute_forward()`,
optionally `init()` — and physics.

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

/// One per device (SlabKit spec): the device's arena and optional pool,
/// built from (queue, device) and held for the model's life. Every
/// allocation a subsystem makes goes through its device's substrate.
struct MemorySubstrate { slabkit::DeviceArena arena;
                         std::optional<slabkit::DevicePool> pool; };
using  Substrates = std::unordered_map<sycl::device, MemorySubstrate>;

/// What a submission consumes (§6.1 LaunchDescription::submit): the
/// queue, the group when the submission belongs to a Parallel owner's
/// group, and the device it targets.
struct SubmitTarget { sk::queue* queue; sk::queue::group* group;
                      sycl::device device; };
/// What a fused owner reads off a kernel description (§6.1): the facts it
/// validates and the parts it packs.
struct KernelParts { int rank; bool group_body; bool has_scratch;
                     const StageRecord& record; /* input/output/body streams */ };

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
    virtual void setup() {}                     ///< structure & wiring (§7.4)
    virtual void init(sk::queue& q, const sycl::device& dev) {}   ///< runs AFTER children's; override by path
    virtual void pre_forward() {}

    // ---- Model's entries ------------------------------------------------------
    /// Model knows its root is a System and has no ctx of its own, so the
    /// root supplies its own: init_tree() == run_init(ctx_);
    /// forward() == run_compute_forward() (no pre/post-compute exists for
    /// the root).
    void init_tree();
    void forward();

    // ---- initializers by path (legal inside init() only) ----------------------
    template <DType DT, int Rank> Initializer<DT, Rank>     get_input_initializer (std::string_view path);
    template <DType DT, int Rank> Initializer<DT, Rank>     get_output_initializer(std::string_view path);
    template <DType DT, int Rank> HostInitializer<DT, Rank> get_input_host_initializer (std::string_view path, sk::queue&);
    template <DType DT, int Rank> HostInitializer<DT, Rank> get_output_host_initializer(std::string_view path, sk::queue&);
    template <class T> void fill_input (std::string_view path, T v);
    template <class T> void fill_output(std::string_view path, T v);

    // ---- retrieval & enumeration (hierarchy-aware; §3.3 guarantee) ------------
    template <DType DT, int Rank> Input<DT, Rank>&  get_input (std::string_view path);
    template <DType DT, int Rank> Output<DT, Rank>& get_output(std::string_view path);
    auto get_inputs()  const;   ///< this System's alias entries
    auto get_outputs() const;
    Subsystem& get_subsystem(std::string_view path);          ///< chains through children

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
    /// alias roots and point their wrappers; then children.
    void prepare_exec(sk::queue&, Substrates&) override;
    /// children.compile() in order.
    void compile() override;
    /// children.run_init in order, then init(*ctx.queue, ctx.device).
    void run_init(const ExecCtx&) override;
    void run_pre_compute(ExecCtx&) override {}     ///< Systems do no work
    void run_post_compute(ExecCtx&) override {}
    /// THE DEFAULT LOOP (§7.3).
    std::unique_ptr<LaunchDescription> run_compute_forward() override;

    ExecCtx ctx_;
    std::vector<std::unique_ptr<Subsystem>> children_;   ///< declared order = exec order
    std::unordered_map<std::string, std::unique_ptr<detail::IOEntry>> aliases_;
};

} // namespace naga::cospan
```

### 7.3 The ordering contract

The stream is an abstract in-order sequence of work items. A kernel is an
item (a custom operation, submitted through the same `SubmitTarget`, is
an item the same way). A group (`queue.submit_group`) is one item:
unordered inside, ordered both ways outside. Async host and memory tasks
are concurrent with each other but ordered after every kernel/group item
before them and before every one after. Placement never costs the order.
Nothing in CoSpan threads dependencies.

The default loop is per child:
`run_pre_compute → d = run_compute_forward → if d, d->submit(target) →
run_post_compute`; a child that submitted itself needs no owner action.

```
for (child : children_) {
    child->run_pre_compute(ctx_);
    auto d = child->run_compute_forward();
    if (d) d->submit(ctx_.target());          // WE submit, routed on d->is_custom_op()
    child->run_post_compute(ctx_);            // copies AFTER the work they depend on —
                                              //   by the stream, not by an event
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
    struct fused         { FusedSystem::Mode mode; int rank = 1; std::optional<sycl::range<3>> wg; };
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
    /// A BASE System configured by properties: device and/or ONE flavor
    /// (parallel | fused — both on one call is an error). Returns
    /// the System to add
    /// children to inline — no class derivation needed. Why: §1 need
    /// 2.2 inline composition · P2 (placement without editing).
    System& add_subsystem(std::string name, const property_list& props = {});
    // add_subsystem constructs the child, registers the name, and runs the
    // child's run_setup BEFORE returning, so later structural calls in this
    // same setup() see it fully declared.

    void connect(std::string_view src, std::string_view dst);                 // identity
    void connect(std::string_view src, std::string_view dst, ConnectionMap);  // §8
    void promote(std::string_view child_path, std::string promoted_name);
    void tie_inputs (std::initializer_list<std::string_view> members, std::string name);
    void tie_outputs(std::initializer_list<std::string_view> members, std::string name,
                     TieOutputsPolicy policy = TieOutputsPolicy::require_atomic);
    void add_design_var(...);  void add_constraint(...);  void add_objective(...);   // §8
};

} // namespace naga::cospan
```

Rules (mechanics in §4; semantics in §8): output → input only; an input
has at most one source, enforced at the ROOT's flag after the first
connection; connecting through any name of a set is the same connection;
`promote` is kind-agnostic and always allowed; ties must precede
connections ("tie before connecting, or connect from the alias");
`tie_outputs` requires atomic member dtypes unless `allow_plain`.

### 7.5 ParallelSystem and FusedSystem — the two policy flavors

```cpp
namespace naga::cospan {

/// CHILDREN ARE COMPONENTS ONLY — enforced by the type system at the add,
/// with the same constraint FusedSystem uses (a System cannot discern
/// children at runtime; why: §1 need 2.2 parallel · P5):
///   template <class T, class... A> requires std::derived_from<T, ComponentBase>
///   T& add_subsystem(std::string, A&&...);
/// (no property-list overload). Nested Parallel is not a concept.
/// run_compute_forward: pre_compute ALL children → queue.submit_group {
/// each child's description submitted with its device (Components inherit
/// the Parallel's) through a SubmitTarget carrying the group } →
/// post_compute ALL after the group returns. The group is ONE item of the
/// stream (§7.3): its children are unordered among themselves and ordered
/// both ways against everything outside. ctx.group is set for the extent
/// of that submit_group and null otherwise. Data independence of children
/// is the USER's assertion (why: P4);
/// the framework checks only cross-sibling shape dependencies (hard
/// error, checked each forward on the children's descriptions).
class ParallelSystem : public System { /* runner overrides as described */ };

/// COMPONENTS ONLY — enforced by the type system at the add, since a
/// System cannot discern children at runtime (why: §1 need 2.2 fuse · P5):
///   template <class T, class... A> requires std::derived_from<T, ComponentBase>
///   T& add_subsystem(std::string, A&&...);
/// (no property-list overload). Rank and wg shape are the fuse's;
/// children compile at the fuse rank (id-only bodies lazily); the
/// sequence config is built once at compile.
/// run_compute_forward: pre-compute ALL children → compute ALL (collect
/// descriptions) → VALIDATE on the descriptions, every forward, host-
/// side and cheap: none null, none custom-op, rank agreement for group
/// bodies, scratch handling, parallel mode: no producer-consumer pairs
/// unless allowed → pack streams + records, stamp group slices, one
/// staging, ONE submission with the sequence config → post-compute ALL.
/// The order is forced: post-compute copies follow the wave in the stream.
/// Why: §1 need 2.2 fuse, §1 need 2.2 opacity.
class FusedSystem : public System {
public:
    enum class Mode : uint8_t { sequential, parallel };
    FusedSystem(Mode, int rank = 1, std::optional<sycl::range<3>> wg = {});
    void allow_producer_consumer();   // parallel-mode override (trust model)
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
        tie_inputs({"d1.z", "d2.z"}, "z");
        promote("d1.y1", "y1");  promote("d2.y2", "y2");
    }
    void init(sk::queue&, const sycl::device&) override { fill_input("z", 1.0); }   // last word over d1/d2 inits
};

class Top : public System {
    void setup() override {
        auto& mda = add_subsystem<SellarMDA>("mda", property_list{property::device{gpu0}});
        auto& obj = add_subsystem<Objective>("obj");
        connect("mda.y1", "obj.y1");
        connect("mda.y2", "obj.y2_head", Slice{subarray<1>({0}, {1})});

        auto& par = add_subsystem("post", property_list{property::parallel{}});
        par.add_subsystem<Report>("r1");  par.add_subsystem<Report>("r2");
    }
};
```

---

## 8. Topology semantics and optimization marks

Binding contract; declarations only. Aliasing mechanics: §4. Structural
APIs: §7. `declare_input`'s coverage option: §8.2.4.

### 8.1 The connection spectrum

Three kinds of "dst sees src's data", and they are NOT one mechanism:

- **identity** — `connect(src, dst)`: dst joins the root's set as a peer
  reading the whole root. Zero-copy.
- **whole-input slice** — a rectangular region of the root whose flat
  size equals dst's: dst is a peer whose `view()` is that region, in the
  ROOT's rank; the wrapper gets `root.data + offset` and the root's
  strides (§2). Zero-copy. This is what the stride metadata exists for.
  Why (for the whole spectrum): §1 need 2.3 connection.
- **computed** — everything else: a region of the source into a REGION
  of the input (several sources may feed one input, regions disjoint),
  or an index map (a gather is not affine; a view cannot express it).
  These produce a MAPPING SYSTEM (§8.2): a copy by necessity, made of
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

class System {   // ...continued from §7.4
public:
    void connect(std::string_view src, std::string_view dst);                // identity
    void connect(std::string_view src, std::string_view dst, ConnectionMap); // §8.1/§8.2
};

} // namespace naga::cospan
```

`map` names an index-map-valued I/O: any integer-dtype entry (an output
produced upstream — neighbour lists, CSR columns — or an unconnected
input root set from the host). §1's "index-map-valued I/O" need is
exactly this convention; no dedicated type exists.

### 8.2 The mapping System

#### 8.2.1 Lifecycle

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
  (the trust model, used correctly — P4); connect the tie alias to `dst`
  (identity); run the coverage check (§8.2.4).

#### 8.2.2 Execution position

The mapping System must run after its sources and before whoever reads
`dst`. The post-setup pass INSERTS it into the owner's child list
IMMEDIATELY BEFORE the child that contains the consumer (the first path
segment of `dst`), not at the end. Sources produced by later siblings
are a feedback loop the model already has; solvers own that,
and the insertion rule does not change it. Why: §1 need 2.2 sequence.

#### 8.2.3 Mapping components

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

#### 8.2.4 Coverage

Uncovered elements of `dst` hold whatever the tied storage holds — which
may be exactly what the owning component initialized, deliberately. So
coverage is checked BY DEFAULT and opted out per input at declaration
(why: P8, §1 need 2.3 initialization):

```cpp
enum class Coverage : uint8_t { required, optional };
/// Every declare_input form (§3.3/§6.2) carries a trailing option
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

### 8.3 View composition

Views compose by offset addition with containment: connecting from a
name whose peer carries view `B`, with a `Slice{A}`, yields
`{B.offset + A.offset, A.extents}` and requires `A ⊆ B` (error naming
both; why: §1 need 2.3 connection · P1). Promoting a peer that carries a view produces an alias that
COPIES the view — the alias presents what the promoted entry sees. All
views are in the root's rank; there is no rank reinterpretation through
a view (rank is authoritative, §3.1).

### 8.4 Cross-level ties

Nothing new: members resolve from the caller's namespace at any depth,
and a member whose root is already a tie alias (a child-level tie) is a
legal member when that alias is unconnected — the merge makes the child
alias a peer of the new root, and the `tie_*` exception marks it like
any named member. Ties layer upward without special rules.

### 8.5 Optimization marks

```cpp
namespace naga::cospan {

/// Per-element or scalar bounds/scaling; host-side vectors sized to the
/// marked region's flat size (scalar forms broadcast).
struct Bounds  { std::optional<std::vector<double>> lower, upper; };
struct Scaling { std::vector<double> ref{1.0}, ref0{0.0}; };   ///< semantics: with the optimizers

class System {   // ...continued
public:
    /// MARKS RIDE ON THE ROOT: marking any name marks the data. Recorded
    /// on the root's IOEntry (§3.3 is_design_var/…), enumerated by
    /// Model. `region` restricts a mark to a subarray (root rank) via
    /// the same view machinery; no index maps on marks. Why: §1 need
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

### 8.6 Samples

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
/// and the root System. It drives the setup chain and the forward; it
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
    /// optimization; memory_report() is how it is discovered.
    /// Why: P3.
    std::unordered_map<sycl::device, size_t> pool_bytes;
    size_t default_pool_bytes = 0;
    size_t pool_node_size     = 1024;
    /// The default device (root System's device unless it overrides;
    /// what every System inherits by default). Defaults to the runtime's
    /// default selector. Validated at Model construction:
    /// ctx.contains(*default_device), else throw.
    std::optional<sycl::device> default_device;
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
    ///      queue_ = sk::queue{ctx_, default_device, handler} — ONE queue,
    ///      one in-order stream (§7.3), existing for everything below;
    ///      `handler` is the async-error handler that wait() surfaces
    ///   1. root.run_setup()                       — declare (§6/§7)
    ///   2. bytes = root.get_exec_info(default_device)   — device -> bytes
    ///   3. substrates[d] = { DeviceArena{queue_, d, kind::device, bytes[d]},
    ///      DevicePool iff configured } — every subsystem allocates
    ///      through its device's substrate
    ///   4. root.prepare_exec(queue_, substrates_) — allocate, point wrappers; ExecCtx complete
    ///   5. root.compile()                        — bodies
    ///   6. root.init_tree()                      — children before owners
    /// After setup the model's STATE is not yet meaningful: it holds
    /// initialization. State is defined after the first forward.
    /// Why: §1 need 2.3 initialization.
    void setup();

    /// root.forward(): enqueue one forward. Returns immediately. Loop it
    /// freely — the stream orders (§7.3).
    void forward();
    /// queue_.wait_and_throw(). The caller's choice, never implicit.
    /// Why: P3.
    void wait();

    // ---- host access (between runs) ---------------------------------------------
    /// Views: wrapper ergonomics on the host — shape(), operator[] with
    /// id<Rank> and flat, size(). On a device-kind root a view is
    /// root.host_view(); on a shared- or host-kind root it is
    /// queue_.wait() then root.get_access(). Either way the cost is
    /// honest and paid only where asked; the read is ordered after the
    /// stream's prior work by SlabKit (§9.4). Read-only variants cost no
    /// writeback.
    template <DType DT, int Rank> HostView<DT, Rank>    view_input (std::string_view path) const;
    template <DType DT, int Rank> HostView<DT, Rank>    view_output(std::string_view path) const;   // read-only
    /// Writable output view: writes land in host staging; written back
    /// (root.from_host, then `copy_d = root` per other device) on
    /// destruction AND on commit() (both; the destructor is the
    /// protection, commit() the mid-scope flush).
    /// Why: §1 need 2.3 host access · P3 (a read-only view costs no
    /// submission).
    template <DType DT, int Rank> HostViewMut<DT, Rank> view_output_mut(std::string_view path);
    /// Setters: root.from_host(span) into the root's primary storage,
    /// then `copy_d = root` per other device, like any root write.
    /// Inputs: meaningful for free inputs (a connected one is overwritten
    /// at the next forward — harmless, like init).
    template <DType DT, int Rank> void set_input (std::string_view path, std::span<const cpp_type_t<DT>>);
    template <DType DT, int Rank> void set_output(std::string_view path, std::span<const cpp_type_t<DT>>);

    // ---- retrieval & observation ------------------------------------------------
    template <DType DT, int Rank> Input<DT, Rank>&  get_input (std::string_view path);
    template <DType DT, int Rank> Output<DT, Rank>& get_output(std::string_view path);
    System& root();
    Subsystem& get_subsystem(std::string_view path);           // root.get_subsystem

    struct MemoryReport {
        struct PerDevice { size_t arena_bytes, pool_bytes, pool_bytes_in_use,
                                  os_allocations, os_bytes_in_use, transient_peak_bytes; };
        std::unordered_map<sycl::device, PerDevice> devices;
    };
    MemoryReport memory_report() const;

    // ---- marks (§8) -------------------------------------------------------------
    std::span<const MarkRecord> design_vars() const;
    std::span<const MarkRecord> constraints() const;
    std::span<const MarkRecord> objectives()  const;

    /// Waits the stream as a convenience (§9.4). Everything else is RAII.
    ~Model();

private:
    sk::context ctx_;
    sk::queue   queue_;
    Substrates  substrates_;
    std::unique_ptr<System> root_;
    ModelConfig cfg_;
};

} // namespace naga::cospan
```

### 9.2 Host access

- `view_*` on a device-kind root = `root.host_view()`; on a shared- or
  host-kind root = `queue_.wait()` then `root.get_access()`.
- `set_*` = `root.from_host(span)` then `copy_d = root` per other device.
- `view_output_mut` stages on the host and does `from_host` on commit and
  on destruction.
- `forward()` enqueues; `wait()` = `queue_.wait_and_throw()`.

### 9.3 Sample — the whole user-side lifecycle

```cpp
auto model = create_model<Top>(ModelConfig{});          // default device, no pools
model.setup();                                          // declare → exec-info → substrates
                                                        //   → prepare → compile → init
model.set_input<DType::f64, 1>("mda.z", {{1.0, 2.0}});  // drive a free input
model.forward();  model.wait();                         // enqueue; wait when results are wanted

auto y1 = model.view_output<DType::f64, 1>("mda.y1");   // host_view, read
auto r  = model.memory_report();                        // r.devices[gpu0].os_bytes_in_use > 0 ?
                                                        //   -> rerun with pool_bytes set to it

// Two devices in one model:
ModelConfig cfg{std::array{gpu0, gpu1}};                // builds the sk::context
cfg.default_device = gpu0;                              // validated with ctx.contains()
auto m2 = create_model<Top>(std::move(cfg));
```

### 9.4 The layering rule

Each layer orders what it submits: SlabKit its tasks, CoSpan the kernels it
launches, the user anything they launch over a root outside the framework.
Teardown relies on AdaptiveCpp queue destruction; waiting before exit is
the user's obligation. `~Model` waits the stream as a convenience.
Members are destroyed in reverse declaration order: `cfg_`, `root_` (every
root's `usm_vector` releases), `substrates_` (arenas and pools), `queue_`,
`ctx_` — no destruction order is relied on beyond that (P9).

---

## 10. System types: the extension contract, by example

Binding contract; declarations only. The base System, Parallel and
Fused: §7. Systems never do work — a System type alters ORCHESTRATION
(what runs, when, on which device), never compute. Work is a Component's,
including custom operations (§5.1).

### 10.1 The principle

Wondering where some unique behaviour goes? It is a custom System type.
The runners are virtual precisely so that a System subclass can override
the orchestration while every Component under it stays untouched and
every owner above it stays ignorant. **No framework primitive exists for
gating, counting, or change detection**: those are ordinary System
subclasses — provided in the library where common, written by users
where not — and their entire mechanism is an early return from
`run_compute_forward`. RunOnceSystem and RunNSystem, the setup-step
patterns, live inside the framework. Why: §1 need 2.2 possibility floor
(do once / run N / run when changed / run if) · P5.

```cpp
namespace naga::cospan {

/// Run the subtree exactly once, ever. Construction-time operators
/// (LBVH, LRPIM, nonlocal-operator assembly at static placement).
class RunOnceSystem : public System {
    bool done_ = false;
protected:
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        if (done_) return nullptr;            // nothing submitted
        done_ = true;
        return System::run_compute_forward(); // the default loop
    }
};

/// Run the subtree the first N forwards, then never.
class RunNSystem : public System {
    size_t remaining_;
public:
    explicit RunNSystem(size_t n) : remaining_{n} {}
protected:
    std::unique_ptr<LaunchDescription> run_compute_forward() override {
        if (remaining_ == 0) return nullptr;
        --remaining_;
        return System::run_compute_forward();
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

These are added like any realized type: `add_subsystem<RunOnceSystem>("operators")`,
`add_subsystem<ConditionalSystem>("regime", "regime_flag")`. They compose by
nesting.

### 10.2 What the contract guarantees a custom System type

- **Runners are virtual, hooks are for users.** A System type overrides
  `run_compute_forward` (and, when it must, `prepare_exec`, `compile`,
  `run_init`); its users override `setup`/`init`/`pre_forward` as with any
  System.
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
guarantee) and a System exposes its children. Recorders, listings and
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
   `queue_ = sk::queue{ctx_, default_device, handler}` — one in-order
   stream. Every submission names the device it targets.
2. Model constructs `Top(top_args...)`. A constructor is plain C++ state.
   Structural calls and declarations outside `setup()` throw.

**State:** one context, one idle queue, one root object, nothing else.

### 11.2 `model.setup()` — the setup chain

`run_setup → get_exec_info → substrates → prepare_exec → compile → init`.
Every step is a Subsystem verb driven from the root; Systems recurse in
declared order (`children_` IS the exec order); nobody asks what kind a
child is. The order is the one of §9.1: (0 construct) 1 run_setup ·
2 get_exec_info · 3 substrates · 4 prepare_exec · 5 compile · 6 init_tree.

#### 11.2.1 Declare — `root.run_setup()`

`System::run_setup` → `setup()`. Inside it: `add_subsystem` constructs the
child, registers the name, sets its owner, **runs the child's `run_setup`
before returning** (why: P6 — no deferred resolution pass);
`connect` / `promote` / `tie_*` resolve paths in this
namespace and apply the set operation (§4) — computed connections (§8)
create or extend the input's mapping System; marks record on roots. Then
this System's post-setup pass: each mapping System — framework
`tie_outputs(allow_plain)`, alias → input, coverage check, insert before
the consumer's child.

`Component::run_setup` → `setup()` → `declare_*` creates entries (roots of
singleton sets; flags by kind; DimRecords; lifetime; coverage). Zero
allocation; no device anywhere in this step.

**State:** final tree and exec order; final sets; marks. No storage.

#### 11.2.2 Exec info — `bytes = root.get_exec_info(default_device)`

`System`: `ctx.device` = own `property::device` else `parent_device`; seed
`{device: 0}`; process **own alias entries first** (a tie alias root
counts its static bytes here); then each child in order, merging returned
maps (sum on key).

`Component`: device = `parent_device`; per own entry `e`:
- root → register this device as `e`'s primary; count its static bytes
  here;
- input peer, `root(e)` already has storage registered on this device →
  nothing;
- input peer, it does not → register this device on `root(e)` and count
  the root's static bytes here (the root will own a copy on this device;
  §4.2);
- output peer (only a tied output is one) whose root's device ≠ this
  device → ERROR naming the tie: producers cannot write one storage
  from two devices.

Every root is processed before any peer of it: alias roots before
children, producers before consumers (acyclic). A System processes its
own alias entries the same way (a tie alias root registers the System's
device as primary; a promotion alias is a peer and registers the
System's device on its root — so `get_output("alias")` from that System
yields memory on that System's device). Returns `{device: bytes}`.
Why: §1 need 2.2 placement · P1 · P5.

#### 11.2.3 Substrates

Model: for each `(device, bytes)`: `substrates[device] =
{ DeviceArena{queue_, device, kind::device, bytes}, DevicePool iff
configured for that device }` — built from `(queue, device)`, with no
context of its own; the one context is the Model's. Every
allocation for a device goes through its substrate.

#### 11.2.4 Prepare — `root.prepare_exec(queue, substrates)`

`System`: complete `ctx` — `queue` = the Model's, `group` = null (only a
Parallel owner sets one, for the extent of its `submit_group`),
`substrate = &substrates[device]`; allocate own alias roots and point their
wrappers; recurse.

`Component`: for every own ROOT entry allocatable now, build its
`usm_vector` per registered device from that device's substrate —
durable∧static from the arena route; durable∧dynamic with a known shape via
the pool route (OS fallback); dynamic-shaped and transient stay
unallocated. Then point every member wrapper at
`root()->storage_on(own device)` — guaranteed
allocated by now (roots precede peers); shape and stride (+ view) set;
static input peers validate against their roots here. An unconnected
input carrying a `FromSourceDim` dimension has no source to derive from:
ERROR here ("source-derived dimension on an unconnected input").

#### 11.2.5 Compile — `root.compile()`

`Component`: device function(s) + id at the owner's fuse rank if any, else
native; individual config; custom-op bodies recorded. `System`: children
in order. Fused and Parallel legality is NOT checked here — it is checked
each forward on the descriptions the children return (cheap, host-side,
and it is where the information is).

#### 11.2.6 Init — `root.init_tree()`

`System`: children in order, then `init(queue, device)`. `Component`:
`init(queue, device)`, with the Model's queue and the subsystem's device
(signature: `init(sk::queue&, const sycl::device&)`). Post-order means an
owner's `init` has the last word by path. Initializers write a root's
primary storage and copy it to the root's other-device storage. Unnecessary
inits (connected inputs) are harmless. An initializer on unallocated
storage throws.

**State:** storage for every static root, initialized; wrappers current;
kernels compiled. The model can run. Its state is not yet meaningful —
it holds initialization; state is defined after the first forward.

### 11.3 `model.forward()`

`Model::forward()` → `root.forward()` → `root.run_compute_forward()`.
Enqueue only. `model.wait()` (`queue_.wait_and_throw()`) waits the stream
when the caller wants the results on the host; a caller looping N forwards
need not wait between them — the stream orders (§7.3).

#### 11.3.1 A System's turn — the ordering contract (§7.3)

```
for child in children:
    child.run_pre_compute(ctx)
    d = child.run_compute_forward()
    if d:  d->submit(ctx.target())            // this System submits;
                                              //   routed on d->is_custom_op()
    child.run_post_compute(ctx)               // a child that submitted itself
                                              //   needs no owner action
return null
```

The stream is an abstract in-order sequence of items: a kernel is an item;
a group is one item (unordered inside, ordered both ways outside); async
host and memory tasks are concurrent with each other and ordered after
every kernel/group item before them and before every one after. Placement
never costs the order; nothing in CoSpan threads dependencies. Kernel
descriptions take the outer-kernel path, custom operations take
`AdaptiveCpp_enqueue_custom_operation`, through the same `SubmitTarget`.
**Parallel** (Components only): pre-compute ALL → `queue.submit_group {
each child's description submitted with its device }` → post-compute ALL
after the group returns. **Fused**: pre-compute ALL → compute ALL (collect;
a null or custom-op description is an error) → validate the wave on the
descriptions (rank agreement, scratch, producer-consumer in parallel mode)
→ pack, stamp group slices, one staging, ONE submission with the sequence
config → post-compute ALL. **Custom System types** (§10): early-return from
`run_compute_forward`.

#### 11.3.2 A Component's turn (§6.5)

1. `run_pre_compute(ctx)`: for each own input driving a shape via
   `FromInput` — read the driving input root's `host_view()` (the one
   forced sync, so the launch geometry below is known), `root.resize(shape)`,
   `point_wrapper`; nothing submitted for the resize, no reallocation
   inside a submitted task. Then `pre_forward()` → dynamic validation → wrapper sync.
2. `run_compute_forward()` → `compute_forward()` → description (kernel or
   custom op). The owner submits it.
3. `run_post_compute(ctx)`: for each output root with storage on other
   devices, for each other device `copy_d = root` — one assignment
   resizes to match and copies, routed by SlabKit — after the owner's
   submission, ordered by the stream.

**State:** every output root holds this forward's values; dynamic roots
sized to this forward.

### 11.4 Host access

`view_*` (device-kind root: `root.host_view()`; shared/host-kind root:
`queue_.wait()` then `root.get_access()`; the writable output view stages
on the host and does `from_host` on destruction and on `commit()`),
`set_*` (`root.from_host(span)`, then `copy_d = root` per other device),
`memory_report()`, `get_subsystem(path)` chaining and
`get_inputs()/get_outputs()` enumeration (§10.3). Host access is §9.2.

### 11.5 End of life

The layering rule (§9.4): each layer orders what it submits; teardown
relies on AdaptiveCpp queue destruction, and waiting before exit is the
user's obligation. `~Model` waits the stream as a convenience. Everything
else is RAII, in reverse declaration order, identical to §9.4: `cfg_`,
`root_` (roots' `usm_vector`s release), `substrates_` (arenas, pools),
`queue_`, `ctx_` (P9).

