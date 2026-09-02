# Spec 01.A — dtypes and I/O wrappers

Part of the phase-1 spec (`docs/spec/01-components-io/`). Binding contract;
declarations only. Sibling documents cover the metadata layer (01.B), the
kernel machinery (01.C), the Component base and helpers (01.D), and the
System-side execution slice (01.E).

```cpp
namespace naga::cospan {

/// The closed set of I/O element types. Extension is a spec revision.
/// Atomic variants are distinct enum entries: declaring an I/O with an
/// atomic dtype changes the wrapper's accessor discipline (atomic
/// references) and nothing else about its storage. An atomic variant
/// exists exactly where sycl::atomic_ref is legal (4/8-byte element
/// types); u8 has no atomic form.
enum class DType : uint8_t {
    f32, f64, i32, u32, i64, u64, u8,
    atomic_f32, atomic_f64, atomic_i32, atomic_u32, atomic_i64, atomic_u64
};

/// Traits:
///   value_dtype_of<DT>      — underlying value dtype (atomic_f32 -> f32).
///     Connection/tie/promotion compatibility SHALL be judged on
///     value_dtype_of, never DT: the atomic modifier is an access
///     discipline over shared storage, not a storage type.
///   cpp_type_t<DT>          — C++ element type (through the modifier).
///   is_atomic_v<DT>         — true for atomic entries.
///   is_differentiable_v<DT> — true for f32/f64, plain or atomic.
///     Data-tier dtypes (integers, u8): gradient-related declarations
///     SHALL NOT EXIST at compile time (requires-gated) — misuse is a
///     compile error, never a runtime one.

// -----------------------------------------------------------------------------
// Wrappers — the only device-side data handles in the framework.
// -----------------------------------------------------------------------------
// Common contract:
//   * Trivially copyable; captured by value into kernels
//     (static_assert-enforced per instantiation).
//   * Wrappers carry NO identity — they are pure device payload
//     {shape, stride, data}. Component APIs that take a member reference
//     (e.g. reshape) resolve it to its IOEntry through the member-address
//     map (spec 01.B); nothing host-only rides into kernel captures.
//   * Raw pointers are private. The metadata layer (spec 01.B,
//     detail::IOEntry) is the sole writer of wrapper fields, via
//     friendship. User code never sees a pointer.
//   * `value_type` is the ONLY element-type spelling kernel bodies may
//     use — the scalar-genericity keystone: phase 7 (AD) re-instantiates
//     the same bodies at derivative-carrying value types with no
//     user-code change.
//   * Rank is 1..3. Indexing accepts sycl::id<Rank> and flat size_t;
//     flat indexing is row-major over shape().
//   * stride() is the element step per dimension in the OWNING
//     allocation: subarray-aliased views (spec 03) are a metadata act,
//     never a wrapper variant.
//   * Wrappers carry primal data only. Gradient access arrives in
//     phase 7 as separate wrapper forms instantiated at the same
//     declaration sites; the metadata layer reserves the slots.

/// Input<DT, Rank> — read-only view. THE INPUT CONTRACT: a kernel SHALL
/// NOT be able to write through an Input — all access is const-reference.
/// Load-bearing twice: ImplicitComponent's no-aliasing design (05) and
/// phase-7 AD purity both assume it. Atomic dtypes are ill-formed on
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
///     model error the tie check (02.C) and the fused-parallel
///     producer-consumer check (01.E) catch.
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

static_assert(std::is_trivially_copyable_v<Input<DType::f32, 2>>);
static_assert(std::is_trivially_copyable_v<Output<DType::atomic_f32, 1>>);

// Spelling samples:
//   Input<DType::f32, 1> x;
//   Output<DType::f64, 2> y;
//   Output<DType::atomic_f32, 1> accum;

} // namespace naga::cospan
```

## Notes binding this document

- Dtype compatibility across connections, ties and promotions is judged on
  `value_dtype_of`; the atomic modifier never blocks aliasing — it changes
  which accessor discipline each side compiles against.
- `tie_outputs` requires the atomic modifier on member outputs by default
  (overridable) — 02.C owns the rule; the dtype system here is what
  makes it checkable.
- DVs, constraints and objectives are metadata marks (01.B / 03);
  nothing in the wrapper layer knows about them.
