#pragma once

// =============================================================================
// naga/cospan/instance.hpp — one declared input, output or alias
// =============================================================================
//
// Every input, output and alias is an INSTANCE, owned by exactly one component or System through a
// shared_ptr (docs/design-review-2026-10-03.md, section 1). Nothing outside the owner holds an instance
// strongly, and there is no global record of anything: no roots, no shadows, no ledgers. An instance is
// generic and passive. It holds:
//
//   - the variable's shared metadata (VarMeta, held by every instance of the variable);
//   - its own device (section 2: devices are never on the metadata) and storage kind;
//   - a shared pointer to the data of ITS device, given to it by materialize;
//   - its connect flags (can it be a source, can it be a target);
//   - for a port, the wrapper its component's kernel reads (the typed leaves below).
//
// When an instance goes away it simply stops holding its data and the metadata: the metadata's weak
// entry for it expires. There is nothing to unregister.
//
// The rules about which variables merge (connect, tie, promote) and who wins live in those calls, a later
// phase; rebind() is the one operation they need from here.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "naga/cospan/dispatch.hpp"
#include "naga/cospan/dtype.hpp"
#include "naga/cospan/metadata.hpp"
#include "naga/cospan/runtime.hpp"
#include "naga/cospan/subsystem.hpp"
#include "naga/cospan/variable.hpp"
#include "naga/cospan/wrappers.hpp"

namespace naga::cospan {
namespace detail {

/// What an instance is: a component's port, a System's promotion, a System's tie alias, or a host view.
enum class Role : uint8_t { port, promotion, tie, host_view };

/// Row-major strides of a 1-padded extents triple. For a rank-R instance the first R strides are exactly
/// rank-R row-major strides.
constexpr std::array<std::size_t, 3> contiguous_strides(const std::array<std::size_t, 3>& e) {
    return {e[1] * e[2], e[2], 1};
}

/// What a wrapper is pointed at: shape and stride, 1-padded.
struct Geometry {
    std::array<std::size_t, 3> shape{1, 1, 1};
    std::array<std::size_t, 3> stride{1, 1, 1};
};

/// The only writer of wrapper fields (friendship; wrappers.hpp): user code never sees a pointer.
struct WrapperAccess {
    template <class W, class P>
    static void write_wrapper(W& w, const Geometry& g, P* data) {
        w.shape_  = make_range<W::rank>(g.shape);
        w.stride_ = make_range<W::rank>(g.stride);
        w.data_   = data;
    }

    template <class W, class P>
    static void write_ragged(W& w, std::size_t n, P* data, const std::uint64_t* offsets,
                             const std::array<std::size_t, 2>& ext, const std::array<const std::uint64_t*, 2>& per) {
        w.n_       = n;
        w.data_    = data;
        w.offsets_ = offsets;
        w.ext_     = ext;
        w.per_     = per;
    }
};

class Instance : public std::enable_shared_from_this<Instance> {
public:
    Instance(std::string n, const Subsystem* decl, IOKind k, Role r, std::vector<DimRecord> d, bool can_source,
             bool can_target)
        : name{std::move(n)}
        , declarer{decl}
        , kind{k}
        , role{r}
        , dims{std::move(d)}
        , can_be_source{can_source}
        , can_be_target{can_target} {}

    Instance(const Instance&)            = delete;
    Instance& operator=(const Instance&) = delete;
    virtual ~Instance()                  = default;

    std::string            name;
    const Subsystem*       declarer;  ///< the component or System that owns this instance
    IOKind                 kind;
    Role                   role;
    std::vector<DimRecord> dims;  ///< as declared HERE; the variable's own definition is on the metadata

    sycl::device device       = default_device();  ///< section 2: set_device overrides it
    sk::kind     storage_kind = sk::kind::device;

    bool can_be_source;
    bool can_be_target;

    std::shared_ptr<VarMeta>         meta;  ///< shared by every instance of the variable
    std::shared_ptr<DeviceDataBase>  data;  ///< the data of THIS instance's device, once materialized

    /// The instance's dotted path from the root System: its declarer's path plus its name, or the name alone
    /// at the root.
    std::string full_path() const {
        return (declarer == nullptr || declarer->path().empty()) ? name : declarer->path() + "." + name;
    }

    /// Where this instance's data lives.
    DataKey key() const { return DataKey{device, storage_kind}; }

    void set_data(std::shared_ptr<DeviceDataBase> sp) { data = std::move(sp); }

    /// Joins another variable: drops this instance's data, takes the metadata, and enters its instance list. The
    /// merge walks of connect, tie and promote re-point the instances of the losing metadata with this.
    void rebind(std::shared_ptr<VarMeta> m) {
        data.reset();
        meta = std::move(m);
        if (meta) meta->add_instance(weak_from_this());
    }

    /// Writes the component's member wrapper from the current shape and data. Aliases have none.
    virtual void point_wrapper() {}

    /// A dense Input port (not RaggedInput): it cannot see per-index extents (wiring.hpp, connect).
    virtual bool is_dense_input_port() const { return false; }
};

// What variable.hpp declared for VarMeta's sake.
inline DataKey             key_of(const Instance& i) { return i.key(); }
inline void                give_data(Instance& i, std::shared_ptr<DeviceDataBase> d) { i.set_data(std::move(d)); }
inline bool                is_output_port(const Instance& i) { return i.kind == IOKind::output && i.role == Role::port; }
inline std::string         path_of(const Instance& i) { return i.full_path(); }
inline const sycl::device& device_of(const Instance& i) { return i.device; }

// -----------------------------------------------------------------------------
// Typed leaves: the instances that hold a member wrapper
// -----------------------------------------------------------------------------

template <DType DT, int Rank>
class TypedInstance : public Instance {
public:
    using value_type = cpp_type_t<DT>;
    using Instance::Instance;

protected:
    /// A dense wrapper: the variable's shape with contiguous strides over this instance's data (null while it
    /// has none).
    template <class W>
    void point(W& w) {
        const ResolvedShape& s = meta->shape;
        if (s.ragged())
            throw std::logic_error("naga::cospan: '" + full_path() + "' is ragged; declare it with RaggedInput / "
                                   "RaggedOutput (a dense Input / Output cannot see per-index extents)");
        Geometry g;
        if (s.known) {
            g.shape  = s.extents;
            g.stride = contiguous_strides(s.extents);
        } else {
            g.shape  = {0, 0, 0};
            g.stride = {0, 0, 0};
        }
        auto* base = data ? static_cast<value_type*>(data->raw()) : nullptr;
        WrapperAccess::write_wrapper(w, g, base);
    }

    /// A ragged wrapper: the data, the block count, and this instance's device copies of the index arrays. A DENSE
    /// variable is accepted too (offsets null: block i is row i of the rectangle).
    template <class W>
    void point_ragged(W& w) {
        const ResolvedShape&                s    = meta->shape;
        auto*                               base = data ? static_cast<value_type*>(data->raw()) : nullptr;
        std::array<std::size_t, 2>          ext{1, 1};
        std::array<const std::uint64_t*, 2> per{nullptr, nullptr};
        const std::uint64_t*                offs = nullptr;
        if (s.ragged()) {
            const auto* rd = dynamic_cast<const RaggedDeviceData<value_type>*>(data.get());
            for (int d = 1; d < Rank; ++d) {
                const auto u = static_cast<std::size_t>(d);
                if (!s.per[u])
                    ext[u - 1] = s.extents[u];
                else if (rd != nullptr && rd->per[u - 1])
                    per[u - 1] = rd->per[u - 1]->data();
            }
            if (rd != nullptr && rd->offs) offs = rd->offs->data();
        } else {
            for (int d = 1; d < Rank; ++d) ext[static_cast<std::size_t>(d - 1)] = s.extents[static_cast<std::size_t>(d)];
        }
        WrapperAccess::write_ragged(w, s.extents[0], base, offs, ext, per);
    }
};

/// An input: can be a target, cannot be a source.
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
class InputInstance final : public TypedInstance<DT, Rank> {
public:
    InputInstance(std::string name, const Subsystem* declarer, const std::array<DimRecord, Rank>& dims,
                  Input<DT, Rank>* member)
        : TypedInstance<DT, Rank>{std::move(name), declarer, IOKind::input, Role::port,
                                  std::vector<DimRecord>(dims.begin(), dims.end()), false, true}
        , member_{member} {}

    void point_wrapper() override {
        if (member_ != nullptr) this->point(*member_);
    }
    Input<DT, Rank>& member() const { return *member_; }
    bool is_dense_input_port() const override { return true; }

private:
    Input<DT, Rank>* member_;
};

/// An output: can be a source, cannot be a target.
template <DType DT, int Rank>
class OutputInstance final : public TypedInstance<DT, Rank> {
public:
    OutputInstance(std::string name, const Subsystem* declarer, const std::array<DimRecord, Rank>& dims,
                   Output<DT, Rank>* member)
        : TypedInstance<DT, Rank>{std::move(name), declarer, IOKind::output, Role::port,
                                  std::vector<DimRecord>(dims.begin(), dims.end()), true, false}
        , member_{member} {}

    void point_wrapper() override {
        if (member_ != nullptr) this->point(*member_);
    }
    Output<DT, Rank>& member() const { return *member_; }

private:
    Output<DT, Rank>* member_;
};

template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
class RaggedInputInstance final : public TypedInstance<DT, Rank> {
public:
    RaggedInputInstance(std::string name, const Subsystem* declarer, const std::array<DimRecord, Rank>& dims,
                        RaggedInput<DT, Rank>* member)
        : TypedInstance<DT, Rank>{std::move(name), declarer, IOKind::input, Role::port,
                                  std::vector<DimRecord>(dims.begin(), dims.end()), false, true}
        , member_{member} {}

    void point_wrapper() override {
        if (member_ != nullptr) this->point_ragged(*member_);
    }

private:
    RaggedInput<DT, Rank>* member_;
};

template <DType DT, int Rank>
class RaggedOutputInstance final : public TypedInstance<DT, Rank> {
public:
    RaggedOutputInstance(std::string name, const Subsystem* declarer, const std::array<DimRecord, Rank>& dims,
                         RaggedOutput<DT, Rank>* member)
        : TypedInstance<DT, Rank>{std::move(name), declarer, IOKind::output, Role::port,
                                  std::vector<DimRecord>(dims.begin(), dims.end()), true, false}
        , member_{member} {}

    void point_wrapper() override {
        if (member_ != nullptr) this->point_ragged(*member_);
    }

private:
    RaggedOutput<DT, Rank>* member_;
};

/// A System-level alias (a promotion or a tie alias): no member wrapper. Its flags are given.
template <DType DT, int Rank>
class AliasInstance final : public TypedInstance<DT, Rank> {
public:
    AliasInstance(std::string name, const Subsystem* declarer, IOKind k, Role role, std::vector<DimRecord> dims,
                  bool can_source, bool can_target)
        : TypedInstance<DT, Rank>{std::move(name), declarer, k, role, std::move(dims), can_source, can_target} {}
};

// -----------------------------------------------------------------------------
// Factories: each makes the instance and, unless told to join a variable, a variable of its own
// -----------------------------------------------------------------------------

inline bool any_per_index(const std::vector<DimRecord>& dims) {
    for (const DimRecord& r : dims)
        if (r.per) return true;
    return false;
}

/// Gives a new instance its own shared metadata (ragged when the port is, or any dimension is per-index) and enters
/// it in that metadata's instance list.
inline void adopt_new_meta(const std::shared_ptr<Instance>& inst, DType dt, int rank, bool ragged) {
    inst->meta = make_var_meta(dt, rank, inst->dims, ragged || any_per_index(inst->dims));
    inst->meta->add_instance(inst);
}

/// The caller owns the returned instance (a component keeps its ports; a System its aliases).
template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
std::shared_ptr<Instance> make_input(std::string name, const Subsystem* declarer, const std::array<DimRecord, Rank>& dims,
                                     Input<DT, Rank>* member) {
    std::shared_ptr<Instance> inst = std::make_shared<InputInstance<DT, Rank>>(std::move(name), declarer, dims, member);
    adopt_new_meta(inst, DT, Rank, false);
    return inst;
}

template <DType DT, int Rank>
std::shared_ptr<Instance> make_output(std::string name, const Subsystem* declarer, const std::array<DimRecord, Rank>& dims,
                                      Output<DT, Rank>* member) {
    std::shared_ptr<Instance> inst = std::make_shared<OutputInstance<DT, Rank>>(std::move(name), declarer, dims, member);
    adopt_new_meta(inst, DT, Rank, false);
    return inst;
}

template <DType DT, int Rank>
    requires(!is_atomic_v<DT>)
std::shared_ptr<Instance> make_ragged_input(std::string name, const Subsystem* declarer,
                                            const std::array<DimRecord, Rank>& dims, RaggedInput<DT, Rank>* member) {
    std::shared_ptr<Instance> inst = std::make_shared<RaggedInputInstance<DT, Rank>>(std::move(name), declarer, dims, member);
    adopt_new_meta(inst, DT, Rank, true);
    return inst;
}

template <DType DT, int Rank>
std::shared_ptr<Instance> make_ragged_output(std::string name, const Subsystem* declarer,
                                             const std::array<DimRecord, Rank>& dims, RaggedOutput<DT, Rank>* member) {
    std::shared_ptr<Instance> inst = std::make_shared<RaggedOutputInstance<DT, Rank>>(std::move(name), declarer, dims, member);
    adopt_new_meta(inst, DT, Rank, true);
    return inst;
}

/// An alias instance of run-time dtype and rank, joining `meta` (a fresh variable of its own when null).
inline std::shared_ptr<Instance> make_alias_instance(DType dt, int rank, std::string name, const Subsystem* declarer,
                                                     IOKind kind, Role role, std::vector<DimRecord> dims, bool can_source,
                                                     bool can_target, std::shared_ptr<VarMeta> meta) {
    std::shared_ptr<Instance> inst;
    dispatch_dtype_rank(dt, rank, [&](auto dtc, auto rc) {
        constexpr DType DT = decltype(dtc)::value;
        constexpr int   R  = decltype(rc)::value;
        inst = std::make_shared<AliasInstance<DT, R>>(std::move(name), declarer, kind, role, std::move(dims), can_source,
                                                      can_target);
    });
    if (meta) {
        inst->meta = std::move(meta);
        inst->meta->add_instance(inst);
    } else {
        adopt_new_meta(inst, dt, rank, false);
    }
    return inst;
}

/// The writable twin handed to cs::init::custom: the same geometry as a member wrapper, over the variable's
/// data on one device, with the variable's shape.
template <DType DT, int Rank>
Initializer<DT, Rank> make_initializer(VarMeta& var, DeviceDataBase& data) {
    Initializer<DT, Rank> init;
    const ResolvedShape&  s = var.shape;
    Geometry              g;
    if (s.known) {
        g.shape  = s.extents;
        g.stride = contiguous_strides(s.extents);
    } else {
        g.shape  = {0, 0, 0};
        g.stride = {0, 0, 0};
    }
    WrapperAccess::write_wrapper(init, g, static_cast<cpp_type_t<DT>*>(data.raw()));
    return init;
}

}  // namespace detail
}  // namespace naga::cospan
