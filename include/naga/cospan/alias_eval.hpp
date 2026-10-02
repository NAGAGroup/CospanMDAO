#pragma once
// =============================================================================
// naga/cospan/alias_eval.hpp: detail::AliasEvaluator
// =============================================================================
//
// What evaluates a tie alias's shape rule and reads the values and shapes a rule depends on.
// A Component and a System each OWN one (a tie alias root is
// owned and resolved by its declaring System). It holds the queue it reads through, the recorders and
// the read log a ChunkedSystem's full pass fills (its topology cache and dependency self-check), and the
// chunk-pass state.
//
// `ensure_storage`: a Component installs one that allocates a root it OWNS before a shape-driving read; a
// System has none, so the source root must already have storage (else the 'has no storage yet' error).

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "naga/cospan/dtype.hpp"
#include "naga/cospan/metadata.hpp"

namespace naga::cospan::detail {

constexpr bool is_integer_dtype(DType dt) noexcept {
    const DType v = value_dtype(dt);
    return v != DType::f32 && v != DType::f64;
}

class AliasEvaluator {
public:
    sk::queue*                       queue_ = nullptr;      ///< set at prepare_exec by the owner
    std::function<void(IOEntry&)>    ensure_;               ///< a Component: allocate_owned; a System: none
    bool                             chunk_pass_ = false;   ///< a ChunkedSystem's full-range pass is running

    /// One recorded read: the root read, the entry being resolved when it was read, and which kind of dependency.
    struct DepRead {
        const IOEntry* root;
        const IOEntry* by;
        bool           shape;
    };
    /// Set by the ChunkedSystem around a full pass: every value read (read_value / read_values, which every alias
    /// and port value read goes through) pushes the root it read; copy_alias_dim pushes the roots whose SHAPE it
    /// copied. Single-threaded, like all submission.
    mutable std::vector<DepRead> read_log_;  ///< the same reads with their readers (the ChunkedSystem's dependency self-check)
    void note_value_dep(IOEntry* root) const {
        if (dep_recorder_ == nullptr) return;
        dep_recorder_->push_back(root);
        read_log_.push_back(DepRead{root, resolving_, false});
    }
    void note_shape_dep(const IOEntry* root) const {
        if (shape_recorder_ == nullptr) return;
        shape_recorder_->push_back(root);
        read_log_.push_back(DepRead{root, resolving_, true});
    }
    std::vector<IOEntry*>*       dep_recorder_   = nullptr;
    std::vector<const IOEntry*>* shape_recorder_ = nullptr;
    const IOEntry*               resolving_      = nullptr;  ///< the entry being resolved

    sk::queue& queue_ref() const {
        if (queue_ == nullptr) throw std::logic_error("naga::cospan: an alias evaluation needs the queue (prepare_exec has not run)");
        return *queue_;
    }
    void ensure_storage(IOEntry& root) const {
        if (ensure_) ensure_(root);
    }

    /// A tie alias's shape as its AliasShape gives it right now (scalar extents, and per-index arrays
    /// for dims >= 1 that are per-index; the extent of a per-index dim is 0).
    struct AliasEval {
        std::array<std::size_t, 3>              ext{1, 1, 1};
        std::array<bool, 3>                     per{false, false, false};
        std::array<std::vector<std::size_t>, 3> per_index;
    };

    /// Inside a ChunkedSystem an internal alias holds ONE chunk [start, start + len) of the leading dimension: a
    /// per-index array that covers the whole run (a boundary root's) is cut to it; one that is already
    /// chunk-sized (an inside entry's, resolved under the chunk) is kept as it is.
    std::vector<std::size_t> chunk_slice(const detail::IOEntry& a, std::vector<std::size_t> v) const {
        if (!a.dim0_override_ || v.size() == *a.dim0_override_) return v;
        const std::size_t first = 0, len = *a.dim0_override_;
        if (v.size() < first + len)
            throw std::invalid_argument("naga::cospan: tie alias '" + a.full_path() + "': a per-index source has " +
                                        std::to_string(v.size()) + " entries but the chunk reaches index " +
                                        std::to_string(first + len));
        return std::vector<std::size_t>(v.begin() + static_cast<std::ptrdiff_t>(first),
                                        v.begin() + static_cast<std::ptrdiff_t>(first + len));
    }

    /// Slot `to` of `out` takes dimension `d` of `up` (per-index too, if that dim is). SHAPE inheritance only:
    /// it never reads a value, so an internal entry of a ChunkedSystem is a fine source. `input`: `up` is an
    /// INPUT source, whose ROOT is read (the input's own shape may not be resolved yet); a sliced input
    /// contributes its view's extents.
    void copy_alias_dim(AliasEval& out, std::size_t to, const detail::IOEntry& src, int d, const detail::IOEntry& a,
                        bool input) const {
        if (input && src.view()) {
            note_shape_dep(src.root());
            out.ext[to] = src.view()->extents[static_cast<std::size_t>(d)];
            return;
        }
        const detail::IOEntry& up = input ? *src.root() : src;
        note_shape_dep(up.root());  // a SHAPE dependency of the topology cache
        if (up.dim_is_per(d)) {
            if (to == 0)
                throw std::invalid_argument("naga::cospan: a per-index dimension of '" + up.full_path() +
                                            "' cannot become dimension 0 of a tie alias");
            out.per[to]       = true;
            out.per_index[to] = chunk_slice(a, up.per_index_extents(d));
            out.ext[to]       = 0;
        } else {
            out.ext[to] = up.extents()[static_cast<std::size_t>(d)];
        }
    }

    /// What the alias's AliasShape gives right now. Every upstream output it reads is a producer that has
    /// already run (checked at setup: it is earlier in registration order); the value reads happen on the
    /// host from a snapshot, as FromInputValues does.
    AliasEval alias_eval(const detail::IOEntry& a) {
        const AliasShape& s = *a.alias_shape_;
        AliasEval         out;
        resolving_ = &a;  // names the alias in a "computed inside a chunked system" error
        switch (s.kind) {
            case AliasShape::Kind::fixed: out.ext = s.extents; break;
            case AliasShape::Kind::from_output_shape:
                for (int d = 0; d < a.rank(); ++d) copy_alias_dim(out, static_cast<std::size_t>(d), *a.alias_src_, d, a, false);
                break;
            case AliasShape::Kind::from_input_shape:
                for (int d = 0; d < a.rank(); ++d) copy_alias_dim(out, static_cast<std::size_t>(d), *a.alias_src_, d, a, true);
                break;
            case AliasShape::Kind::from_output_values:
                for (int d = 0; d < a.rank(); ++d)
                    out.ext[static_cast<std::size_t>(d)] = read_value(*a.alias_src_, static_cast<std::size_t>(d));
                break;
            case AliasShape::Kind::dims:
                for (int d = 0; d < a.rank(); ++d) {
                    const auto      u  = static_cast<std::size_t>(d);
                    const AliasDim& sp = s.dim_specs[u];
                    switch (sp.how) {
                        case AliasDim::How::fixed: out.ext[u] = sp.extent; break;
                        case AliasDim::How::per_index_host: {
                            std::vector<std::size_t> v = sp.values;
                            if (a.dim0_override_) {  // inside a ChunkedSystem: this chunk's slice of the host array
                                const std::size_t first = 0, len = *a.dim0_override_;
                                if (v.size() < first + len)
                                    throw std::invalid_argument("naga::cospan: tie alias '" + a.full_path() + "': per-index dimension " +
                                                                std::to_string(d) + " has " + std::to_string(v.size()) +
                                                                " extents but the chunk reaches index " + std::to_string(first + len));
                                v = std::vector<std::size_t>(v.begin() + static_cast<std::ptrdiff_t>(first),
                                                             v.begin() + static_cast<std::ptrdiff_t>(first + len));
                            }
                            out.per[u]       = true;
                            out.per_index[u] = std::move(v);
                            break;
                        }
                        // An input source is read through its ROOT (host snapshot), as an output is.
                        case AliasDim::How::output_value:
                            out.ext[u] = read_value(sp.input ? *sp.src->root() : *sp.src, sp.index);
                            break;
                        case AliasDim::How::output_values:
                            out.per[u]       = true;
                            out.per_index[u] = chunk_slice(a, read_values(sp.input ? *sp.src->root() : *sp.src, sp.index));
                            break;
                        case AliasDim::How::output_dim:
                            copy_alias_dim(out, u, *sp.src, static_cast<int>(sp.index), a, sp.input);
                            break;
                    }
                }
                break;
            case AliasShape::Kind::unspecified: {  // a free tie root: the shape of its initializer
                const auto& is = a.init_spec();
                if (!is || !is->shape())
                    throw std::logic_error("naga::cospan: tie alias '" + a.full_path() +
                                           "' has an unspecified shape and nothing supplies it (no connected output, no initializer with a shape)");
                if (is->shape()->rank != a.rank())
                    throw std::invalid_argument("naga::cospan: shape mismatch: tie alias '" + a.full_path() + "' has rank " +
                                                std::to_string(a.rank()) + " but its initializer gives a rank-" +
                                                std::to_string(is->shape()->rank) + " shape");
                const InitShape& sh = *is->shape();
                out.ext             = sh.extents;
                for (std::size_t u = 1; u < 3; ++u)
                    if (sh.per[u]) {  // a ragged init shape: its per-index array of the full range (cut to the chunk inside one)
                        out.per[u]       = true;
                        out.per_index[u] = chunk_slice(a, sh.per_index[u]);
                    }
                break;
            }
        }
        // A per-index dim has no scalar extent, whichever spec made it (copy_alias_dim sets 0 as well).
        for (std::size_t u = 0; u < 3; ++u)
            if (out.per[u]) out.ext[u] = 0;
        resolving_ = nullptr;
        return out;
    }

    static std::string alias_shape_str(const std::array<std::size_t, 3>& ext, const std::array<bool, 3>& per, int rank) {
        std::string out;
        for (int d = 0; d < rank; ++d) {
            if (d > 0) out += "x";
            out += per[static_cast<std::size_t>(d)] ? std::string{"(per-index)"} : std::to_string(ext[static_cast<std::size_t>(d)]);
        }
        return out;
    }

    /// A ROOT tie alias takes the shape its rule gives now. Per-index dims are resolved on the host and applied like
    /// any ragged root's: update_shape computes the block offsets and resizes the storage; the owner uploads the
    /// device copies (run_pre_compute) once the root is allocated.
    void apply_alias_root(detail::IOEntry& a) {
        const AliasEval ev = alias_eval(a);
        for (int d = 0; d < a.rank(); ++d) {
            const auto         u = static_cast<std::size_t>(d);
            detail::DimRecord& r = a.dims_[u];
            r.extent             = ev.ext[u];
            if (d >= 1) {
                r.per       = ev.per[u];
                r.per_index = ev.per_index[u];
            }
        }
        a.update_shape();
    }

    /// ChunkedSystem: resolve the shape of an OUTSIDE tie alias (a boundary root) from its rule alone, through this
    /// Component's alias resolution. Its value / shape reads go to `deps` / `shapes` (the topology cache) when given,
    /// and a source computed inside the chunked system is refused as in the full pass.
    void resolve_alias_root(detail::IOEntry& a, std::vector<detail::IOEntry*>* deps, std::vector<const detail::IOEntry*>* shapes) {
        dep_recorder_   = deps;
        shape_recorder_ = shapes;
        chunk_pass_     = true;
        const auto reset = [&] {
            dep_recorder_   = nullptr;
            shape_recorder_ = nullptr;
            chunk_pass_     = false;
        };
        try {
            apply_alias_root(a);
        } catch (...) {
            reset();
            throw;
        }
        reset();
    }

    /// The VALUES of an integer-dtype input of rank 1 [N] (column 0) or rank 2 [N, k] (column < k),
    /// on the host: one per row (FromInputValues). Negative values are an error.
    std::vector<std::size_t> read_values(detail::IOEntry& src, std::size_t column) {
        check_not_chunk_internal(src);
        note_value_dep(src.root());  // a VALUE dependency of the topology cache
        if (!detail::is_integer_dtype(src.dtype()))
            throw std::invalid_argument("naga::cospan: '" + std::string{src.name()} +
                                        "' drives per-index extents by value and must have an integer dtype");
        if (src.rank() > 2)
            throw std::invalid_argument("naga::cospan: '" + std::string{src.name()} +
                                        "' drives per-index extents by value and must be rank 1 [N] or rank 2 [N,k]");
        const std::size_t k = src.rank() == 2 ? src.extents()[1] : 1;
        if (column >= k)
            throw std::out_of_range("naga::cospan: '" + std::string{src.name()} + "' has no column " +
                                    std::to_string(column));
        ensure_storage(*src.root());
        if (!src.root()->has_storage())
            throw std::logic_error("naga::cospan: '" + std::string{src.name()} +
                                   "' drives per-index extents by value but has no storage yet");
        std::vector<std::size_t> out(src.extents()[0]);
        detail::dispatch_dtype(src.dtype(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            // Read through the root's cached host copy and its geometry (as take_snapshot does), building only
            // the small output: no whole-root temporary.
            const detail::IOEntry::Geometry g     = src.geometry();
            const std::vector<T>&           whole = detail::storage_of<T>(src).read_primary(queue_ref());
            const std::size_t               count = detail::element_count(g);
            for (std::size_t i = 0; i < out.size(); ++i) {
                if (i * k + column >= count)
                    throw std::out_of_range("naga::cospan: '" + std::string{src.name()} + "' has no element " +
                                            std::to_string(i * k + column));
                const std::size_t off = detail::element_offset(i * k + column, g);
                if (off >= whole.size())
                    throw std::out_of_range("naga::cospan: '" + src.full_path() +
                                            "' presents elements past its root's storage");
                const T v = whole[off];
                if constexpr (std::is_signed_v<T>)
                    if (v < 0)
                        throw std::invalid_argument("naga::cospan: '" + std::string{src.name()} +
                                                    "' holds a negative extent (row " + std::to_string(i) + ")");
                out[i] = static_cast<std::size_t>(v);
            }
        });
        return out;
    }

    /// during a ChunkedSystem's full-range pass a shape may not be driven by a value of a root
    /// that is internal to the chunked system (it would be computed per chunk).
    void check_not_chunk_internal(const detail::IOEntry& src) const {
        if (!chunk_pass_) return;
        // INTERNAL roots, and the re-rooted OUTPUT roots (an inside output promoted or consumed outside):
        // both are computed per chunk.
        const detail::IOEntry& root = *src.root();
        std::string            by{root.chunk_internal_by_};
        if (by.empty() && root.chunk_rerooted_ && root.kind() == IOKind::output)
            by = root.declarer() != nullptr ? root.declarer()->path() : std::string{"?"};
        if (by.empty()) return;
        const std::string who = resolving_ != nullptr ? std::string{resolving_->name()} : std::string{src.name()};
        throw std::invalid_argument("naga::cospan: '" + who + "' is sized by a value computed inside chunked system '" +
                                    by +
                                    "' (the value of '" + std::string{src.name()} +
                                    "'); chunked sizes must come from outside");
    }

    /// The VALUE of element `index` of an integer-dtype input, on the host.
    std::size_t read_value(detail::IOEntry& src, std::size_t index) {
        check_not_chunk_internal(src);
        note_value_dep(src.root());  // a VALUE dependency of the topology cache
        if (!detail::is_integer_dtype(src.dtype()))
            throw std::invalid_argument("naga::cospan: '" + std::string{src.name()} +
                                        "' drives a shape by value and must have an integer dtype");
        ensure_storage(*src.root());  // a Component allocates the roots it owns; a System needs them present
        if (!src.root()->has_storage())
            throw std::logic_error("naga::cospan: '" + std::string{src.name()} +
                                   "' drives a shape by value but has no storage yet");
        std::size_t out = 0;
        detail::dispatch_dtype(src.dtype(), [&](auto tag) {
            using T = typename decltype(tag)::type;
            // The root's cached host copy through the entry's geometry: one element, no whole-root temporary.
            const detail::IOEntry::Geometry g = src.geometry();
            if (index >= detail::element_count(g))
                throw std::out_of_range("naga::cospan: '" + std::string{src.name()} + "' has no element " +
                                        std::to_string(index));
            const std::vector<T>& whole = detail::storage_of<T>(src).read_primary(queue_ref());
            const std::size_t     off   = detail::element_offset(index, g);
            if (off >= whole.size())
                throw std::out_of_range("naga::cospan: '" + src.full_path() + "' presents elements past its root's storage");
            const T v = whole[off];
            if constexpr (std::is_signed_v<T>)
                if (v < 0)
                    throw std::invalid_argument("naga::cospan: '" + std::string{src.name()} +
                                                "' holds a negative extent");
            out = static_cast<std::size_t>(v);
        });
        return out;
    }

};

}  // namespace naga::cospan::detail