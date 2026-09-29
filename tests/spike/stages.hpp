#pragma once
#include <type_traits>
#include "core.hpp"

namespace spike {

struct Scale   { float* out; float k; };             // out[i] = k * i
struct AddOne  { const float* in; float* out; };     // out[i] = in[i] + 1
struct Reverse { const float* in; float* out; };     // group body: reverse within each work-group

template <class B>
SYCL_EXTERNAL void kb_fn(sycl::nd_item<1> it, FnOffsetInfo& info, const std::byte* bodies,
                         std::byte* local_base) {
    const StageRecord& r = info.next();  // first statement, unconditional
    const std::size_t  g = it.get_group_linear_id();
    if (g < r.group_offset || g - r.group_offset >= r.group_count) return;  // whole groups only
    const std::size_t flat = (g - r.group_offset) * it.get_local_range(0) + it.get_local_linear_id();
    const B           b    = *reinterpret_cast<const B*>(bodies + r.body_off);

    if constexpr (std::is_same_v<B, Reverse>) {
        float*            tile = reinterpret_cast<float*>(local_base);
        const std::size_t l    = it.get_local_linear_id();
        const std::size_t n    = it.get_local_range(0);
        tile[l] = flat < r.flat_extent ? b.in[flat] : 0.f;
        sycl::group_barrier(it.get_group());
        if (flat < r.flat_extent) b.out[flat] = tile[n - 1 - l];
    } else {
        if (flat >= r.flat_extent) return;
        if constexpr (std::is_same_v<B, Scale>) b.out[flat] = b.k * static_cast<float>(flat);
        else b.out[flat] = b.in[flat] + 1.f;
    }
}

}  // namespace spike
