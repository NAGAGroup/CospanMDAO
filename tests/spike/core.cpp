#include "core.hpp"

namespace spike {

void launch_r1(sycl::queue& q, sycl::AdaptiveCpp_jit::dynamic_function_config& cfg,
               std::size_t total_groups, std::size_t wg, const StageRecord* records,
               const std::byte* bodies, std::size_t local_bytes) {
    q.submit([&](sycl::handler& h) {
         sycl::local_accessor<std::byte, 1> local{sycl::range<1>{local_bytes ? local_bytes : 1}, h};
         h.parallel_for(sycl::nd_range<1>{total_groups * wg, wg},
                        cfg.apply([=](sycl::nd_item<1> it) {
                            FnOffsetInfo info{records, 0};
                            kb_entry_r1(it, info, bodies,
                                        local.get_multi_ptr<sycl::access::decorated::no>().get());
                        }));
     }).wait();
}

}  // namespace spike
