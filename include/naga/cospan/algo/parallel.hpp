#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <sycl/sycl.hpp>
#include <slabkit/array.hpp>
#include <slabkit/context.hpp>

namespace naga::cospan::algo {

static constexpr std::size_t kDefaultWorkgroupSize = 256;

/// Exclusive prefix sum over input array.
/// \tparam T Numeric type supporting addition
/// \param input Pointer to input array (unchanged)
/// \param output Pointer to output array, must be valid USM memory of size >= n
/// \param n Number of elements
/// \param q SYCL queue for execution
/// \param ctx SlabContext for device memory allocation
/// \param device_idx Device index within ctx (default: 0)
/// \param init Identity element for the scan (default: T{0})
/// \return SYCL event that completes when scan is ready
template <typename T>
sycl::event exclusive_scan(const T* input, T* output, std::size_t n,
                           sycl::queue& q,
                           std::shared_ptr<slabkit::Context> ctx,
                           int device_idx = 0,
                           T init = T{0});

/// Stable radix sort of keys in-place, permuting values to match.
/// \tparam KeyT Type of keys (integral type, must be 32-bit)
/// \tparam ValueT Type of values
/// \param keys Pointer to keys array to sort (modified in-place, valid USM memory)
/// \param values Pointer to values array corresponding to keys (reordered, valid USM memory)
/// \param n Number of elements
/// \param q SYCL queue for execution
/// \param ctx SlabContext for device memory allocation
/// \param device_idx Device index within ctx (default: 0)
/// \return SYCL event that completes when sort is ready
/// \pre keys and values point to valid USM memory of size >= n
template <typename KeyT, typename ValueT>
sycl::event sort_by_key(KeyT* keys, ValueT* values, std::size_t n, sycl::queue& q,
                        std::shared_ptr<slabkit::Context> ctx, int device_idx = 0);

// =============================================================================
// exclusive_scan Implementation
// =============================================================================

template <typename T>
sycl::event exclusive_scan(const T* input, T* output, std::size_t n,
                           sycl::queue& q,
                           std::shared_ptr<slabkit::Context> ctx,
                           int device_idx,
                           T init) {
  if (n == 0) {
    return q.single_task([](){});
  }

  if (n == 1) {
    T init_val = init;
    T* out_ptr = output;
    return q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
      out_ptr[0] = init_val;
    });
  }

  auto device = q.get_device();
  std::size_t max_wg =
      device.get_info<sycl::info::device::max_work_group_size>();
  std::size_t wg_size = std::min(max_wg, kDefaultWorkgroupSize);
  std::size_t num_groups = (n + wg_size - 1) / wg_size;

  auto block_sums = slabkit::make_device_array<T>(sycl::range<1>(num_groups), ctx, device_idx);
  auto block_sums_view = block_sums.view();

  const T* in_ptr = input;
  T* out_ptr = output;
  T* bs_ptr  = block_sums.data();

  // Phase 1: Block-level scan
  sycl::event e1 = q.submit([&](sycl::handler& cgh) {
    auto local_mem =
        sycl::local_accessor<T, 1>(sycl::range<1>(wg_size), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>(num_groups * wg_size, wg_size),
        [=](sycl::nd_item<1> item) {
          std::size_t gid = item.get_global_id(0);
          std::size_t lid = item.get_local_id(0);
          std::size_t group_id = item.get_group(0);
          std::size_t local_size = item.get_local_range(0);

          // Load -- pad with identity for out-of-range elements
          T val = (gid < n) ? in_ptr[gid] : T{0};
          local_mem[lid] = val;
          sycl::group_barrier(item.get_group());

          // Up-sweep (reduce)
          for (std::size_t stride = 1; stride < local_size; stride *= 2) {
            std::size_t idx = (lid + 1) * stride * 2 - 1;
            if (idx < local_size)
              local_mem[idx] += local_mem[idx - stride];
            sycl::group_barrier(item.get_group());
          }

          // Store block sum and clear root
          if (lid == local_size - 1) {
            block_sums_view[sycl::id<1>{group_id}] = local_mem[lid];
            local_mem[lid] = T{0};
          }
          sycl::group_barrier(item.get_group());

          // Down-sweep
          for (std::size_t stride = local_size / 2; stride >= 1; stride /= 2) {
            std::size_t idx = (lid + 1) * stride * 2 - 1;
            if (idx < local_size) {
              T tmp = local_mem[idx];
              local_mem[idx] += local_mem[idx - stride];
              local_mem[idx - stride] = tmp;
            }
            sycl::group_barrier(item.get_group());
          }

          if (gid < n)
            out_ptr[gid] = local_mem[lid];
        });
  });

  if (num_groups > 1) {
    // Phase 2: Scan block sums (initialize block sum offsets)
    sycl::event e_block_init = q.single_task([=, num_groups = num_groups, init_val = init]() {
      T running = init_val;
      for (std::size_t i = 0; i < num_groups; ++i) {
        T old = block_sums_view[sycl::id<1>{i}];
        block_sums_view[sycl::id<1>{i}] = running;
        running += old;
      }
    });

    // Phase 3: Add block offsets (depends on block sum initialization)
    return q.submit([&](sycl::handler& cgh) {
      cgh.depends_on(e_block_init);
      cgh.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                       [=](sycl::nd_item<1> item) {
                         std::size_t gid = item.get_global_id(0);
                         std::size_t group_id = item.get_group(0);
                         if (gid < n)
                           out_ptr[gid] += block_sums_view[sycl::id<1>{group_id}];
                       });
    });
  } else if (init != T{0}) {
    // Single block -- apply init offset
    T init_val = init;
    return q.parallel_for(sycl::range<1>(n),
                       [=](sycl::id<1> idx) { out_ptr[idx] += init_val; });
  }
  return e1;
}

// =============================================================================
// sort_by_key Implementation
// =============================================================================

template <typename KeyT, typename ValueT>
sycl::event sort_by_key(KeyT* keys, ValueT* values, std::size_t n, sycl::queue& q,
                        std::shared_ptr<slabkit::Context> ctx, int device_idx) {
  if (n <= 1) {
    return q.single_task([](){});
  }

  static_assert(sizeof(KeyT) == 4,
                "sort_by_key: only 32-bit key types are supported");

  constexpr uint32_t RADIX_BITS = 8;
  constexpr uint32_t RADIX = 1u << RADIX_BITS;                   // 256
  constexpr uint32_t NUM_PASSES = sizeof(KeyT) * 8 / RADIX_BITS; // 4
  constexpr uint32_t RADIX_MASK = RADIX - 1u;

  auto device = q.get_device();
  std::size_t max_wg =
      device.get_info<sycl::info::device::max_work_group_size>();
  std::size_t wg_size = std::min(max_wg, kDefaultWorkgroupSize);
  std::size_t num_groups = (n + wg_size - 1) / wg_size;

  // Ping-pong buffers
  auto alt_keys = slabkit::make_device_array<KeyT>(sycl::range<1>(n), ctx, device_idx);
  auto alt_values = slabkit::make_device_array<ValueT>(sycl::range<1>(n), ctx, device_idx);

  // hist_matrix[g * RADIX + d] = count of digit d in WG g
  auto hist_matrix =
      slabkit::make_device_array<uint32_t>(sycl::range<1>(num_groups * RADIX), ctx, device_idx);
  // scan_out[g * RADIX + d] = absolute output base for WG g, digit d
  auto scan_out =
      slabkit::make_device_array<uint32_t>(sycl::range<1>(num_groups * RADIX), ctx, device_idx);

  auto alt_keys_view = alt_keys.view();
  auto alt_values_view = alt_values.view();
  auto hist_matrix_view = hist_matrix.view();
  auto scan_out_view = scan_out.view();

  KeyT*   src_keys = keys;
  KeyT*   dst_keys = alt_keys.data();
  ValueT* src_values = values;
  ValueT* dst_values = alt_values.data();

  bool keys_in_place = true; // track which buffer holds the result

  sycl::event pass_event{};
  for (uint32_t pass = 0; pass < NUM_PASSES; ++pass) {
    uint32_t radix_shift = pass * RADIX_BITS;

     // --- Kernel 1: count histogram into hist_matrix ---
     sycl::event hist_event = [&]() {
       KeyT* sk = src_keys;
       return q.submit([=, n = n, num_groups = num_groups](sycl::handler& cgh) {
         if (pass > 0) {
           cgh.depends_on(pass_event);
         }
         auto local_hist =
             sycl::local_accessor<uint32_t, 1>(sycl::range<1>(RADIX), cgh);

         cgh.parallel_for(
             sycl::nd_range<1>(num_groups * wg_size, wg_size),
             [=](sycl::nd_item<1> item) {
               std::size_t lid = item.get_local_id(0);
               std::size_t lrange = item.get_local_range(0);
               std::size_t group_id = item.get_group(0);
               std::size_t gid = item.get_global_id(0);

               // Zero local histogram
               for (uint32_t d = lid; d < RADIX; d += lrange)
                 local_hist[d] = 0u;
               sycl::group_barrier(item.get_group());

               // Count
               if (gid < n) {
                 uint32_t key = static_cast<uint32_t>(sk[gid]);
                 uint32_t digit = (key >> radix_shift) & RADIX_MASK;
                 sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                  sycl::memory_scope::work_group,
                                  sycl::access::address_space::local_space>
                     la(local_hist[digit]);
                 la.fetch_add(1u);
               }
               sycl::group_barrier(item.get_group());

               // Store to global hist_matrix row
               for (uint32_t d = lid; d < RADIX; d += lrange)
                 hist_matrix_view[sycl::id<1>{group_id * RADIX + d}] = local_hist[d];
             });
       });
     }();

     // --- Kernel 2: column-wise exclusive scan + add global digit offsets
     // (single_task for correctness across backends) ---
     sycl::event scan_event = [&]() {
       uint32_t ng = static_cast<uint32_t>(num_groups);
       return q.submit([&](sycl::handler& cgh) {
         cgh.depends_on(hist_event);
         cgh.single_task([=]() {
           uint32_t totals[RADIX] = {};
           // Pass 1: column exclusive scan for each digit
           for (uint32_t d = 0; d < RADIX; ++d) {
             uint32_t running = 0u;
             for (uint32_t g = 0; g < ng; ++g) {
               uint32_t cnt = hist_matrix_view[sycl::id<1>{g * RADIX + d}];
               scan_out_view[sycl::id<1>{g * RADIX + d}] = running;
               running += cnt;
             }
             totals[d] = running;
           }
           // Pass 2: add global digit offsets
           for (uint32_t d = 0; d < RADIX; ++d) {
             uint32_t global_offset = 0u;
             for (uint32_t i = 0; i < d; ++i)
               global_offset += totals[i];
             for (uint32_t g = 0; g < ng; ++g)
               scan_out_view[sycl::id<1>{g * RADIX + d}] += global_offset;
           }
         });
       });
     }();

     // --- Kernel 3: scatter (turn-based for stability) ---
     pass_event = [&]() {
       KeyT*   sk = src_keys;
       ValueT* sv = src_values;
       KeyT*   dk = dst_keys;
       ValueT* dv = dst_values;
       return q.submit([=, n = n, num_groups = num_groups](sycl::handler& cgh) {
         cgh.depends_on(scan_event);
         auto wg_base =
             sycl::local_accessor<uint32_t, 1>(sycl::range<1>(RADIX), cgh);
         auto local_cnt =
             sycl::local_accessor<uint32_t, 1>(sycl::range<1>(RADIX), cgh);

         cgh.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size),
                          [=](sycl::nd_item<1> item) {
                            std::size_t lid = item.get_local_id(0);
                            std::size_t lrange = item.get_local_range(0);
                            std::size_t group_id = item.get_group(0);
                            std::size_t gid = item.get_global_id(0);

                            // Load per-WG base offsets from scan_out
                            for (uint32_t d = lid; d < RADIX; d += lrange) {
                              wg_base[d] = scan_out_view[sycl::id<1>{group_id * RADIX + d}];
                              local_cnt[d] = 0u;
                            }
                            sycl::group_barrier(item.get_group());

                            // Pre-read key/digit for this thread
                            bool has_element = (gid < n);
                            uint32_t my_digit = 0u;
                            if (has_element) {
                              uint32_t key = static_cast<uint32_t>(sk[gid]);
                              my_digit = (key >> radix_shift) & RADIX_MASK;
                            }

                            // Turn-based rank assignment (stable: lid 0 first)
                            uint32_t my_rank = 0u;
                            for (std::size_t turn = 0; turn < lrange; ++turn) {
                              if (lid == turn && has_element)
                                my_rank = local_cnt[my_digit]++;
                              sycl::group_barrier(item.get_group());
                            }

                            if (has_element) {
                              uint32_t out_idx = wg_base[my_digit] + my_rank;
                              dk[out_idx] = sk[gid];
                              dv[out_idx] = sv[gid];
                            }
                          });
       });
     }();

     // Swap src <-> dst for next pass
     std::swap(src_keys, dst_keys);
     std::swap(src_values, dst_values);
     keys_in_place = !keys_in_place;
   }
   
   pass_event.wait();

   static_assert(NUM_PASSES % 2 == 0,
                 "sort_by_key: odd number of passes requires final copy-back, "
                 "not implemented");

   // After NUM_PASSES (which is 4, even), src_keys == original keys pointer
   // So the final result is in place. Confirm:
   static_assert(NUM_PASSES == 4, "NUM_PASSES must be 4");

   return pass_event;
}

}  // namespace naga::cospan::algo
