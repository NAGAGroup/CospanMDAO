// Smoke check for the header-only skeleton and declaration layer (not the test
// suite): entries built directly (via the internal EntryAccess stand-in for
// System), allocated through the r01 §4.4 routes, pointed at wrappers, driven by
// real kernels, and read back through HostView.

#include <array>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <naga/cospan/cospan.hpp>

namespace cs = naga::cospan;
namespace sk = naga::slabkit;
using cs::DType;
using cs::detail::EntryAccess;
using cs::detail::HostViewAccess;
using cs::detail::InputEntry;
using cs::detail::OutputEntry;

// ---- wrapper triviality and accessor discipline, at compile time ------------
static_assert(std::is_trivially_copyable_v<cs::Input<DType::f32, 3>>);
static_assert(std::is_trivially_copyable_v<cs::Input<DType::u8, 1>>);
static_assert(std::is_trivially_copyable_v<cs::Output<DType::f64, 2>>);
static_assert(std::is_trivially_copyable_v<cs::Output<DType::atomic_u32, 1>>);
static_assert(std::is_trivially_copyable_v<cs::Initializer<DType::atomic_i64, 2>>);
static_assert(std::is_trivially_copyable_v<cs::Local<DType::f32, 2>>);
static_assert(std::is_same_v<decltype(std::declval<const cs::Input<DType::f32, 1>&>()[std::size_t{0}]),
                             const float&>);
static_assert(std::is_same_v<decltype(std::declval<const cs::Output<DType::f32, 1>&>()[std::size_t{0}]),
                             float&>);
static_assert(std::is_same_v<decltype(std::declval<const cs::Output<DType::atomic_u32, 1>&>()[std::size_t{0}]),
                             cs::Output<DType::atomic_u32, 1>::atomic_type>);

static int failures = 0;

static void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

template <class... S>
static std::array<cs::detail::DimRecord, sizeof...(S)> dims(const S&... s) {
    return {cs::detail::make_dim_record(s)...};
}

/// One arena sized for the given roots (fixed shapes), plus a pool.
static cs::Substrates make_substrates(sk::queue& q, const sycl::device& dev,
                                      std::initializer_list<const cs::detail::IOEntry*> roots) {
    std::size_t bytes = 1024;
    for (const auto* r : roots) bytes += r->static_bytes() + 1024;
    cs::Substrates subs;
    subs.emplace(dev, cs::MemorySubstrate{sk::DeviceArena{q, dev, sk::kind::device, bytes},
                                          sk::DevicePool{q, dev, sk::kind::device, 1 << 20}});
    return subs;
}

// ---- 1. arena route: a producer, a connected consumer, a second output ------
static void test_arena(sk::queue& q, const sycl::device& dev) {
    std::printf("-- arena route, fixed shapes\n");
    cs::Output<DType::f32, 1> a_out, c_out;
    cs::Input<DType::f32, 1>  b_in;
    OutputEntry<DType::f32, 1> a{"a", dims(1000), &a_out};
    InputEntry<DType::f32, 1>  b{"b_in", dims(1000), &b_in};
    OutputEntry<DType::f32, 1> c{"c", dims(1000), &c_out};

    for (cs::detail::IOEntry* e : {static_cast<cs::detail::IOEntry*>(&a), static_cast<cs::detail::IOEntry*>(&b),
                                   static_cast<cs::detail::IOEntry*>(&c)})
        EntryAccess::set_device(*e, dev);
    EntryAccess::merge(a, b);  // b_in becomes a peer of a
    EntryAccess::register_device(a, dev);
    EntryAccess::register_device(c, dev);

    check(a.is_root() && !b.is_root() && b.root() == &a && a.peers().size() == 1, "merge: b_in is a's peer");
    check(a.static_bytes() == 4000 && c.static_bytes() == 4000 && b.static_bytes() == 0,
          "static_bytes: roots count, peers do not");
    check(a.arena_eligible() && c.arena_eligible() && !b.arena_eligible(), "arena_eligible");

    cs::Substrates subs = make_substrates(q, dev, {&a, &c});
    b.validate_connection();
    a.allocate(q, subs);
    b.allocate(q, subs);  // peers own nothing
    c.allocate(q, subs);
    a.point_wrapper();
    b.point_wrapper();
    c.point_wrapper();

    check(subs.at(dev).arena->used() >= 8000, "arena: both roots were bumped from it");
    check(subs.at(dev).pool->outstanding() == 0, "pool: untouched by fixed roots");
    bool peer_refused = false;
    try {
        (void)b.storage_on(dev);
    } catch (const std::logic_error&) {
        peer_refused = true;
    }
    check(peer_refused, "storage_on refuses a peer");
    check(a.storage_on(dev) != nullptr && b_in.size() == 1000 && b_in.stride()[0] == 1, "peer wrapper geometry");

    q.parallel_for(sycl::range<1>{1000}, [out = a_out](sycl::id<1> i) { out[i] = 2.0f * static_cast<float>(i[0]); });
    q.parallel_for(sycl::range<1>{1000}, [in = b_in, co = c_out](sycl::id<1> i) { co[i] = in[i] + 1.0f; });

    auto hc = HostViewAccess::view<DType::f32, 1>(c, q);
    auto ha = HostViewAccess::view<DType::f32, 1>(a, q);
    auto hb = HostViewAccess::view<DType::f32, 1>(b, q);
    bool good = hc.size() == 1000 && ha.size() == 1000 && hb.size() == 1000;
    for (std::size_t i = 0; i < 1000 && good; ++i)
        good = ha[i] == 2.0f * static_cast<float>(i) && hb[sycl::id<1>{i}] == ha[i] &&
               hc[i] == 2.0f * static_cast<float>(i) + 1.0f;
    check(good, "a[i]=2i, c[i]=b_in[i]+1 read back through HostView");
}

// ---- 2. pool route: a dynamic root, resolved, reshaped (grown) --------------
static void test_dynamic_pool(sk::queue& q, const sycl::device& dev) {
    std::printf("-- pool route, dynamic root\n");
    cs::Output<DType::f32, 2> w_out;
    // dim 0 is the VALUE of element 0 of a sibling input "n"; dim 1 is fixed.
    OutputEntry<DType::f32, 2> d{"dyn", dims(cs::FromInput{"n", 0}, 4), &w_out};
    EntryAccess::set_device(d, dev);
    EntryAccess::register_device(d, dev);

    check(d.is_dynamic() && !d.arena_eligible() && d.static_bytes() == 0, "dynamic: not arena eligible");
    check(d.flat_size() == 0, "dynamic: unresolved extent starts at 0");

    cs::Substrates subs = make_substrates(q, dev, {&d});
    EntryAccess::set_resolved_extent(d, 0, 10);  // what the Component does after reading "n"
    d.update_shape();
    check(d.extents()[0] == 10 && d.extents()[1] == 4 && d.extents()[2] == 1, "update_shape applies the resolved extent");

    d.allocate(q, subs);
    check(subs.at(dev).pool->outstanding() > 0, "pool: the dynamic root was allocated from it");
    check(subs.at(dev).arena->used() == 0, "arena: untouched by a dynamic root");
    d.point_wrapper();
    check(w_out.shape()[0] == 10 && w_out.shape()[1] == 4 && w_out.stride()[0] == 4 && w_out.stride()[1] == 1,
          "wrapper shape and stride");

    q.parallel_for(sycl::range<2>{10, 4}, [w = w_out](sycl::id<2> ij) {
        w[ij] = 10.0f * static_cast<float>(ij[0]) + static_cast<float>(ij[1]);
    });
    {
        auto h = HostViewAccess::view<DType::f32, 2>(d, q);
        bool good = h.size() == 40 && h.shape()[0] == 10 && h.shape()[1] == 4;
        for (std::size_t f = 0; f < 40 && good; ++f)
            good = h[f] == 10.0f * static_cast<float>(f / 4) + static_cast<float>(f % 4);
        check(good, "2-D kernel write read back (flat, row-major)");
    }

    EntryAccess::set_resolved_extent(d, 0, 25);  // grow past capacity: reallocates through the pool
    d.update_shape();
    d.point_wrapper();
    q.parallel_for(sycl::range<2>{25, 4}, [w = w_out](sycl::id<2> ij) {
        w[ij] = 10.0f * static_cast<float>(ij[0]) + static_cast<float>(ij[1]);
    });
    auto h = HostViewAccess::view<DType::f32, 2>(d, q);
    bool good = h.size() == 100 && h.shape()[0] == 25;
    for (std::size_t i = 0; i < 25 && good; ++i)
        for (std::size_t j = 0; j < 4 && good; ++j)
            good = h[sycl::id<2>{i, j}] == 10.0f * static_cast<float>(i) + static_cast<float>(j);
    check(good, "grown 25x4 root, kernel write read back");
}

// ---- 3. a slice view: offset into the root, the root's strides ---------------
static void test_slice(sk::queue& q, const sycl::device& dev) {
    std::printf("-- slice view\n");
    cs::Output<DType::f32, 2> grid_out;
    cs::Input<DType::f32, 2>  win_in;
    cs::Output<DType::f32, 1> wo_out;
    OutputEntry<DType::f32, 2> grid{"grid", dims(8, 8), &grid_out};
    InputEntry<DType::f32, 2>  win{"win", dims(2, 3), &win_in};
    OutputEntry<DType::f32, 1> wo{"win_copy", dims(6), &wo_out};
    for (cs::detail::IOEntry* e : {static_cast<cs::detail::IOEntry*>(&grid), static_cast<cs::detail::IOEntry*>(&win),
                                   static_cast<cs::detail::IOEntry*>(&wo)})
        EntryAccess::set_device(*e, dev);
    EntryAccess::register_device(grid, dev);
    EntryAccess::register_device(wo, dev);
    EntryAccess::set_view(win, cs::subarray<2>({3, 2}, {2, 3}));
    EntryAccess::merge(grid, win);

    cs::Substrates subs = make_substrates(q, dev, {&grid, &wo});
    win.validate_connection();
    grid.allocate(q, subs);
    win.allocate(q, subs);
    wo.allocate(q, subs);
    grid.point_wrapper();
    win.point_wrapper();
    wo.point_wrapper();

    check(win_in.shape()[0] == 2 && win_in.shape()[1] == 3 && win_in.stride()[0] == 8 && win_in.stride()[1] == 1,
          "slice: view extents over the root's strides");
    check(win.stride()[0] == 8 && win.view()->offset[0] == 3 && win.view()->offset[1] == 2, "slice: entry metadata");

    q.parallel_for(sycl::range<2>{8, 8}, [g = grid_out](sycl::id<2> ij) {
        g[ij] = 10.0f * static_cast<float>(ij[0]) + static_cast<float>(ij[1]);
    });
    q.parallel_for(sycl::range<2>{2, 3}, [w = win_in, o = wo_out](sycl::id<2> rc) { o[rc[0] * 3 + rc[1]] = w[rc]; });

    auto hc = HostViewAccess::view<DType::f32, 1>(wo, q);
    bool good = hc.size() == 6;
    for (std::size_t r = 0; r < 2 && good; ++r)
        for (std::size_t c = 0; c < 3 && good; ++c)
            good = hc[r * 3 + c] == 10.0f * static_cast<float>(3 + r) + static_cast<float>(2 + c);
    check(good, "slice: kernel read of root.data+offset with root strides");

    auto hw = HostViewAccess::view<DType::f32, 2>(win, q);
    good = hw.size() == 6 && hw.shape()[0] == 2 && hw.shape()[1] == 3;
    for (std::size_t r = 0; r < 2 && good; ++r)
        for (std::size_t c = 0; c < 3 && good; ++c)
            good = hw[sycl::id<2>{r, c}] == 10.0f * static_cast<float>(3 + r) + static_cast<float>(2 + c);
    check(good, "slice: HostView presents the region only");

    {  // write through the region; the rest of the root is untouched
        auto m = HostViewAccess::view_mut<DType::f32, 2>(win, q);
        m[sycl::id<2>{0, 0}] = -1.0f;
    }
    auto hg = HostViewAccess::view<DType::f32, 2>(grid, q);
    check(hg[sycl::id<2>{3, 2}] == -1.0f && hg[sycl::id<2>{3, 3}] == 33.0f && hg[sycl::id<2>{0, 0}] == 0.0f,
          "slice: HostViewMut writes the region into the root");

    // validation refuses what it must
    InputEntry<DType::f32, 2> bad{"bad_win", dims(2, 2), nullptr};
    EntryAccess::set_device(bad, dev);
    EntryAccess::set_view(bad, cs::subarray<2>({3, 2}, {2, 3}));
    EntryAccess::merge(grid, bad);
    bool refused = false;
    try {
        bad.validate_connection();  // flat size 4 against a view of 6
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check(refused, "validate_connection: flat-size mismatch is refused");

    InputEntry<DType::f32, 2> outside{"outside_win", dims(2, 3), nullptr};
    EntryAccess::set_view(outside, cs::subarray<2>({7, 7}, {2, 3}));
    EntryAccess::merge(grid, outside);
    refused = false;
    try {
        outside.validate_connection();
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    check(refused, "validate_connection: a view outside the root is refused");
}

// ---- 4. host-kind root (D3) and an atomic output ------------------------------
static void test_host_and_atomic(sk::queue& q, const sycl::device& dev) {
    std::printf("-- host-kind root, atomic output\n");
    cs::Output<DType::i32, 1>        h_out;
    cs::Output<DType::atomic_u32, 1> acc_out;
    OutputEntry<DType::i32, 1>        h{"host_out", dims(16), &h_out};
    OutputEntry<DType::atomic_u32, 1> acc{"acc", dims(4), &acc_out};
    EntryAccess::set_device(h, dev);
    EntryAccess::set_device(acc, dev);
    EntryAccess::register_device(h, dev);
    EntryAccess::register_device(acc, dev);
    EntryAccess::set_storage_kind(h, sk::kind::host);

    check(!h.arena_eligible() && h.static_bytes() == 0 && acc.arena_eligible(), "host-kind root takes no arena bytes");

    cs::Substrates subs = make_substrates(q, dev, {&h, &acc});
    h.allocate(q, subs);
    acc.allocate(q, subs);
    h.point_wrapper();
    acc.point_wrapper();
    check(subs.at(dev).arena->used() >= 16 && subs.at(dev).arena->used() < 1024, "arena holds the atomic root only");

    q.parallel_for(sycl::range<1>{16}, [o = h_out](sycl::id<1> i) { o[i] = 3 * static_cast<int>(i[0]); });
    q.parallel_for(sycl::range<1>{1000}, [a = acc_out](sycl::id<1> i) { a[i[0] % 4].fetch_add(1u); });

    {
        auto hv = HostViewAccess::view<DType::i32, 1>(h, q);
        bool good = hv.size() == 16;
        for (std::size_t i = 0; i < 16 && good; ++i) good = hv[i] == 3 * static_cast<int>(i);
        check(good, "host-kind root: kernel write read back (queue.wait + get_access)");
    }
    {
        auto m = HostViewAccess::view_mut<DType::i32, 1>(h, q);
        m[5] = 999;
    }  // destruction commits
    auto hv2 = HostViewAccess::view<DType::i32, 1>(h, q);
    check(hv2[5] == 999 && hv2[4] == 12 && hv2[6] == 18, "HostViewMut: destruction commits into the root");

    // the atomic root, read back as its value dtype
    auto ha = HostViewAccess::view<DType::u32, 1>(acc, q);
    check(ha.size() == 4 && ha[0] == 250 && ha[1] == 250 && ha[2] == 250 && ha[3] == 250,
          "atomic Output: 1000 colliding fetch_add, 250 per slot; read as value dtype u32");
}

int main() {
    try {
        sycl::device dev{sycl::default_selector_v};
        std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
        sk::context ctx(dev);
        sk::queue   q(ctx, dev);
        test_arena(q, dev);
        q.wait();
        test_dynamic_pool(q, dev);
        q.wait();
        test_slice(q, dev);
        q.wait();
        test_host_and_atomic(q, dev);
        q.wait();
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        return 2;
    }
    std::printf(failures == 0 ? "SMOKE PASS\n" : "SMOKE FAIL (%d)\n", failures);
    return failures == 0 ? 0 : 1;
}
