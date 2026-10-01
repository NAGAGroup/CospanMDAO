#pragma once

// The oracle of the chunked topology cache: the same model run on the LEGACY per-chunk resolution and on the
// cached REPLAY must give bitwise-equal outputs and identical per-chunk shapes. After every chunk the hook
// serialises, for every inside entry and internal root: its extents, per-index arrays, block offsets and where its
// wrapper's data pointer points relative to its root's storage. (The device index arrays are not read: the equal
// outputs prove them.)

#include <cstddef>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <naga/cospan/cospan.hpp>

namespace tor {

namespace cs = naga::cospan;

inline std::string describe_entry(const cs::detail::IOEntry& e) {
    std::ostringstream o;
    o << e.full_path() << " | " << e.extents()[0] << "x" << e.extents()[1] << "x" << e.extents()[2] << " | ragged " << e.is_ragged();
    for (int d = 1; d < e.rank(); ++d) {
        o << " | per" << d << " [";
        for (const std::size_t v : e.per_index_raw(d)) o << v << ",";
        o << "]";
    }
    o << " | offs [";
    for (const std::uint64_t v : e.block_offsets()) o << v << ",";
    o << "] | data ";
    try {
        const auto* base = static_cast<const char*>(e.root()->storage_on(e.placement()));
        const auto* data = static_cast<const char*>(e.probe_data());
        if (data == nullptr || base == nullptr)
            o << "null";
        else
            o << (data - base);
    } catch (const std::exception&) {
        o << "n/a";
    }
    return o.str();
}

struct Capture {
    std::vector<std::string> chunks;  ///< one string per chunk: every entry, in order
    std::vector<float>       out;
};

inline Capture run_once(cs::Model& m, const char* chunked_path, const std::function<std::vector<float>(cs::Model&)>& read) {
    auto&   sys = dynamic_cast<cs::ChunkedSystem&>(m.get_subsystem(chunked_path));
    Capture c;
    sys.set_chunk_hooks(nullptr, [&](std::size_t) {
        std::string s;
        for (const cs::detail::IOEntry* e : sys.chunk_entries()) s += describe_entry(*e) + "\n";
        c.chunks.push_back(std::move(s));
    });
    m.forward();
    m.wait();
    sys.set_chunk_hooks(nullptr, nullptr);
    c.out = read(m);
    return c;
}

/// Builds a model with the given replay mode (the ChunkedSystem constructor takes the process default).
template <class Make>
cs::Model make_model(bool replay, Make&& make) {
    const bool prev = cs::ChunkedSystem::default_replay();
    cs::ChunkedSystem::set_default_replay(replay);
    cs::Model m = make();
    m.setup();
    cs::ChunkedSystem::set_default_replay(prev);
    return m;
}

/// Run `forwards` forwards of the model on both paths. `between(m, f)` is called before forward f (f >= 1) on both
/// models (to re-initialise an input).
template <class Make>
void oracle(const char* chunked_path, Make&& make, const std::function<std::vector<float>(cs::Model&)>& read, int forwards = 2,
            const std::function<void(cs::Model&, int)>& between = {}) {
    cs::Model legacy = make_model(false, make);
    cs::Model replay = make_model(true, make);
    CHECK_FALSE(dynamic_cast<cs::ChunkedSystem&>(legacy.get_subsystem(chunked_path)).replay());
    CHECK(dynamic_cast<cs::ChunkedSystem&>(replay.get_subsystem(chunked_path)).replay());
    for (int f = 0; f < forwards; ++f) {
        INFO("forward " << f);
        if (f > 0 && between) {
            between(legacy, f);
            between(replay, f);
        }
        const Capture l = run_once(legacy, chunked_path, read);
        const Capture r = run_once(replay, chunked_path, read);
        REQUIRE(l.chunks.size() == r.chunks.size());
        CHECK(l.chunks.size() >= 1);
        for (std::size_t c = 0; c < l.chunks.size(); ++c) {
            INFO("chunk " << c);
            CHECK(l.chunks[c] == r.chunks[c]);
        }
        CHECK(l.out == r.out);  // bitwise (no NaN in these models)
        CHECK(dynamic_cast<cs::ChunkedSystem&>(replay.get_subsystem(chunked_path)).topology_cached());
        CHECK_FALSE(dynamic_cast<cs::ChunkedSystem&>(legacy.get_subsystem(chunked_path)).topology_cached());
    }
}

}  // namespace tor
