#pragma once

// Helpers for ChunkedSystem tests.
//
//   tcc::against_plain  runs a chunked model and its unchunked twin (the same Top with a plain System where the
//                       ChunkedSystem was), forward for forward, and requires bitwise-equal outputs. The chunk
//                       loop's event window must hold no resolution, upload, allocation, free, wait or resize.
//   tcc::Recorder       records, per chunk, the shape of chosen inside entries (leading range, dim 0, per-index
//                       arrays, rebased block offsets) so a test can compare them with values computed by hand.
//   tcc::capture_forward / describe_entry: one forward with a description of every inside entry per chunk.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <naga/cospan/cospan.hpp>

namespace tcc {

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
    o << "]";
    return o.str();
}

struct Capture {
    std::vector<std::string> chunks;  ///< one string per chunk: every inside entry, in order
    std::vector<float>       out;
    cs::diag::EventCounts    window;  ///< the chunk loop's event window of this forward
};

inline Capture capture_forward(cs::Model& m, const char* chunked_path, const std::function<std::vector<float>(cs::Model&)>& read) {
    auto&   sys = dynamic_cast<cs::ChunkedSystem&>((*m.get_subsystem(chunked_path)));
    Capture c;
    sys.set_chunk_hooks(nullptr, [&](std::size_t) {
        std::string s;
        for (const cs::detail::IOEntry* e : sys.chunk_entries()) s += describe_entry(*e) + "\n";
        c.chunks.push_back(std::move(s));
    });
    m.forward();
    m.wait();
    sys.set_chunk_hooks(nullptr, nullptr);
    c.window = cs::diag::last_window();
    c.out    = read(m);
    return c;
}

/// The chunked model and its plain twin, `forwards` forwards each. `between(m, f)` is called on both before forward f
/// (f >= 1), to re-initialise an input. Returns the chunked model's captures (one per forward).
template <class MakeChunked, class MakePlain>
std::vector<Capture> against_plain(const char* chunked_path, MakeChunked&& make_chunked, MakePlain&& make_plain,
                                   const std::function<std::vector<float>(cs::Model&)>& read, int forwards = 2,
                                   const std::function<void(cs::Model&, int)>& between = {}) {
    cs::Model chunked = make_chunked();
    chunked.setup();
    cs::Model plain = make_plain();
    plain.setup();
    std::vector<Capture> caps;
    for (int f = 0; f < forwards; ++f) {
        INFO("forward " << f);
        if (f > 0 && between) {
            between(chunked, f);
            between(plain, f);
        }
        plain.forward();
        plain.wait();
        const std::vector<float> want = read(plain);
        Capture                  c    = capture_forward(chunked, chunked_path, read);
        CHECK(c.chunks.size() >= 1);
        CHECK(c.out == want);  // bitwise (no NaN in these models)
        // The chunk loop's window: nothing resolved, uploaded, allocated, freed, waited for or RESIZED.
        CHECK(c.window.resizes == 0);
        CHECK(c.window.resolve_shapes == 0);
        CHECK(c.window.full_passes == 0);
        CHECK(c.window.ragged_uploads == 0);
        CHECK(c.window.allocations == 0);
        CHECK(c.window.frees == 0);
        CHECK(c.window.queue_waits == 0);
        CHECK(dynamic_cast<cs::ChunkedSystem&>((*chunked.get_subsystem(chunked_path))).topology_cached());
        caps.push_back(std::move(c));
    }
    return caps;
}

/// What one inside entry looked like in one chunk.
struct EntryShape {
    std::size_t                                        dim0 = 0;
    std::vector<std::size_t>                           per1, per2;  ///< the per-index arrays of dims 1 and 2 (empty: scalar)
    std::vector<std::uint64_t>                         offsets;     ///< the block offsets, rebased to start at 0
    std::optional<std::pair<std::size_t, std::size_t>> range;       ///< the leading range of a connected input
};

/// Records, per chunk and per entry path, the shape the entry has when the chunk is about to run.
class Recorder {
public:
    std::vector<std::vector<EntryShape>> chunks;  ///< chunks[c][k]: entry paths[k] in chunk c

    void attach(cs::ChunkedSystem& sys, std::vector<std::string> paths) {
        chunks.clear();
        sys.set_chunk_hooks(
            [this, &sys, paths = std::move(paths)](std::size_t) {
                std::vector<EntryShape> row;
                for (const std::string& p : paths) {
                    const cs::detail::IOEntry& e = sys.entry_at(p);
                    EntryShape                 s;
                    s.dim0 = e.extents()[0];
                    if (e.rank() > 1) s.per1 = e.per_index_raw(1);
                    if (e.rank() > 2) s.per2 = e.per_index_raw(2);
                    s.offsets = e.block_offsets();
                    s.range   = e.leading_range();
                    row.push_back(std::move(s));
                }
                chunks.push_back(std::move(row));
            },
            nullptr);
    }
};

}  // namespace tcc
