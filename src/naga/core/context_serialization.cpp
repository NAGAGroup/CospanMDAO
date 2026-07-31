#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "naga/core/context.hpp"

// =============================================================================
// Model snapshot (spec: scratch/MODEL_SNAPSHOT_SCRATCH.md, 2026-07-21).
//
// "Model" = weights plus ALL state (recurrent slots, statevars, buffers) —
// never training state (no grads, no optimizer moments). Generic over
// entries: the framework knows nothing about what any buffer means.
//
// Three rules carry the whole design:
//   1. OWNERS ONLY — an entry serializes iff !is_target. Targets (including
//      share members, whose nested leaves have data == nullptr) alias their
//      owner's storage and need nothing at load.
//   2. LOGICAL SLOT ORDER — recurrent owners save prev then curr through the
//      current slot-index roles, and load writes through the destination's
//      roles. Rotation parity never matters.
//   3. DTYPE-GENERIC — payloads are raw bytes (count × dtype_size), so the
//      data tier serializes identically to f32.
// =============================================================================

namespace naga::core {

namespace {
constexpr const char*    kSnapshotMagic   = "naga-model-snapshot";
constexpr std::uint32_t  kSnapshotVersion = 1;
} // namespace

void Context::save(const std::string& path) {
    assert_phase(Phase::Finalized, "save()");

    // Drain all queues before device→host transfers.
    for (auto& q : queue_pool_) q.wait();

    std::ofstream os(path, std::ios::binary);
    if (!os)
        throw std::runtime_error("Context::save(): cannot open '" + path +
                                 "' for writing");
    cereal::BinaryOutputArchive archive(os);

    archive(std::string(kSnapshotMagic));
    archive(kSnapshotVersion);

    // Owners only (rule 1).
    std::uint64_t n_owners = 0;
    for (const auto& entry : entries_)
        if (!entry.is_target) ++n_owners;
    archive(n_owners);

    for (const auto& entry : entries_) {
        if (entry.is_target) continue;

        const size_t count = shape3_size(entry.shape3);
        const size_t bytes = count * dtype_size(entry.dtype);

        // Slot pair present ⇔ recurrent owner (StateVar or back-edge cut
        // source). Logical order: prev, then curr (rule 2).
        auto sit = primal_slot_idx_.find(entry.name);
        const bool slotted = (sit != primal_slot_idx_.end());

        std::vector<const void*> slot_ptrs;
        if (slotted) {
            const auto& slots = primal_slots_.at(entry.name);
            slot_ptrs = {slots[sit->second.prev].get(),
                         slots[sit->second.curr].get()};
        } else {
            slot_ptrs = {entry.data};
        }

        archive(entry.name);
        archive(static_cast<std::uint8_t>(entry.dtype));
        archive(static_cast<std::uint64_t>(count));
        archive(static_cast<std::uint32_t>(slot_ptrs.size()));

        std::vector<std::uint8_t> host(bytes);
        for (const void* ptr : slot_ptrs) {
            if (ptr == nullptr)
                throw std::runtime_error(
                    "Context::save(): owning entry '" + entry.name +
                    "' has null storage post-finalize — this is a bug");
            queue_pool_[0].memcpy(host.data(), ptr, bytes).wait();
            archive(host);
        }
    }
}

void Context::load(const std::string& path) {
    assert_phase(Phase::Finalized, "load()");

    // Drain all queues before host→device transfers.
    for (auto& q : queue_pool_) q.wait();

    std::ifstream is(path, std::ios::binary);
    if (!is)
        throw std::runtime_error("Context::load(): cannot open '" + path +
                                 "'");
    cereal::BinaryInputArchive archive(is);

    std::string   magic;
    std::uint32_t version = 0;
    archive(magic);
    if (magic != kSnapshotMagic)
        throw std::runtime_error(
            "Context::load(): '" + path + "' is not a model snapshot "
            "(bad magic — pre-rewrite files are not loadable)");
    archive(version);
    if (version != kSnapshotVersion)
        throw std::runtime_error(
            "Context::load(): snapshot version " + std::to_string(version) +
            " unsupported (expected " + std::to_string(kSnapshotVersion) +
            ")");

    std::uint64_t n_file = 0;
    archive(n_file);

    std::uint64_t matched = 0, skipped = 0;
    std::vector<std::uint8_t> payload;

    for (std::uint64_t i = 0; i < n_file; ++i) {
        std::string   name;
        std::uint8_t  dtype_raw = 0;
        std::uint64_t count     = 0;
        std::uint32_t n_slots   = 0;
        archive(name);
        archive(dtype_raw);
        archive(count);
        archive(n_slots);

        auto it = entry_map_.find(name);
        if (it == entry_map_.end()) {
            // Unknown in this graph: consume payloads and move on. Staged
            // builds legitimately differ by bolt-on components; the summary
            // line below keeps this honest.
            for (std::uint32_t s = 0; s < n_slots; ++s) archive(payload);
            ++skipped;
            continue;
        }

        auto& entry = entries_[it->second];
        if (entry.is_target) {
            // The entry was an OWNER when the snapshot was written but is a
            // connection target/alias in this graph (a later revision
            // promoted or connected it). Its data now lives under the new
            // owner: if that owner is in the snapshot it is restored there;
            // if not, it keeps fresh init. Either way the alias's stale
            // payload must not be applied — skip it loudly (2026-07-25,
            // in-graph oracle rework: sourceless rs in1s / gt became
            // promotion members).
            std::fprintf(stderr,
                         "[SNAPSHOT] skip '%s': owner-turned-alias in this "
                         "graph revision (data owned elsewhere now)\n",
                         name.c_str());
            for (std::uint32_t s = 0; s < n_slots; ++s) archive(payload);
            ++skipped;
            continue;
        }

        const auto dtype = static_cast<DType>(dtype_raw);
        if (entry.dtype != dtype)
            throw std::runtime_error(
                "Context::load(): entry '" + name + "' dtype mismatch: "
                "file has " + dtype_name(dtype) + ", graph has " +
                dtype_name(entry.dtype));

        const size_t expected = shape3_size(entry.shape3);
        if (count != expected)
            throw std::runtime_error(
                "Context::load(): entry '" + name + "' size mismatch: "
                "file has " + std::to_string(count) + ", graph expects " +
                std::to_string(expected));

        auto sit = primal_slot_idx_.find(entry.name);
        const bool slotted = (sit != primal_slot_idx_.end());
        const std::uint32_t graph_slots = slotted ? 2u : 1u;
        if (n_slots != graph_slots) {
            // Recurrence structure differs: the entry was (or wasn't) a
            // recurrent owner when the snapshot was written. Same class as
            // the owner-turned-alias case above — graph revisions
            // legitimately change cut structure (e.g. an observer register
            // dissolving); the stale payload must not be applied. Skip
            // loudly; the entry keeps its fresh init (2026-07-26,
            // COMMIT-AT-DEQ rework).
            std::fprintf(stderr,
                         "[SNAPSHOT] skip '%s': slot-arity mismatch (file "
                         "%u, graph %u) — recurrence structure changed in "
                         "this graph revision\n",
                         name.c_str(), n_slots, graph_slots);
            for (std::uint32_t s = 0; s < n_slots; ++s) archive(payload);
            ++skipped;
            continue;
        }

        // Destination pointers in logical order (prev, curr) — rule 2.
        std::vector<void*> slot_ptrs;
        if (slotted) {
            const auto& slots = primal_slots_.at(entry.name);
            slot_ptrs = {slots[sit->second.prev].get(),
                         slots[sit->second.curr].get()};
        } else {
            slot_ptrs = {entry.data};
        }

        const size_t bytes = count * dtype_size(entry.dtype);
        for (void* ptr : slot_ptrs) {
            archive(payload);
            if (payload.size() != bytes)
                throw std::runtime_error(
                    "Context::load(): entry '" + name + "' payload size " +
                    std::to_string(payload.size()) + " != expected " +
                    std::to_string(bytes) + " bytes — corrupt snapshot");
            queue_pool_[0].memcpy(ptr, payload.data(), bytes).wait();
        }
        ++matched;
    }

    // Honest accounting: silent FQN drift shows up here as huge skipped /
    // untouched counts instead of a quiet no-op load.
    std::uint64_t n_owners = 0;
    for (const auto& entry : entries_)
        if (!entry.is_target) ++n_owners;
    std::fprintf(stderr,
                 "[SNAPSHOT] loaded '%s': matched=%llu skipped_unknown=%llu "
                 "untouched=%llu\n",
                 path.c_str(),
                 static_cast<unsigned long long>(matched),
                 static_cast<unsigned long long>(skipped),
                 static_cast<unsigned long long>(n_owners - matched));
}

} // namespace naga::core
