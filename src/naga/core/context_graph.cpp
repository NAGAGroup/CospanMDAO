#include "naga/core/context.hpp"
#include "naga/core/component.hpp"
#include <stack>
#include <functional>
#include <iostream>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace naga::core {

Context::ComponentGraph Context::build_component_graph() const {
    ComponentGraph g;

    // Assign indices to all registered components
    g.N = static_cast<int>(component_nodes_.size());
    g.idx_to_name.resize(g.N);
    g.adj.resize(g.N);
    for (int i = 0; i < g.N; ++i) {
        g.idx_to_name[i] = component_nodes_[i].name;
        g.name_to_idx[component_nodes_[i].name] = i;
    }

    // Helper: map a component name to its index, with FQN walk-up fallback.
    // Needed because resolve_source_component may return a name not directly
    // in name_to_idx (e.g. a nested component name) — walk up the FQN prefix
    // chain until we find a registered ancestor. This is a harmless no-op
    // fallback for well-formed component graphs where all names resolve directly.
    auto find_idx = [&](const std::string& comp_name) -> int {
        std::string name = comp_name;
        while (!name.empty()) {
            auto it = g.name_to_idx.find(name);
            if (it != g.name_to_idx.end()) return it->second;
            auto pos = name.rfind('/');
            if (pos == std::string::npos) return -1;
            name = name.substr(0, pos);
        }
        return -1;
    };

    // For each entry that is a target (is_target == true), build edges using
    // the canonical resolve_source_component / resolve_target_components helpers
    // so that passthrough chains are handled consistently with the rest of the
    // framework.
    for (const auto& entry : entries_) {
        if (!entry.is_target) continue;

        // Resolve target and source components via FQN walk-up + connection tracing
        auto tgt_comps = resolve_target_components(entry.name);
        std::string src_comp = resolve_source_component(entry.sources[0]);
        int src_idx = find_idx(src_comp);

        if (src_idx < 0) continue;

        for (const auto& tgt_comp : tgt_comps) {
            int tgt_idx = find_idx(tgt_comp);
            if (tgt_idx < 0) continue;
            if (src_idx == tgt_idx) continue;

            // Dedup and record edge
            auto& adj_list = g.adj[src_idx];
            if (std::find(adj_list.begin(), adj_list.end(), tgt_idx) == adj_list.end())
                adj_list.push_back(tgt_idx);

            g.edge_entries[{src_idx, tgt_idx}].emplace_back(entry.sources[0], entry.name);
        }
    }

    // Pass 2: share-group edge union. Components that co-write a shared
    // output (members of a single shared-storage owner promotion) must share
    // execution-ordering edges, so that every producer feeding ANY member
    // runs before ALL members (forward), and every consumer of ANY member
    // runs after ALL members. The share alias itself creates no edge (the
    // owner promotion is sourceless and system-owned), so the union is
    // synthesized here at component granularity.
    // Downstream ordering (SCC/back-edge/layers) is unchanged and operates on the
    // enriched adjacency.
    auto add_union_edge = [&](int a, int b,
                              const std::vector<std::pair<std::string,std::string>>& entries_for_edge) {
        if (a == b) return;
        auto& adj_list = g.adj[a];
        if (std::find(adj_list.begin(), adj_list.end(), b) == adj_list.end())
            adj_list.push_back(b);
        // Mirror edge_entries from the motivating real edge so that if this
        // synthetic edge is later chosen as a back-edge, grad-slot attribution
        // (which keys on the source owning entry) still resolves correctly.
        auto& dst = g.edge_entries[{a, b}];
        for (const auto& e : entries_for_edge)
            if (std::find(dst.begin(), dst.end(), e) == dst.end())
                dst.push_back(e);
    };

    for (const auto& entry : entries_) {
        if (entry.kind != EntryKind::Promotion || !entry.is_shareable) continue;
        // Multi-level re-share (2026-07-17): a nested share owner (itself
        // enrolled as a member of a higher-level share) is subsumed by its
        // ROOT owner's union pass — running it separately would union only a
        // subset of the storage's writers, leaving consumers under-ordered
        // relative to the other share groups' members. Only root owners run.
        if (entry.is_target) continue;
        // entry is a ROOT shared-storage owner. Flatten the member set
        // transitively: expand nested share owners down to the LEAF fqns.
        // Both the component group and consumer detection must operate on
        // leaves, because real connections source from leaf computing
        // members, never from owner promotions.
        //
        // ONLY enrolled members (is_share_member) are followed: with the
        // connect-from-owner relax (share_outputs), owner
        // targets ALSO contain ordinary consumer inputs — flattening those
        // would misclassify consumers as group members (producers would gain
        // bogus edges into them, and the real member→consumer ordering would
        // never be synthesized).
        std::unordered_set<std::string> member_fqns;
        {
            std::vector<std::string> stack;
            for (const auto& t : entry.targets) {
                auto tit = entry_map_.find(t);
                if (tit != entry_map_.end() &&
                    entries_[tit->second].is_share_member)
                    stack.push_back(t);
            }
            while (!stack.empty()) {
                std::string m = std::move(stack.back());
                stack.pop_back();
                auto mit = entry_map_.find(m);
                if (mit == entry_map_.end()) continue;
                const Entry& me = entries_[mit->second];
                if (me.kind == EntryKind::Promotion && me.is_shareable) {
                    for (const auto& t : me.targets) {
                        auto tit = entry_map_.find(t);
                        if (tit != entry_map_.end() &&
                            entries_[tit->second].is_share_member)
                            stack.push_back(t);
                    }
                } else {
                    member_fqns.insert(m);
                }
            }
        }
        // Collect the distinct component-graph node indices that own the members.
        std::vector<int> group;
        for (const auto& member_fqn : member_fqns) {
            for (const auto& comp : resolve_target_components(member_fqn)) {
                int idx = find_idx(comp);
                if (idx >= 0 &&
                    std::find(group.begin(), group.end(), idx) == group.end())
                    group.push_back(idx);
            }
        }
        if (group.size() < 2) continue;  // nothing to union
        std::sort(group.begin(), group.end());
        std::unordered_set<int> gset(group.begin(), group.end());

        // External producers: nodes u (outside the group) with an edge into any
        // group member. Capture, per producer, the edge_entries of the real
        // edge(s) it has into the group (to mirror onto synthetic edges).
        std::vector<int> producers;
        std::unordered_map<int, std::vector<std::pair<std::string,std::string>>> prod_entries;
        for (int u = 0; u < g.N; ++u) {
            if (gset.count(u)) continue;
            bool feeds = false;
            for (int v : g.adj[u]) {
                if (!gset.count(v)) continue;
                feeds = true;
                auto it = g.edge_entries.find({u, v});
                if (it != g.edge_entries.end())
                    for (const auto& e : it->second)
                        prod_entries[u].push_back(e);
            }
            if (feeds) producers.push_back(u);
        }

        // External consumers: nodes w (outside the group) that consume the
        // SHARED STORAGE itself — i.e. the motivating edge's source entry
        // RESOLVES (through its owning chain) to this share's root owner.
        // Direct member-fqn sources resolve trivially; promotion-mediated
        // reads (a subsystem promoting a member field outward,
        // 2026-07-17) resolve through the promotion to the member and on to
        // the owner. A consumer of a member's PRIVATE output (e.g. a sensor's
        // regular readout) is NOT a share consumer — its source resolves to
        // its own entry, not the owner: unioning it to all members fabricates
        // phantom cycles (observed: a consumer reading a member's
        // private outputs resurrected an already-cut SCC).
        auto consumes_this_share = [&](const std::string& src_fqn) -> bool {
            if (member_fqns.count(src_fqn)) return true;
            auto sit = entry_map_.find(src_fqn);
            if (sit == entry_map_.end()) return false;
            const Entry& owning = get_owning_entry(entries_[sit->second]);
            return owning.name == entry.name;
        };
        std::vector<int> consumers;
        std::unordered_map<int, std::vector<std::pair<std::string,std::string>>> cons_entries;
        {
            std::unordered_set<int> seen;
            for (int gi : group) {
                for (int w : g.adj[gi]) {
                    if (gset.count(w)) continue;
                    auto it = g.edge_entries.find({gi, w});
                    if (it == g.edge_entries.end()) continue;
                    bool consumes_share = false;
                    for (const auto& e : it->second)
                        if (consumes_this_share(e.first)) { consumes_share = true; break; }
                    if (!consumes_share) continue;
                    if (seen.insert(w).second) consumers.push_back(w);
                    for (const auto& e : it->second)
                        if (consumes_this_share(e.first))
                            cons_entries[w].push_back(e);
                }
            }

            // Consumers connected FROM the owner (or through a plain
            // promotion over it) have NO real edge at all — the alias is
            // sourceless, so Pass 1 derived nothing and the scan above cannot
            // see them. Synthesize their membership here: any non-member
            // target entry whose owning chain resolves to this root owner is
            // a consumer of the shared storage; it gets the full
            // member→consumer union (and, when the consumer is a marked
            // entry point, those synthetic edges are the CUT — back-edge
            // attribution keys on the owner via get_owning_entry, making the
            // owner the recurrence carrier. This is how the register pattern
            // model/u → u_register(cut) → model/u_prev works.)
            for (const auto& ce : entries_) {
                if (!ce.is_target || ce.is_share_member) continue;
                if (ce.sources.empty()) continue;
                if (!consumes_this_share(ce.sources[0])) continue;
                for (const auto& comp : resolve_target_components(ce.name)) {
                    int w = find_idx(comp);
                    if (w < 0 || gset.count(w)) continue;
                    if (seen.insert(w).second) consumers.push_back(w);
                    cons_entries[w].emplace_back(entry.name, ce.name);
                }
            }
        }

        std::sort(producers.begin(), producers.end());
        std::sort(consumers.begin(), consumers.end());

        // Union in-edges: every producer -> every group member.
        for (int p : producers)
            for (int gi : group)
                add_union_edge(p, gi, prod_entries[p]);
        // Union out-edges: every group member -> every consumer.
        for (int gi : group)
            for (int c : consumers)
                add_union_edge(gi, c, cons_entries[c]);
        // No intra-group edges are added (members are mutually concurrent).

        std::cerr << "[SHARE] owner=" << entry.name << " group_size=" << group.size()
                  << " producers=" << producers.size()
                  << " consumers=" << consumers.size() << "\n";
    }

    return g;
}

std::vector<std::vector<int>> Context::compute_sccs(
    const ComponentGraph& g,
    const std::set<std::pair<int,int>>& cuts) const {
    int N = g.N;
    std::vector<int> disc(N, -1);
    std::vector<int> low(N, 0);
    std::vector<bool> on_stack(N, false);

    std::vector<int> tarjan_stack;         // the "SCC stack" of visited nodes
    std::vector<std::vector<int>> sccs;
    int timer = 0;

    // Iterative Tarjan using an explicit work stack of (node, edge_iterator_position) frames
    struct Frame {
        int node;
        int edge_pos;  // index into g.adj[node] — which neighbor to process next
    };

    for (int start = 0; start < N; ++start) {
        if (disc[start] != -1) continue;  // already visited

        std::vector<Frame> work_stack;
        work_stack.push_back({start, 0});
        disc[start] = low[start] = timer++;
        on_stack[start] = true;
        tarjan_stack.push_back(start);

        while (!work_stack.empty()) {
            Frame& frame = work_stack.back();
            int u = frame.node;
            const auto& neighbors = g.adj[u];

            if (frame.edge_pos < static_cast<int>(neighbors.size())) {
                int v = neighbors[frame.edge_pos];
                ++frame.edge_pos;

                // Declared cuts (entry-point inputs) are not edges for cycle
                // purposes — any cycle through an entry point dissolves here.
                if (cuts.count({u, v})) continue;

                if (disc[v] == -1) {
                    // Tree edge — push v onto work stack
                    disc[v] = low[v] = timer++;
                    on_stack[v] = true;
                    tarjan_stack.push_back(v);
                    work_stack.push_back({v, 0});
                } else if (on_stack[v]) {
                    // Back/cross edge to a node already on the SCC stack
                    low[u] = std::min(low[u], disc[v]);
                }
                // If v is already fully processed (not on_stack), ignore it
            } else {
                // All neighbors of u processed — pop frame
                work_stack.pop_back();

                // Propagate low value to parent
                if (!work_stack.empty()) {
                    int parent = work_stack.back().node;
                    low[parent] = std::min(low[parent], low[u]);
                }

                // If u is an SCC root (low[u] == disc[u]), pop the SCC
                if (low[u] == disc[u]) {
                    std::vector<int> scc;
                    while (true) {
                        int w = tarjan_stack.back();
                        tarjan_stack.pop_back();
                        on_stack[w] = false;
                        scc.push_back(w);
                        if (w == u) break;
                    }
                    sccs.push_back(std::move(scc));
                }
            }
        }
    }

    return sccs;
}

void Context::register_back_edge(const ComponentGraph& g, int u, int v,
                                 const std::string& src_fqn,
                                 const std::string& tgt_fqn) {
    auto entry_it = entry_map_.find(src_fqn);
    if (entry_it == entry_map_.end()) return;

    // Walk back through passthrough/target entries to find the real owning
    // non-target entry for slot allocation (get_owning_entry follows
    // sources[0] until !is_target).
    const Entry& owner = get_owning_entry(entries_[entry_it->second]);
    Entry& owner_mut = entries_[entry_map_.at(owner.name)];
    owner_mut.is_back_edge_source = true;
    // The cut source is the RECURRENT OUTPUT: it needs the prev/curr primal
    // slot pair for the deferred read and the grad slot pair for the adjoint
    // transport. DVs are excluded — they are constant within a settle
    // (deferred == fresh) and the optimizer requires their single-slot grad.
    if (!owner_mut.is_design_variable) {
        // Recurrent outputs are f32-only in P1: the slot rotation and adjoint
        // transport machinery assume float storage. A data-tier (or f64)
        // entry acting as a cut source has no supported semantics yet — fail
        // loudly rather than mis-allocate (FRAMEWORK_CLEANUP S1 plumbing).
        if (owner_mut.dtype != DType::f32)
            throw std::runtime_error(
                "Context::register_back_edge(): cut source '" + owner.name +
                "' has dtype " + dtype_name(owner_mut.dtype) +
                " — recurrent outputs are f32-only in P1");
        owner_mut.is_recurrent_output = true;
    }

    back_edges_.push_back(BackEdge{
        g.idx_to_name[u], g.idx_to_name[v],
        owner.name, tgt_fqn
    });
    back_edge_source_entries_.insert(owner.name);

    // Per-connection deferral: exactly this target entry reads the prev slot.
    // Guard: only flag the target when it actually belongs to the cut consumer
    // v. Share-group union edges mirror edge_entries from a DIFFERENT member's
    // real connection (kept for carrier/slot attribution above); flagging that
    // mirrored target would defer an uncut connection on the wrong component —
    // an unintended one-tick lag (audit 2026-07-11). Consumers reached only by
    // a synthetic edge read shared storage through the owner's dual slots, so
    // no per-connection deferral applies to them.
    bool tgt_belongs_to_v = false;
    for (const auto& comp : resolve_target_components(tgt_fqn)) {
        if (comp == g.idx_to_name[v]) { tgt_belongs_to_v = true; break; }
    }
    auto tgt_it = entry_map_.find(tgt_fqn);
    if (tgt_belongs_to_v && tgt_it != entry_map_.end())
        entries_[tgt_it->second].is_deferred_read = true;

    std::cerr << "[BE] " << g.idx_to_name[u]
              << " -> " << g.idx_to_name[v]
              << " (src_entry=" << owner.name
              << (tgt_belongs_to_v
                      ? ", deferred_tgt=" + tgt_fqn
                      : ", mirrored_tgt=" + tgt_fqn + " [no deferral]")
              << ")\n";
}

std::set<std::pair<int,int>> Context::identify_declared_cuts(const ComponentGraph& g) {
    std::set<std::pair<int,int>> cuts;
    for (const auto& ep : entry_points_) {
        auto it = g.name_to_idx.find(ep);
        if (it == g.name_to_idx.end()) continue;
        int v = it->second;
        for (int u = 0; u < g.N; ++u) {
            if (std::find(g.adj[u].begin(), g.adj[u].end(), v) == g.adj[u].end())
                continue;
            cuts.insert({u, v});
            auto eit = g.edge_entries.find({u, v});
            if (eit != g.edge_entries.end())
                for (const auto& [src_fqn, tgt_fqn] : eit->second)
                    register_back_edge(g, u, v, src_fqn, tgt_fqn);
        }
    }
    return cuts;
}

std::vector<int> Context::scc_dfs_order(
    const ComponentGraph& g,
    const std::vector<int>& entry_seeds,
    const std::unordered_set<int>& in_scc,
    const std::set<std::pair<int,int>>& cuts)
{
    std::unordered_set<int> visited;
    std::unordered_set<int> on_stack;
    std::vector<int> postorder;

    struct Frame { int node; int edge_pos; };
    std::vector<Frame> dfs;

    // Core DFS loop, shared between the seeded pass and the remaining-node
    // guard pass. Registers emergent back-edges via register_back_edge.
    auto run_dfs = [&]() {
        while (!dfs.empty()) {
            Frame& frame = dfs.back();
            int u = frame.node;
            bool pushed = false;

            while (frame.edge_pos < static_cast<int>(g.adj[u].size())) {
                int v = g.adj[u][frame.edge_pos];
                ++frame.edge_pos;

                if (!in_scc.count(v)) continue;      // skip inter-SCC edges
                if (cuts.count({u, v})) continue;    // declared cut — already deferred

                if (on_stack.count(v)) {
                    // Emergent back-edge: u → v, v is an ancestor in this DFS
                    // tree, and no entry point declared a register on this
                    // cycle. The traversal cuts here.
                    auto eit = g.edge_entries.find(std::make_pair(u, v));
                    if (eit != g.edge_entries.end())
                        for (const auto& [src_fqn, tgt_fqn] : eit->second)
                            register_back_edge(g, u, v, src_fqn, tgt_fqn);
                } else if (!visited.count(v)) {
                    visited.insert(v);
                    on_stack.insert(v);
                    dfs.push_back({v, 0});
                    pushed = true;
                    break;
                }
            }

            if (!pushed) {
                postorder.push_back(u);
                on_stack.erase(u);
                dfs.pop_back();
            }
        }
    };

    // Seed from all entry nodes simultaneously (sorted for determinism)
    for (int seed : entry_seeds) {
        visited.insert(seed);
        on_stack.insert(seed);
        dfs.push_back({seed, 0});
    }
    run_dfs();

    // Handle any SCC nodes not reachable from entry seeds
    // (shouldn't happen in a well-formed SCC, but guard for robustness)
    std::vector<int> remaining;
    for (int v : in_scc) {
        if (!visited.count(v)) remaining.push_back(v);
    }
    std::sort(remaining.begin(), remaining.end());
    for (int seed : remaining) {
        if (visited.count(seed)) continue;
        visited.insert(seed);
        on_stack.insert(seed);
        dfs.push_back({seed, 0});
        run_dfs();
    }

    // Reverse postorder = valid topological order for the non-back-edge subgraph
    std::reverse(postorder.begin(), postorder.end());
    return postorder;
}

void Context::compute_dag_and_order(ComponentGraph& g,
                                    const std::vector<std::vector<int>>& sccs,
                                    const std::set<std::pair<int,int>>& cuts) {
    int N = g.N;
    int n_sccs = static_cast<int>(sccs.size());

    // Map each node to its SCC id
    std::vector<int> node_scc(N, -1);
    for (int s = 0; s < n_sccs; ++s)
        for (int v : sccs[s])
            node_scc[v] = s;

    // For each non-trivial SCC: run the single DFS that simultaneously
    // identifies back-edges and produces the intra-SCC execution order.
    // Store per-SCC node orderings for later expansion.
    std::vector<std::vector<int>> scc_orders(n_sccs);

    for (int s = 0; s < n_sccs; ++s) {
        const auto& scc = sccs[s];
        if (scc.size() <= 1) {
            scc_orders[s] = scc;  // trivial — single node
            continue;
        }

        std::unordered_set<int> in_scc(scc.begin(), scc.end());

        // Find entry seeds for the intra-SCC DFS. We seed from two sources:
        //
        // 1. Nodes with incoming edges from OUTSIDE the SCC (external entry points).
        //    These are reachable from the global entry points and are the natural
        //    starting nodes for the DFS.
        //
        // 2. Nodes with ZERO intra-SCC incoming edges (intra-SCC roots).
        //    These nodes have no intra-SCC predecessor, so any intra-SCC edge
        //    pointing TO them must be a back-edge. Seeding from them ensures
        //    the DFS visits them first, so edges like ssm_readout→firing/u_prev
        //    are correctly identified as back-edges (firing has zero intra-SCC
        //    incoming non-back-edges, so it should be a DFS root).
        //
        // Seeding from both sets simultaneously gives the DFS the correct
        // starting points to identify all cycle-closing back-edges properly.
        std::vector<int> entry_seeds;
        {
            std::unordered_set<int> seen;

            // Source 1: nodes with external incoming edges
            for (int u = 0; u < N; ++u) {
                if (in_scc.count(u)) continue;
                for (int v : g.adj[u]) {
                    if (in_scc.count(v) && !seen.count(v)) {
                        seen.insert(v);
                        entry_seeds.push_back(v);
                    }
                }
            }

            // Source 2: intra-SCC roots (zero intra-SCC incoming edges)
            // Compute intra-SCC in-degree for each node
            std::unordered_map<int,int> intra_indegree;
            for (int v : scc) intra_indegree[v] = 0;
            for (int u : scc)
                for (int v : g.adj[u])
                    if (in_scc.count(v))
                        ++intra_indegree[v];
            for (int v : scc) {
                if (intra_indegree[v] == 0 && !seen.count(v)) {
                    seen.insert(v);
                    entry_seeds.push_back(v);
                }
            }

            if (entry_seeds.empty()) {
                int min_node = *std::min_element(scc.begin(), scc.end());
                entry_seeds.push_back(min_node);
            }
            std::sort(entry_seeds.begin(), entry_seeds.end());
        }

        std::cerr << "[SCC] size=" << scc.size() << " entry_seeds:";
        for (int v : entry_seeds) std::cerr << " " << g.idx_to_name[v];
        std::cerr << "\n";

        size_t be_before = back_edges_.size();
        scc_orders[s] = scc_dfs_order(g, entry_seeds, in_scc, cuts);

        // Any back-edge registered by this DFS is an EMERGENT cut: this SCC
        // is a cycle (or cycle cluster) with no entry point on it, so delay
        // placement was chosen by the traversal, not declared by the model.
        // Deterministic (entry seeds are pinned) but incidental — the carrier
        // may sit on a poorly-conditioned edge. Permissive by design; warn
        // loudly so the modeler can declare a register instead.
        if (back_edges_.size() > be_before) {
            std::cerr << "[SCC-WARN] cycle without entry point:";
            for (int v : scc) std::cerr << " " << g.idx_to_name[v];
            std::cerr << "\n[SCC-WARN]   emergent cut(s):";
            for (size_t b = be_before; b < back_edges_.size(); ++b)
                std::cerr << " " << back_edges_[b].source_component_name
                          << "->" << back_edges_[b].target_component_name;
            std::cerr << "\n[SCC-WARN]   delay placement here is incidental — "
                         "mark a component in this cycle as an entry point to "
                         "declare it.\n";
        }
    }

    // Now build back-edge set for condensation (back_edges_ populated above)
    std::set<std::pair<int,int>> be_pairs;
    for (const auto& be : back_edges_) {
        auto sit = g.name_to_idx.find(be.source_component_name);
        auto tit = g.name_to_idx.find(be.target_component_name);
        if (sit != g.name_to_idx.end() && tit != g.name_to_idx.end())
            be_pairs.insert({sit->second, tit->second});
    }

    // Build condensation DAG (inter-SCC non-back-edges only)
    std::vector<std::set<int>> cond_adj(n_sccs);
    std::vector<int> cond_indegree(n_sccs, 0);
    for (int u = 0; u < N; ++u) {
        for (int v : g.adj[u]) {
            if (be_pairs.count({u, v})) continue;
            int su = node_scc[u], sv = node_scc[v];
            if (su == sv) continue;
            if (!cond_adj[su].count(sv)) {
                cond_adj[su].insert(sv);
                ++cond_indegree[sv];
            }
        }
    }

    // Compute reachable SCC set from entry points (BFS over all edges)
    std::unordered_set<int> reachable_sccs;
    {
        std::vector<int> bfs;
        for (const auto& ep : entry_points_) {
            auto it = g.name_to_idx.find(ep);
            if (it == g.name_to_idx.end()) continue;
            int sid = node_scc[it->second];
            if (!reachable_sccs.count(sid)) {
                reachable_sccs.insert(sid);
                bfs.push_back(sid);
            }
        }
        // Expand via condensation edges
        for (size_t i = 0; i < bfs.size(); ++i)
            for (int t : cond_adj[bfs[i]])
                if (!reachable_sccs.count(t)) { reachable_sccs.insert(t); bfs.push_back(t); }
        // Also via back-edges (nodes reachable via cycles)
        bool changed = true;
        while (changed) {
            changed = false;
            for (int u = 0; u < N; ++u) {
                if (!reachable_sccs.count(node_scc[u])) continue;
                for (int v : g.adj[u]) {
                    int sv = node_scc[v];
                    if (!reachable_sccs.count(sv)) { reachable_sccs.insert(sv); changed = true; }
                }
            }
        }
    }

    // Kahn's on condensation — seed all in-degree-0 SCCs
    std::vector<int> q;
    for (int s = 0; s < n_sccs; ++s)
        if (cond_indegree[s] == 0)
            q.push_back(s);
    std::sort(q.begin(), q.end());

    std::vector<int> order;
    order.reserve(N);
    int reachable_count = 0;
    for (int s = 0; s < n_sccs; ++s)
        if (reachable_sccs.count(s))
            reachable_count += static_cast<int>(sccs[s].size());

    std::vector<int> cond_indegree_copy = cond_indegree;

    while (!q.empty()) {
        int s = q.front(); q.erase(q.begin());

        if (!reachable_sccs.count(s)) {
            for (int v : sccs[s])
                std::cerr << "Warning: component '" << g.idx_to_name[v]
                          << "' is unreachable from entry points and will not execute.\n";
        } else {
            // Append this SCC's nodes in DFS reverse-postorder
            for (int v : scc_orders[s])
                order.push_back(v);
        }

        std::vector<int> nexts;
        for (int t : cond_adj[s])
            if (--cond_indegree_copy[t] == 0)
                nexts.push_back(t);
        std::sort(nexts.begin(), nexts.end());
        q.insert(q.end(), nexts.begin(), nexts.end());
        std::sort(q.begin(), q.end());
    }

    // Hard completeness check: every reachable component must appear in order
    if (static_cast<int>(order.size()) != reachable_count) {
        throw std::runtime_error(
            "compute_dag_and_order: " +
            std::to_string(reachable_count - static_cast<int>(order.size())) +
            " reachable component(s) dropped from execution order — "
            "back-edge set is not a valid feedback arc set");
    }

    execution_order_.clear();
    execution_order_.reserve(order.size());
    for (int idx : order)
        execution_order_.push_back(component_nodes_[idx].ptr);

    // Log final execution order
    for (int idx : order)
        std::cerr << "[ORDER] " << g.idx_to_name[idx] << "\n";
}

void Context::compute_backward_active_sets(const ComponentGraph& g) {
    int N = g.N;

    // Build reverse adjacency list following ALL edges (including back-edges).
    // Every node that participates in any path into an objective must appear
    // in the backward active set — including nodes whose only outgoing edges
    // are back-edges (they still need their backward() called so gradients
    // flow through the recurrent connection).
    std::vector<std::vector<int>> radj(N);
    for (int u = 0; u < N; ++u)
        for (int v : g.adj[u])
            radj[v].push_back(u);

    // BFS backward from seed indices via all reverse edges.
    auto bfs = [&](const std::vector<int>& seeds) {
        std::unordered_set<Component*> result;
        std::vector<bool> visited(N, false);
        std::queue<int> q;
        for (int s : seeds) {
            if (!visited[s]) { visited[s] = true; q.push(s); }
        }
        while (!q.empty()) {
            int u = q.front(); q.pop();
            result.insert(component_nodes_[u].ptr);
            for (int p : radj[u])
                if (!visited[p]) { visited[p] = true; q.push(p); }
        }
        return result;
    };

    // Seed backward pass from objective-owning components. Objectives are
    // where seed_objective_grad() injects the adjoint, so the reverse-reachable
    // set from them is exactly the set of components with nonzero gradient
    // flow — anything else would do provably-zero backward work. (This
    // replaces the old declared-exit-point API, which was redundant: every
    // exit point in practice was an objective owner.)
    std::vector<int> obj_seeds;
    {
        std::unordered_set<int> seen;
        for (const auto& entry : entries_) {
            if (!entry.is_objective) continue;
            // Walk up the FQN prefix chain to the registered owning component.
            std::string name = entry.name;
            int idx = -1;
            while (true) {
                auto pos = name.rfind('/');
                if (pos == std::string::npos) break;
                name = name.substr(0, pos);
                auto it = g.name_to_idx.find(name);
                if (it != g.name_to_idx.end()) { idx = it->second; break; }
            }
            if (idx >= 0 && seen.insert(idx).second)
                obj_seeds.push_back(idx);
        }
        std::sort(obj_seeds.begin(), obj_seeds.end());
    }
    if (obj_seeds.empty())
        std::cerr << "Warning: no objectives assigned to any registered "
                     "component — backward active set is empty "
                     "(forward-only context).\n";

    // For K_backward_ > 1, additional backward iterations are seeded from
    // back-edge source components (recurrent state producers).
    std::vector<int> be_seeds;
    for (const auto& be : back_edges_) {
        auto it = g.name_to_idx.find(be.source_component_name);
        if (it != g.name_to_idx.end())
            be_seeds.push_back(it->second);
    }

    backward_active_sets_.resize(K_backward_);
    backward_active_sets_[0] = bfs(obj_seeds);
    for (size_t k = 1; k < K_backward_; ++k)
        backward_active_sets_[k] = bfs(be_seeds);
}

const Context::Entry& Context::get_owning_entry(const Entry& entry) const {
    if (!entry.is_target) return entry;
    if (entry.sources.empty())
        throw std::runtime_error("get_owning_entry: target entry '" + entry.name + "' has no sources");
    auto it = entry_map_.find(entry.sources[0]);
    if (it == entry_map_.end())
        throw std::runtime_error("get_owning_entry: source '" + entry.sources[0] + "' not found in entry_map");
    return get_owning_entry(entries_[it->second]);
}

void Context::assign_component_layers(const ComponentGraph& g) {
    // Build back-edge exclusion set (component graph index pairs) for O(1) lookup
    std::set<std::pair<int,int>> be_set;
    for (const auto& be : back_edges_) {
        auto si = g.name_to_idx.find(be.source_component_name);
        auto ti = g.name_to_idx.find(be.target_component_name);
        if (si != g.name_to_idx.end() && ti != g.name_to_idx.end())
            be_set.insert({si->second, ti->second});
    }

    // Build predecessor list from g.adj, excluding back-edges.
    // g.adj[u] contains successors of u; pred[v] contains non-back-edge predecessors of v.
    std::vector<std::vector<int>> pred(g.N);
    for (int u = 0; u < g.N; ++u)
        for (int v : g.adj[u])
            if (!be_set.count({u, v}))
                pred[v].push_back(u);

    // Map component name -> index in execution_order_ (separate index space from g.name_to_idx).
    // NOTE: component graph indices (g.name_to_idx) and execution_order_ indices are two separate
    // index spaces. pred[] is indexed by component graph index; component_layer_[] is indexed by
    // execution_order_ position. The lookups below translate between them.
    std::unordered_map<std::string, size_t> exec_idx;
    for (size_t i = 0; i < execution_order_.size(); ++i)
        exec_idx[execution_order_[i]->name()] = i;

    // Assign layers. execution_order_ is a valid topological order of the non-back-edge subgraph,
    // so all non-back-edge predecessors of execution_order_[i] appear at positions < i and
    // already have their layer assigned when we process component i.
    component_layer_.assign(execution_order_.size(), 0);
    for (size_t i = 0; i < execution_order_.size(); ++i) {
        auto nit = g.name_to_idx.find(execution_order_[i]->name());
        if (nit == g.name_to_idx.end()) continue;
        int cidx = nit->second;
        size_t max_pred = 0;
        bool has_pred = false;
        for (int p : pred[cidx]) {
            auto eit = exec_idx.find(g.idx_to_name[p]);
            if (eit == exec_idx.end()) continue;
            max_pred = std::max(max_pred, component_layer_[eit->second]);
            has_pred = true;
        }
        component_layer_[i] = has_pred ? max_pred + 1 : 0;
    }

    // Bucket into components_by_layer_
    size_t n_layers = *std::max_element(component_layer_.begin(), component_layer_.end()) + 1;
    components_by_layer_.assign(n_layers, {});
    for (size_t i = 0; i < execution_order_.size(); ++i)
        components_by_layer_[component_layer_[i]].push_back(execution_order_[i]);

    // Max layer width = number of queues needed
    max_layer_width_ = 0;
    for (auto& layer : components_by_layer_)
        max_layer_width_ = std::max(max_layer_width_, layer.size());

    // Assign queue index = position within layer
    component_queue_idx_.resize(execution_order_.size());
    for (auto& layer : components_by_layer_)
        for (size_t pos = 0; pos < layer.size(); ++pos)
            component_queue_idx_[exec_idx[layer[pos]->name()]] = pos;

    // Diagnostic logging
    for (size_t i = 0; i < execution_order_.size(); ++i)
        std::cerr << "[LAYER] " << execution_order_[i]->name()
                  << " layer=" << component_layer_[i]
                  << " queue=" << component_queue_idx_[i] << "\n";
    std::cerr << "[LAYER] pool_size=" << max_layer_width_ << "\n";
}

} // namespace naga::core
