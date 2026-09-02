# Spec 02.C — aliasing and the finalize pass

Phase-2 chunk C. Binding contract; declarations only. Amends 01.B's
IOEntry (source/target slots are REPLACED by what follows) and gives 01.E's
structural APIs their mechanics.

## 1. The model: one relation, root + peers

Every IOEntry belongs to exactly one SET. A set has one ROOT — the entry
that owns the data and, for outputs, the entry that produces it — and any
number of PEERS: names for the same data (promotions, tie aliases) and
consumers of it (connected inputs). Every entry points at its root; the
root holds the peer list. Merging a set into another steals the peer list
and repoints every member — no chains exist and nothing is ever walked
at forward time: a peer reads `root()->data()`, one dereference.

There are no source/target links. Everything they would carry is a
property of the set: the root is the producer; each input peer is a
consumer; connection validation is each input peer against its root; a
subarray connection is a fact on the input peer ("which part of the root
I see"); `FromInput` (01.B) registers the consumer's DimRecord with its
input's root; later cycle analysis is a graph over components through
root ownership.

## 2. Aliasing rules: two-level check, construction-time flags

Each entry carries `can_be_source` and `can_be_target`. An operation on
an entry passes iff BOTH the entry's own flags AND its root's flags allow
it: the entry's flags say "what was I created to support"; the root's
flags say "has aliasing since invalidated that". Flags are set at
construction and NEVER mutated afterwards, with exactly ONE exception
(§3, tie_*). Roots need no snapshot logic: a root's own flags are its
set's flags.

```cpp
namespace naga::cospan::detail {

class IOEntry {
public:
    // ---- identity, shape, marks, lifetime: unchanged from 01.B ----------

    // ---- the set --------------------------------------------------------
    IOEntry*       root();                 ///< self iff is_root()
    const IOEntry* root() const;
    bool           is_root() const;
    std::span<IOEntry* const> peers() const;   ///< root only; empty otherwise
    bool           owns_data() const;      ///< == is_root()

    // ---- aliasing flags (construction-time; see §2/§3) ------------------
    bool can_be_source() const;
    bool can_be_target() const;

    // ---- consumer view (input peers only) -------------------------------
    /// The subarray of the root this peer sees; whole-root when unset.
    std::optional<SubarrayRef> view() const;

    // ---- capabilities (01.B, restated against the set) ------------------
    /// validate_connection: INPUT PEERS ONLY, against root() — flat-size
    /// (or view() subarray size) equality; after update_shape; every
    /// forward when either side is_dynamic().
    /// allocate(Model::MemoryTargets&): ROOTS ONLY; no-op on peers.
    /// sync_wrapper: reads root()->data() (+ view() offset/stride) into
    /// the member wrapper — which is why root regrowth needs no
    /// notification: every peer re-reads at its next sync.

private:
    IOEntry*              root_  = this;
    std::vector<IOEntry*> peers_;          // meaningful iff root_ == this
    bool can_be_source_, can_be_target_;
    std::optional<SubarrayRef> view_;
    friend class naga::cospan::System;     // the four structural ops
};

} // namespace naga::cospan::detail
```

## 3. The structural operations, as set operations

`merge(into, from)`: append `from`'s peers and `from` itself to `into`'s
peer list; set every moved entry's root to `into`. O(|set|); models are
never large enough for this to matter, and the flatness is what makes
forward free.

**declare_input / declare_output** (01.D): a new entry, root of a
singleton set. Inputs: `can_be_source = false, can_be_target = true`.
Outputs: the reverse.

**promote(path, name)** — kind-agnostic, ALWAYS allowed (including on
already-connected or already-aliased entries: promotion exists to give a
higher level a name, e.g. probing an adopted System's internals). Create
the alias, register it in the calling System's namespace, join it to the
resolved entry's set as a PEER (never a root), with its flags a SNAPSHOT
of the root's flags at creation. The snapshot is what keeps a promoted
input from becoming sourceable after its set is later connected. The
alias enumerates under `get_inputs()`/`get_outputs()` by the flags it
snapshotted.

**connect(src, dst)** — check `src.can_be_source ∧ root(src).can_be_source
∧ dst.can_be_target ∧ root(dst).can_be_target`, else throw naming the
failing level. Then `merge(root(src), root(dst))`. No flag changes: the
second connection to any name in the set now fails at the ROOT check
(the root is an output root, `can_be_target == false`) — fan-in is
blocked structurally. Fan-out is the same op repeated. Subarray forms
record `view()` on the input peer. Connecting through ANY name of a set
is the same connection.

**tie_inputs({m...}, name)** — the alias is a NEW ROOT with
`can_be_target = true, can_be_source = false`. Each member's root must be
an input root with `can_be_target == true` — a member whose set is
already connected fails here ("tie before connecting"). Merge every
member's set into the alias. THE EXCEPTION: mark each member
`can_be_target = false`, so a later connect *through a member name*
fails at the direct check. Data-wise it would be identical to connecting
through the alias; the mark exists so a tie is visible at every use
site — a discipline rule, not a correctness one, and the only
post-construction flag mutation in the system.

**tie_outputs({m...}, name, policy)** — the alias is a NEW ROOT with
`can_be_source = true, can_be_target = false`; policy `require_atomic`
(default) checks every member's dtype is atomic (01.A), `allow_plain`
is the user's assertion. Each member's root must be an output root whose
set contains NO input-kind peer — a downstream connection already exists
otherwise ("tie before connecting, or connect from the alias"); output-
kind peers (promotions) ride along. Merge. THE EXCEPTION, mirrored: mark
each member `can_be_source = false`.

Path resolution for every operation is from the caller's namespace,
dot-separated, descending children; the last segment resolves an alias
registered in that System or a Component's declared entry.

## 4. What owns memory

Every entry that is a root at finalize owns storage — including
unconnected inputs and unconnected `tie_inputs` aliases: that is how
top-level design variables and shared inputs get memory with no
IndepVarComp. Route by lifetime (02.B `MemoryTargets`): durable ∧ static
roots from the arena; all other roots via the dynamic chain at first
need and on growth. Peers own nothing, ever.

## 5. The finalize pass — local, two walks

`Model::setup` step 2 (02.B) is not a resolution pass: aliasing is
complete the moment the structural calls return. Finalize is:

1. **Fold** — depth-first, every System reports the durable ∧ static
   bytes of ITS OWN root entries (its alias roots) and every Component
   reports its own, via `get_inputs()/get_outputs()`. Model sums. No
   registry, no table.
2. Model constructs the arena at that size (and the pool iff the knob
   is set).
3. **Allocate** — the same walk again; every root allocates from the
   targets handed down; peers do nothing.
4. `FromInput` registration — each DimRecord with a `from_input` spec
   registers itself with its input's root (the producing output), whose
   Component's `post_launch` will enqueue the D2H copies.
5. Static connection validation — each static input peer against its
   root, once; dynamic pairs re-validate each forward (01.B).
6. Compile (01.E) and sync — every wrapper reads its root.

## 6. Sample

```cpp
// declared: d1.out (root O1), d2.out (root O2), c.in (root I1)
promote("d1.out", "y");       // y joins O1's set as a peer; flags = O1's
connect("y", "c.in");         // O1 absorbs I1's set: {y, c.in} peers of O1
connect("d2.out", "c.in");    // FAILS at the root check: root(c.in) = O1,
                              //   an output root, can_be_target == false
tie_outputs({"d1.out","d2.out"}, "sum");   // FAILS: O1's set has an
                              //   input-kind peer (c.in) — connected already
```
