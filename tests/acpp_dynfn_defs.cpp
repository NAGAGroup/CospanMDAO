// =============================================================================
// acpp_dynfn_defs.cpp — the second translation unit of the dynamic-function probe
// =============================================================================
//
// Definitions that live ONLY here. test_acpp_dynfn.cpp names them in a dynamic
// function configuration and never calls them, so the JIT has to link them from
// this unit's exported symbols (SYCL_EXTERNAL).
//
// Every definition writes to a slot of its own, so a wrong mapping shows up as
// a wrong value in a different slot.

#include <sycl/sycl.hpp>

// (a) define: out[1] = 42
SYCL_EXTERNAL void probe_ext_set(int* out, sycl::item<1>) { out[1] = 42; }

// (b) plain call sequence: out[2] and out[3] each get one definition's work
SYCL_EXTERNAL void probe_ext_seq0(int* out, sycl::item<1>) { out[2] += 3; }
SYCL_EXTERNAL void probe_ext_seq1(int* out, sycl::item<1>) { out[3] += 4; }

// (c) indexed call sequence of three: definition k writes out[k] = k*10 + (k+1),
// where k is the index the pass passes in
SYCL_EXTERNAL void probe_ext_idx0(int* out, int k, sycl::item<1>) { out[k] = k * 10 + 1; }
SYCL_EXTERNAL void probe_ext_idx1(int* out, int k, sycl::item<1>) { out[k] = k * 10 + 2; }
SYCL_EXTERNAL void probe_ext_idx2(int* out, int k, sycl::item<1>) { out[k] = k * 10 + 3; }

// (d) indexed call sequence of one: out[k] = k*10 + 7, with k = 0
SYCL_EXTERNAL void probe_ext_one0(int* out, int k, sycl::item<1>) { out[k] = k * 10 + 7; }
