# examples/run_benchmarks.nu — the fusion benchmarks (gemver-chain, ns-residual) on CUDA, with the run protocol written down.
#
#   nu examples/run_benchmarks.nu [--only all|gemver|ns-residual] [--timed 5] [--max-warmups 8]
#   pixi run benchmarks            # the same, through the root pixi task
#
# Run it from the repository root. It needs the cuda environment (pixi run -e cuda ...) and, for the cuBLAS reference, the
# cublas environment built: `pixi run -e cublas build-cublas`.
#
# THE PROTOCOL
#   1. JIT warm-up. AdaptiveCpp compiles kernels on first use and prints "... new binaries being JIT-compiled ..." on stderr
#      while its runtime optimization has not settled. For every CoSpan binary and argument set the script runs the program as
#      fresh sequential processes until stderr no longer carries that warning, at most --max-warmups (8) runs, and reports
#      how many it needed (or that it did not converge). The cuBLAS program is ahead-of-time compiled: one discarded run.
#      EVERY VARIANT RUNS IN ITS OWN PROCESS (--variant <v>), per size, for both benchmarks (ns-residual: also per range_split strategy of the fused variant): in one process (--variant all)
#      the variants interfere (the fused ones slow down and turn bimodal), so `all` is used for the correctness run only.
#   2. Timed runs. After the warm-up, --timed (5) runs; every number is the MEDIAN of the five, with the min-max spread.
#      Each run times 20 iterations after 3 warm-up iterations inside the process (the program's own flags).
#   3. Settings. ACPP_ADAPTIVITY_LEVEL and every other environment variable are left at their defaults, and every binary
#      is the one the pixi environment built (the same compile flags everywhere).
#   4. A correctness run per argument set (--check, one iteration) is recorded next to the timings.
#   5. The environment is recorded: GPU, driver, CUDA, date.
#
# Raw stdout and stderr of every run, the results as markdown and as JSON: build/benchmarks/.

const jit_marker = 'new binaries being JIT-compiled'
const out_dir = 'build/benchmarks'

# ---- running and parsing --------------------------------------------------------------------------------------------

# Run `cmd` once with stdout and stderr in files named after `tag`; a non-zero exit is reported in `ok`, not raised.
def run-logged [cmd: list<string>, tag: string] {
    mkdir $out_dir
    let out = $'($out_dir)/($tag).out'
    let err = $'($out_dir)/($tag).err'
    let ok = try {
        run-external ...$cmd o> $out e> $err
        true
    } catch {
        false
    }
    {ok: $ok, stdout: (open --raw $out), stderr: (open --raw $err)}
}

# Fresh sequential runs until the JIT warning is gone from stderr, at most `max` of them.
def warm-up [cmd: list<string>, tag: string, max: int] {
    mut runs = 0
    mut converged = false
    while $runs < $max and not $converged {
        $runs += 1
        let r = (run-logged $cmd $'($tag)-warm($runs)')
        if not $r.ok {
            error make {msg: $'warm-up run ($runs) of ($tag) failed: ($cmd | str join " "); see ($out_dir)/($tag)-warm($runs).out and .err'}
        }
        $converged = not ($r.stderr | str contains $jit_marker)
    }
    {runs: $runs, converged: $converged}
}

def median [xs: list<float>] {
    let s = ($xs | sort)
    let n = ($s | length)
    if ($n mod 2) == 1 { $s | get (($n - 1) // 2) } else { (($s | get ($n // 2 - 1)) + ($s | get ($n // 2))) / 2 }
}

# Rows of a gemver-chain or cuBLAS table: variant, ms/chain, GB/s.
def parse-gemver-rows [text: string] {
    $text | lines | each {|l| $l | parse --regex '^(?<variant>[a-z]+)\s+(?<ms>\d+\.\d+)\s+\S+\s+(?<gbs>\d+\.\d+)\s*$' } | flatten
}

# Rows of the ns-residual table: variant (unfused, fused-flatten, ..., handwritten), ms/eval, GB/s.
def parse-ns-rows [text: string] {
    $text | lines | each {|l| $l | parse --regex '^(?<variant>[a-z-]+)\s+(?<ms>\d+\.\d+)\s+(?<gbs>\d+\.\d+)\s*$' } | flatten
}

# The worst relative L2 of each variant from the "check <variant> vs reference, rel L2: x1 1e-07 ..." lines.
def parse-gemver-checks [text: string] {
    $text | lines | each {|l|
        $l | parse --regex '^\s*check (?<variant>\w+)\s+vs reference, rel L2:(?<vals>.*?)\s+(?<status>OK|FAIL)\s*$'
    } | flatten | each {|r|
        let vals = ($r.vals | parse --regex '(?<name>[a-zA-Z0-9]+) (?<v>[0-9.]+e[-+][0-9]+|inf|nan)' | get v | each {|v| try { $v | into float } catch { 1.0e300 } })
        {variant: $r.variant, worst: ($vals | math max), status: $r.status}
    }
}

# Per variant, from the "  check <variant>: R_relL2=.. det_rel=.. atomic_minmax_equal=.. atomic_norm_gap_rel=.. ... OK" lines.
def parse-ns-checks [text: string] {
    $text | lines | each {|l|
        $l | parse --regex '^\s+check (?<variant>[a-z-]+): R_relL2=(?<l2>\S+) det_rel=(?<det>\S+) atomic_minmax_equal=(?<mm>\w+) atomic_norm_gap_rel=(?<gap>\S+).* (?<status>OK|FAIL)\s*$'
    } | flatten
}

# Median, min and max of `field` per variant over a list of runs' rows.
def summarize [rows: table, field: string] {
    $rows | group-by variant | items {|v, rs|
        let xs = ($rs | get $field | each {|x| $x | into float })
        {variant: $v, median: (median $xs), min: ($xs | math min), max: ($xs | math max)}
    }
}

# `timed` runs of `cmd`; returns the parsed rows of every run and how many of them still printed the JIT warning.
def timed-runs [cmd: list<string>, tag: string, timed: int, parse: closure] {
    mut rows = []
    mut warned = 0
    for i in 1..$timed {
        let r = (run-logged $cmd $'($tag)-run($i)')
        if not $r.ok { error make {msg: $'timed run ($i) of ($tag) failed: ($cmd | str join " "); see ($out_dir)/($tag)-run($i).out and .err'} }
        if ($r.stderr | str contains $jit_marker) { $warned += 1 }
        $rows = ($rows | append (do $parse $r.stdout))
    }
    {rows: $rows, warned: $warned}
}

def fmt [x: float] {
    if $x >= 100 { $x | into string --decimals 1 } else if $x >= 10 { $x | into string --decimals 2 } else { $x | into string --decimals 3 }
}

# 1.93e-7 style, three significant digits (what the programs print for errors).
def sci [x: float] {
    if $x == 0 { return '0' }
    if $x > 1.0e200 { return 'inf' }
    let e = ($x | math log 10 | math floor | into int)
    let m = ($x / (10.0 ** $e) | math round --precision 2)
    $'($m)e($e)'
}

# ---- the environment -------------------------------------------------------------------------------------------------

def environment [] {
    let smi = '/usr/lib/wsl/lib/nvidia-smi'
    let smi = if ($smi | path exists) { $smi } else { 'nvidia-smi' }
    let gpu = (^$smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | str trim)
    let cuda = (^$smi | lines | where {|l| $l | str contains 'CUDA' } | first | str trim)
    {
        date: (date now | format date '%Y-%m-%d %H:%M %Z')
        gpu: $gpu
        nvidia_smi_header: $cuda
        acpp_adaptivity_level: ($env.ACPP_ADAPTIVITY_LEVEL? | default 'unset (AdaptiveCpp default)')
    }
}

# ---- gemver-chain ----------------------------------------------------------------------------------------------------
# EVERY VARIANT RUNS IN ITS OWN PROCESS (--variant <v>): run in one process (--variant all) the variants interfere with each
# other (the fused ones slow down and turn bimodal), so all is used for the correctness run only.

def bench-gemver [timed: int, max_warm: int] {
    mut table = []
    mut notes = []
    for n in [2048 4096 8192 16384] {
        let tag = $"gemver-n($n)"
        let args = ["--n" $"($n)" "--iters" "20" "--warmup" "3"]
        mut rows = []
        for v in [naive crafted unfused parallel fused onepass] {
            print $"gemver-chain n=($n) ($v): JIT warm-up ..."
            let cmd = (["pixi" "run" "-e" "cuda" "gemver-chain"] | append $args | append ["--variant" $v])
            let warm = (warm-up $cmd $"($tag)-($v)" $max_warm)
            let runs = (timed-runs $cmd $"($tag)-($v)" $timed {|t| parse-gemver-rows $t })
            $rows = ($rows | append $runs.rows)
            $notes = ($notes | append {n: $n, variant: $v, warmup_runs: $warm.runs, converged: $warm.converged, timed_runs_with_jit_warning: $runs.warned})
        }

        print $"gemver-chain n=($n): cuBLAS ..."
        let cublas = (["pixi" "run" "-e" "cublas" "gemver-chain-cublas"] | append $args)
        let _ = (run-logged $cublas $"($tag)-cublas-discard")   # ahead-of-time compiled: no JIT, one discarded run
        let cb_runs = (timed-runs $cublas $"($tag)-cublas" $timed {|t| parse-gemver-rows $t })
        $rows = ($rows | append $cb_runs.rows)
        $notes = ($notes | append {n: $n, variant: "cublas", warmup_runs: 0, converged: true, timed_runs_with_jit_warning: 0})

        # correctness: every variant in one process, one iteration, against the host reference
        let chk = (run-logged (["pixi" "run" "-e" "cuda" "gemver-chain" "--n" $"($n)" "--iters" "1" "--warmup" "1" "--variant" "all" "--check"]) $"($tag)-check")
        let cb_chk = (run-logged (["pixi" "run" "-e" "cublas" "gemver-chain-cublas" "--n" $"($n)" "--iters" "1" "--warmup" "1" "--check"]) $"($tag)-cublas-check")
        let checks = ((parse-gemver-checks $chk.stdout) | append (parse-gemver-checks $cb_chk.stdout))

        let ms = (summarize $rows "ms")
        let gbs = (summarize $rows "gbs")
        for m in $ms {
            let g = ($gbs | where variant == $m.variant | first)
            let c = ($checks | where variant == $m.variant)
            $table = ($table | append {
                n: $n, variant: $m.variant, ms_median: $m.median, ms_min: $m.min, ms_max: $m.max, gbs_median: $g.median,
                worst_rel_l2: (if ($c | is-empty) { null } else { $c.0.worst }), check: (if ($c | is-empty) { "not run" } else { $c.0.status })
            })
        }
    }
    {table: $table, notes: $notes}
}

# ---- ns-residual ------------------------------------------------------------------------------------------------------
# Every variant, and every range_split strategy of the fused variant, in its own process. Walls BC, minmod limiter.
# GB/s = nx * ny * 32 B / time: the NOMINAL traffic, 4 state fields read + 4 R fields written per cell, f32 (not measured).

def bench-ns [timed: int, max_warm: int] {
    mut table = []
    mut notes = []
    for n in [1024 2048 4096] {
        let tag = $"ns-n($n)"
        let args = ["--nx" $"($n)" "--ny" $"($n)" "--iters" "20" "--warmup" "3" "--bc" "walls" "--limiter" "minmod"]
        mut rows = []
        for spec in [
            {name: "unfused", extra: ["--variant" "unfused"]}
            {name: "fused-flatten", extra: ["--variant" "fused" "--range-split" "flatten"]}
            {name: "fused-pad", extra: ["--variant" "fused" "--range-split" "pad"]}
            {name: "fused-padded", extra: ["--variant" "fused" "--range-split" "padded"]}
            {name: "handwritten", extra: ["--variant" "handwritten"]}
        ] {
            print $"ns-residual n=($n) ($spec.name): JIT warm-up ..."
            let cmd = (["pixi" "run" "-e" "cuda" "ns-residual"] | append $args | append $spec.extra)
            let warm = (warm-up $cmd $"($tag)-($spec.name)" $max_warm)
            let runs = (timed-runs $cmd $"($tag)-($spec.name)" $timed {|t| parse-ns-rows $t })
            $rows = ($rows | append $runs.rows)
            $notes = ($notes | append {n: $n, variant: $spec.name, warmup_runs: $warm.runs, converged: $warm.converged, timed_runs_with_jit_warning: $runs.warned})
        }
        # correctness: every variant and strategy in one process, one iteration, against the host double reference
        let chk = (run-logged (["pixi" "run" "-e" "cuda" "ns-residual" "--nx" $"($n)" "--ny" $"($n)" "--iters" "1" "--warmup" "1" "--bc" "walls" "--limiter" "minmod" "--variant" "all" "--range-split" "all" "--check"]) $"($tag)-check")
        let checks = (parse-ns-checks $chk.stdout)

        let bytes = ($n | into float) * ($n | into float) * 32.0
        for r in (summarize $rows "ms") {
            let c = ($checks | where variant == $r.variant)
            $table = ($table | append {
                n: $n, variant: $r.variant, ms_median: $r.median, ms_min: $r.min, ms_max: $r.max, gbs: ($bytes / ($r.median * 1e6)),
                r_rel_l2: (if ($c | is-empty) { null } else { $c.0.l2 }), norm_gap: (if ($c | is-empty) { null } else { $c.0.gap }),
                check: (if ($c | is-empty) { "not run" } else { $c.0.status })
            })
        }
    }
    {table: $table, notes: $notes}
}

# ---- the report -------------------------------------------------------------------------------------------------------

def gemver-rows-md [rows: table] {
    $rows | each {|r|
        $'| ($r.n) | ($r.variant) | (fmt $r.ms_median) | (fmt $r.ms_min)–(fmt $r.ms_max) | (fmt $r.gbs_median) | (if $r.worst_rel_l2 == null { "-" } else { sci $r.worst_rel_l2 }) | ($r.check) |'
    } | str join (char nl)
}

def gemver-markdown [res: record] {
    let head = "| n | variant | ms/chain median | min–max | GB/s nominal | worst rel L2 | check |\n|---:|---|---:|---|---:|---:|---|"
    let main_rows = ($res.table | where variant in ["naive" "crafted" "cublas"])
    let other_rows = ($res.table | where variant not-in ["naive" "crafted" "cublas"])
    let notes = ($res.notes | each {|r| $'- n = ($r.n), ($r.variant): warm-up runs needed: ($r.warmup_runs), ($r.converged | if $in { "converged" } else { "DID NOT CONVERGE" }); timed runs that still printed the JIT warning: ($r.timed_runs_with_jit_warning)' } | str join (char nl))
    $"## gemver-chain, one process per variant\n\nMain table\n\n($head)\n(gemver-rows-md $main_rows)\n\nOther variants\n\n($head)\n(gemver-rows-md $other_rows)\n\n($notes)\n"
}

def ns-markdown [res: record] {
    let rows = ($res.table | each {|r|
        $'| ($r.n) | ($r.variant) | (fmt $r.ms_median) | (fmt $r.ms_min)–(fmt $r.ms_max) | (fmt $r.gbs) | (if $r.r_rel_l2 == null { "-" } else { $r.r_rel_l2 }) | (if $r.norm_gap == null { "-" } else { $r.norm_gap }) | ($r.check) |'
    } | str join (char nl))
    let notes = ($res.notes | each {|r| $'- n = ($r.n), ($r.variant): warm-up runs needed: ($r.warmup_runs), ($r.converged | if $in { "converged" } else { "DID NOT CONVERGE" }); timed runs that still printed the JIT warning: ($r.timed_runs_with_jit_warning)' } | str join (char nl))
    $"## ns-residual, one process per variant and strategy, walls BC, minmod\n\nGB/s = nx * ny * 32 B / time, the nominal traffic: 4 state fields read + 4 R fields written per cell, f32, that is nx*ny*8*4 B, not measured.\n\n| n, nx = ny | variant | ms/eval median | min–max | GB/s nominal | R rel L2 vs reference | atomic-vs-deterministic norm gap | check |\n|---:|---|---:|---|---:|---:|---:|---|\n($rows)\n\n($notes)\n"
}

def main [--only: string = 'all', --timed: int = 5, --max-warmups: int = 8] {
    if $only not-in ['all' 'gemver' 'ns-residual'] { error make {msg: '--only must be all, gemver or ns-residual'} }
    mkdir $out_dir
    let env_info = (environment)
    print ($env_info | table --expand)

    mut md = $"# fusion benchmarks\n\n```\n($env_info | to yaml)```\n\nprotocol: JIT warm-up runs until the warning is gone, at most ($max_warmups) of them, then ($timed) timed runs; median and min–max.\n\n"
    mut json = {environment: $env_info}
    if $only in ['all' 'gemver'] {
        let g = (bench-gemver $timed $max_warmups)
        $md = $md + (gemver-markdown $g)
        $json = ($json | insert gemver $g)
    }
    if $only in ['all' 'ns-residual'] {
        let s = (bench-ns $timed $max_warmups)
        $md = $md + "\n" + (ns-markdown $s)
        $json = ($json | insert ns_residual $s)
    }
    $md | save --force $'($out_dir)/results.md'
    $json | to json --indent 2 | save --force $'($out_dir)/results.json'
    print $md
    print $'results: ($out_dir)/results.md and results.json; raw logs next to them'
}
