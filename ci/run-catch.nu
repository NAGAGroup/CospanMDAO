# Run a Catch2 test binary through pixi, skipping known failures by exact test name.
#
#   nu ci/run-catch.nu <manifest> <environment> <executable> [<exclude file>]
#
# The exclude file lists one exact test name per line; blank lines and lines starting with # are ignored. Each name
# becomes a Catch2 `~"name"` term (terms separated by a space are ANDed), with Catch2's special characters escaped.
# linux-64 and win-64 never pass an exclude file: their failures are reported, not excluded.
def main [
    manifest: string       # pixi manifest to run in
    environment: string    # pixi environment
    executable: string     # test binary on that environment's PATH
    exclude_file: string = ""  # optional list of test names to skip
] {
    let names = if $exclude_file != "" and ($exclude_file | path exists) {
        open --raw $exclude_file
        | lines
        | each {|l| $l | str trim }
        | where {|l| $l != "" and not ($l | str starts-with "#") }
    } else {
        []
    }

    # Catch2 test-spec escapes: backslash first, then comma, brackets, asterisk.
    let filter = if ($names | is-empty) {
        []
    } else {
        [($names
            | each {|n|
                let e = ($n | str replace --all '\' '\\' | str replace --all ',' '\,' | str replace --all '[' '\[' | str replace --all ']' '\]' | str replace --all '*' '\*')
                $'~"($e)"'
            }
            | str join ' ')]
    }

    if ($names | is-not-empty) { print $"excluding ($names | length) known failure\(s\) from ($exclude_file)" }
    ^pixi run --manifest-path $manifest -e $environment $executable ...$filter
}
