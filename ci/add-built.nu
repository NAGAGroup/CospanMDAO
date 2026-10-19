# Add the built test packages (.conda files) to ci/run/pixi.toml's `built` feature.
#
#   nu ci/add-built.nu <directory holding the .conda files, searched recursively>
#
# win-64 builds on a box with Visual Studio and runs on the GPU box, which has none: the build job uploads the .conda files
# it produced, and this adds them to the run manifest so the run environments install them. When a package name occurs
# more than once, the newest file wins.
def main [packages: string] {
    let manifest = ($env.FILE_PWD | path join 'run' 'pixi.toml')
    let pattern = ($packages | path expand | path join '**' '*.conda' | str replace --all '\' '/')
    let files = (glob $pattern)
    if ($files | is-empty) { error make {msg: $"add-built.nu: no .conda files under ($packages)"} }

    # name-version-build.conda: the package name is everything before the last two dashes.
    let newest = (
        $files
        | each {|f| {path: $f, name: ($f | path basename | parse --regex '^(?<n>.+)-[^-]+-[^-]+\.conda$' | get n.0), modified: (ls $f | get modified.0)} }
        | sort-by modified --reverse
        | uniq-by name
    )
    for p in $newest {
        print $"pixi add --feature built ($p.name): ($p.path)"
        ^pixi add --manifest-path $manifest --feature built --no-install $p.path
        if $env.LAST_EXIT_CODE != 0 { error make {msg: $"add-built.nu: pixi add failed for ($p.path)"} }
    }
}
