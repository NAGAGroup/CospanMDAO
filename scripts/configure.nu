def main [preset: string = "default"] {
  cmake -S . -B $"(echo build | path join $preset)" --preset $preset
  ln -sf $"(echo build | path join $preset compile_commands.json)" compile_commands.json
}
