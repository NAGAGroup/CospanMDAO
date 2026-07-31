def main [preset: string = "default"] {
  cmake --build $"(echo build | path join $preset)"
}
