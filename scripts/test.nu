def main [preset: string = "default"] {
  ctest --test-dir $"(echo build | path join $preset)" --output-on-failure
}
