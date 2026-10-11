"""Sphinx configuration for the CoSpan documentation (minimal)."""

import os
from pathlib import Path

# The repository root, from this file's own location: docs/site/conf.py -> root.
ROOT = Path(__file__).resolve().parents[2]

project = "CoSpan"
release = os.environ.get("PIXI_PROJECT_VERSION", "dev")

extensions = ["myst_parser", "breathe"]
source_suffix = {".md": "markdown"}

html_theme = "alabaster"  # Sphinx's default theme

breathe_projects = {"cospan": str(ROOT / "build" / "docs" / "doxygen" / "xml")}
breathe_default_project = "cospan"
