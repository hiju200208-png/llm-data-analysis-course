"""Allow notebooks opened from this folder to import the project's ``src`` package."""

from pathlib import Path

# Keep implementation modules in the project's top-level src directory.
__path__.append(str(Path(__file__).resolve().parents[2] / "src"))
