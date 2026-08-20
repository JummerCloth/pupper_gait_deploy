#!/usr/bin/env python3
"""Install this repo's controller changes into the pupperv3-monorepo and rebuild.

Same idea as lab 5's rebuild_neural_controller.py, but it also ships the C++ that
teaches the neural controller about motion-reference ("mimic") policies -- the
observation frame size and the gait reference tables now come out of the policy
JSON, so older 36-dim policies keep loading unchanged.

    python3 install.py --dry-run     # show what would happen
    python3 install.py               # install + rebuild
    python3 install.py --no-build    # install only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_MONOREPO = Path("/home/pi/pupperv3-monorepo")


def file_mappings(monorepo: Path) -> list[tuple[Path, Path, str]]:
  """(source, destination, description), in install order."""
  pkg = monorepo / "ros2_ws" / "src" / "neural_controller"
  launch = pkg / "launch"
  return [
    (
      REPO / "src" / "gait_reference.hpp",
      pkg / "include" / "neural_controller" / "gait_reference.hpp",
      "Gait reference lookup (new header)",
    ),
    (
      REPO / "src" / "gait_reference_json.hpp",
      pkg / "include" / "neural_controller" / "gait_reference_json.hpp",
      "Gait reference JSON parser (new header, shared with the test)",
    ),
    (
      REPO / "src" / "neural_controller.hpp",
      pkg / "include" / "neural_controller" / "neural_controller.hpp",
      "Neural controller header (runtime frame size + gait state)",
    ),
    (
      REPO / "src" / "neural_controller.cpp",
      pkg / "src" / "neural_controller.cpp",
      "Neural controller source (parses gait_reference, fills the frame)",
    ),
    (
      REPO / "src" / "neural_controller_parameters.yaml",
      pkg / "src" / "neural_controller_parameters.yaml",
      "Controller parameter declarations (adds kp_scale)",
    ),
    (
      REPO / "src" / "CMakeLists.txt",
      pkg / "CMakeLists.txt",
      "Package CMakeLists (adds the gait parity test)",
    ),
    (
      REPO / "test" / "test_gait_reference.cpp",
      pkg / "test" / "test_gait_reference.cpp",
      "Gait reference parity test",
    ),
    (
      REPO / "test" / "gait_golden.json",
      pkg / "test" / "gait_golden.json",
      "Gait parity golden data (generated from the trained env)",
    ),
    (
      REPO / "test" / "jump_golden.json",
      pkg / "test" / "jump_golden.json",
      "Jump parity golden data (generated from the trained env)",
    ),
    (
      REPO / "config.yaml",
      launch / "config.yaml",
      "Controller config (adds neural_controller_mimic)",
    ),
    (
      REPO / "launch.py",
      launch / "launch.py",
      "Launch file (spawns neural_controller_mimic)",
    ),
    (
      REPO / "mimic_policy.json",
      launch / "mimic_policy.json",
      "Motion-reference policy loaded by neural_controller_mimic",
    ),
    (
      REPO / "jump_policy.json",
      launch / "jump_policy.json",
      "One-shot jump policy loaded by neural_controller_jump (R2)",
    ),
  ]


def install(monorepo: Path, dry_run: bool) -> bool:
  print("=" * 70)
  print("Installing mimic-policy controller support")
  print("=" * 70)
  if dry_run:
    print("\nDRY RUN - nothing will be written\n")

  errors = 0
  for source, destination, description in file_mappings(monorepo):
    print(f"\n{description}")
    print(f"  source:      {source}")
    print(f"  destination: {destination}")

    if not source.exists():
      if source.name in ("mimic_policy.json", "jump_policy.json"):
        print("  SKIPPED: no policy here yet -- run download_policy.py first")
        continue
      print("  ERROR: source file does not exist")
      errors += 1
      continue

    if dry_run:
      print("  would replace existing file" if destination.exists() else "  would create file")
      continue

    try:
      destination.parent.mkdir(parents=True, exist_ok=True)
      if destination.exists():
        backup = destination.with_suffix(destination.suffix + ".backup")
        shutil.copy2(destination, backup)
        print(f"  backup: {backup.name}")
      shutil.copy2(source, destination)
      print("  installed")
    except OSError as e:
      print(f"  ERROR: {e}")
      errors += 1

  print(f"\n{'=' * 70}\nErrors: {errors}\n{'=' * 70}")
  return errors == 0


def rebuild(monorepo: Path, dry_run: bool) -> bool:
  ros2_ws = monorepo / "ros2_ws"
  build_script = ros2_ws / "build.sh"
  print("\n" + "=" * 70)
  print("Rebuilding the ROS 2 workspace")
  print("=" * 70)

  if not build_script.exists():
    print(f"ERROR: build script not found: {build_script}")
    return False
  if dry_run:
    print(f"would run: bash {build_script} (in {ros2_ws})")
    return True

  result = subprocess.run(["bash", str(build_script)], cwd=str(ros2_ws), check=False)
  if result.returncode != 0:
    print(f"\nBuild FAILED (exit code {result.returncode})")
    return False
  print("\nBuild succeeded.")
  return True


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--monorepo", type=Path, default=DEFAULT_MONOREPO)
  parser.add_argument("--dry-run", "-n", action="store_true")
  parser.add_argument("--no-build", action="store_true")
  args = parser.parse_args()

  if not args.monorepo.exists():
    print(f"ERROR: monorepo not found at {args.monorepo} (pass --monorepo)")
    return 1

  if not install(args.monorepo, args.dry_run):
    print("\nInstall failed. Not rebuilding.")
    return 1

  if args.no_build:
    print("\nSkipping rebuild (--no-build).")
    return 0
  if not rebuild(args.monorepo, args.dry_run):
    return 1

  print("\nDone. Launch with: ros2 launch neural_controller launch.py")
  return 0


if __name__ == "__main__":
  sys.exit(main())
