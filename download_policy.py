#!/usr/bin/env python3
"""Download an mjlab-exported policy JSON from a W&B run and validate it.

mjlab logs training checkpoints as .pt files, which the Pi cannot convert (that
needs a GPU and MuJoCo-Warp). The conversion happens on the training machine, and
Pupper runs now do it automatically: when a run ends -- on a normal finish or on
Ctrl+C -- policy.json is uploaded to the W&B run's Files. This script pulls it back
down. A run that was killed outright (SIGKILL, preemption) leaves no JSON and needs
a manual export against its last checkpoint.

To convert an older checkpoint by hand instead:

    uv run export-pupper-policy Mjlab-Trot-Bumpy-Pupper-v3 \
        --wandb-run-path mjlab/pdfzwf3l --upload-wandb

    python3 download_policy.py mjlab/pdfzwf3l        # entity/project/run-id
    python3 download_policy.py pdfzwf3l --project mjlab
    python3 download_policy.py --file policy.json --output mimic_policy.json

Validation is not cosmetic: a frame-size or joint-order mismatch between the
policy and the controller config is silent on the robot until the legs move.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Joint order the controller config uses; the policy's observation and action
# layout is defined in this order, so a mismatch is unrecoverable.
EXPECTED_JOINT_NAMES = [
  "leg_front_r_1", "leg_front_r_2", "leg_front_r_3",
  "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
  "leg_back_r_1", "leg_back_r_2", "leg_back_r_3",
  "leg_back_l_1", "leg_back_l_2", "leg_back_l_3",
]  # fmt: skip

BASE_OBSERVATION_SIZE = 36
ACTION_SIZE = 12


def resolve_run_path(run: str, project: str | None, entity: str | None) -> str:
  """Accept 'entity/project/run', 'project/run', or a bare run id."""
  import wandb

  parts = run.split("/")
  if len(parts) == 3:
    return run
  api = wandb.Api()
  if len(parts) == 2:
    project, run_id = parts
  else:
    run_id = parts[0]
    if project is None:
      raise SystemExit("Pass --project (or a full entity/project/run path).")
  entity = entity or api.default_entity
  if entity is None:
    raise SystemExit("Could not determine the W&B entity. Run 'wandb login'.")
  return f"{entity}/{project}/{run_id}"


# W&B writes its own JSON into every run's files. Downloading one of those instead
# of the policy produced a confusing KeyError deep in validation, so they are
# excluded from the fallback search by name.
WANDB_INTERNAL_PREFIXES = ("wandb-", "wandb/", "media/", "config.yaml", "requirements")


def _is_candidate_policy(name: str) -> bool:
  if not name.endswith(".json"):
    return False
  return not any(name.startswith(p) for p in WANDB_INTERNAL_PREFIXES)


def fetch(run_path: str, file_name: str, dest_dir: Path) -> Path:
  """Download ``file_name`` from the run's files, falling back to artifacts."""
  import wandb

  api = wandb.Api()
  run = api.run(run_path)
  print(f"Run: {run.name} ({run_path})")

  names = [f.name for f in run.files()]
  if file_name in names:
    candidates = [file_name]
  else:
    candidates = sorted(n for n in names if _is_candidate_policy(n))
    if candidates:
      print(f"'{file_name}' not found; falling back to: {candidates[0]}")
  if candidates:
    name = candidates[0]
    print(f"Downloading run file: {name}")
    run.file(name).download(root=str(dest_dir), replace=True)
    return dest_dir / name

  for art in run.logged_artifacts():
    if ".json" in art.name:
      print(f"Downloading artifact: {art.name}")
      root = Path(art.download(root=str(dest_dir)))
      jsons = sorted(root.glob("*.json"))
      if jsons:
        return jsons[0]

  raise SystemExit(
    f"No policy JSON in {run_path} (it has {len(names)} files, none of them a "
    f"policy).\n\n"
    "mjlab uploads policy.json when a run ends -- on a normal finish or on "
    "Ctrl+C. A run that is still training, or that was killed outright "
    "(SIGKILL, cluster preemption, a crash), never gets that far.\n\n"
    "Export it by hand from the training machine:\n"
    f"  uv run export-pupper-policy <TASK-ID> --wandb-run-path {run_path} --upload-wandb"
  )


def validate(path: Path) -> dict:
  """Check the deploy contract, and report what the controller will do with it."""
  policy = json.loads(path.read_text())

  # Fail with something readable if this is not a policy at all. Downloading the
  # wrong file used to surface as a KeyError from the middle of validation, which
  # says nothing about what actually went wrong.
  missing = [k for k in ("in_shape", "layers") if k not in policy]
  if missing:
    keys = ", ".join(sorted(policy)[:8]) or "(none)"
    raise SystemExit(
      f"{path.name} is not an mjlab policy export -- missing {', '.join(missing)}.\n"
      f"Its top-level keys are: {keys}\n\n"
      "If this is 'wandb-metadata.json' or similar, the run had no policy.json "
      "and an unrelated file was picked up. Export the policy from the training "
      "machine (see this script's docstring), or pass --file to name the right one."
    )

  history = policy.get("observation_history")
  frame = policy.get("single_observation_size", BASE_OBSERVATION_SIZE)
  in_dim = policy["in_shape"][1]
  gait = policy.get("gait_reference")
  jump = policy.get("jump_reference")
  slot = policy.get("jump_slot")

  problems = []
  if history is None:
    problems.append("missing observation_history")
  elif history * frame != in_dim:
    problems.append(
      f"observation_history ({history}) * single_observation_size ({frame}) != in_shape ({in_dim})"
    )
  for key in ("default_joint_pos", "joint_lower_limits", "joint_upper_limits"):
    if key not in policy:
      problems.append(f"missing {key}")
    elif len(policy[key]) != ACTION_SIZE:
      problems.append(f"{key} has {len(policy[key])} entries, expected {ACTION_SIZE}")
  if policy["layers"][-1]["shape"][1] != ACTION_SIZE:
    problems.append(f"policy outputs {policy['layers'][-1]['shape'][1]} actions, expected {ACTION_SIZE}")

  if gait is not None and jump is not None:
    problems.append("both gait_reference and jump_reference present; only one is allowed")
  if gait is None and jump is None:
    if frame != BASE_OBSERVATION_SIZE:
      problems.append(
        f"single_observation_size is {frame} but there is no gait_reference or "
        "jump_reference block; the controller cannot fill the extra dimensions"
      )
  elif jump is not None:
    if frame != BASE_OBSERVATION_SIZE + ACTION_SIZE:
      problems.append(
        f"jump_reference present but single_observation_size is {frame}, "
        f"expected {BASE_OBSERVATION_SIZE + ACTION_SIZE}"
      )
    n = jump.get("n_samples", 0)
    rows = jump.get("jump_table", [])
    if len(rows) != n or any(len(r) != ACTION_SIZE for r in rows):
      problems.append(f"jump_table is not {n}x{ACTION_SIZE}")
    for key in ("frequency", "crouch_hold_s", "phase_start"):
      if key not in jump:
        problems.append(f"jump_reference missing {key}")
    if jump.get("joint_names") and jump["joint_names"] != EXPECTED_JOINT_NAMES:
      problems.append("jump_reference joint_names do not match the controller config order")
  else:
    if frame != BASE_OBSERVATION_SIZE + ACTION_SIZE:
      problems.append(
        f"gait_reference present but single_observation_size is {frame}, "
        f"expected {BASE_OBSERVATION_SIZE + ACTION_SIZE}"
      )
    n = gait.get("n_samples", 0)
    # lift_table is optional: mixed-gait policies ship it (lift-in-place while
    # turning); older trot/gallop policies don't and keep trotting there.
    tables = ["trot_table", "gallop_table"]
    for optional in ("lift_table", "gallop_back_table"):
      if optional in gait:
        tables.append(optional)
    for table in tables:
      rows = gait.get(table, [])
      if len(rows) != n or any(len(r) != ACTION_SIZE for r in rows):
        problems.append(f"{table} is not {n}x{ACTION_SIZE}")
    if gait.get("joint_names") and gait["joint_names"] != EXPECTED_JOINT_NAMES:
      problems.append("gait_reference joint_names do not match the controller config order")

  # The game-mode jump slot rides on the gait reference; it never stands alone.
  if slot is not None:
    if gait is None:
      problems.append("jump_slot present without a gait_reference block")
    n = slot.get("n_samples", 0)
    rows = slot.get("jump_table", [])
    if len(rows) != n or any(len(r) != ACTION_SIZE for r in rows):
      problems.append(f"jump_slot table is not {n}x{ACTION_SIZE}")
    for key in ("playback_s", "active_s", "cross_fade_s", "grid_s", "busy_s"):
      if key not in slot:
        problems.append(f"jump_slot missing {key}")

  print()
  if slot:
    kind = "(mixed gaits + jump slot / game mode)"
  elif gait:
    kind = "(mimic / motion reference)"
  elif jump:
    kind = "(one-shot jump)"
  else:
    kind = "(plain proprio)"
  print(f"  frame size:          {frame} {kind}")
  print(f"  observation history: {history}  -> input {in_dim}")
  print(f"  kp / kd:             {policy.get('kp')} / {policy.get('kd')}")
  print(f"  action scale:        {policy.get('action_scale')}")
  if jump:
    print(
      f"  jump table:          {jump['n_samples']} phase samples, {jump['frequency']:.3f} Hz "
      f"once triggered, {jump['crouch_hold_s']:.2f} s crouch hold"
    )
    print(
      f"  trigger:             joy button {jump.get('trigger_button', 7)} (R2); the robot "
      "holds the crouch until pressed"
    )
  if gait:
    print(f"  gait tables:         {gait['n_samples']} phase samples, {gait['frequency']:.3f} Hz")
    if "lift_table" in gait:
      split = "direction-split fast tables" if "gallop_back_table" in gait else "one fast table (time-reversed backward)"
      print(
        f"  mixed gaits:         trot -> fast above |vx| = {gait['gallop_speed']}, "
        f"lift-in-place when turning, {split}"
      )
    else:
      print(
        f"  gallop:              {gait['gallop_freq_mult']}x cadence above |vx| = {gait['gallop_speed']}"
      )
    print(f"  blend speed:         {gait['blend_speed']}")
  if slot:
    print(
      f"  jump slot:           {slot['n_samples']} samples over {slot['playback_s']:.2f} s "
      f"({slot['active_s']:.2f} s active), grid {slot['grid_s']:.2f} s, busy {slot['busy_s']:.2f} s"
    )
    print(
      f"  game buttons:        jump on {slot.get('trigger_button', 0)} (X), run on "
      f"{slot.get('run_button', 1)} (circle); caps {slot.get('walk_speed_cap', 0.49)} / "
      f"{slot.get('run_speed_cap', 1.5)} m/s"
    )

  if problems:
    print("\nPolicy FAILED validation:")
    for p in problems:
      print(f"  - {p}")
    raise SystemExit(1)
  print("\nPolicy looks consistent with the controller config.")
  return policy


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("run", nargs="?", help="W&B run: entity/project/run-id, project/run-id, or run-id")
  parser.add_argument("--project", default="mjlab")
  parser.add_argument("--entity", default=None)
  parser.add_argument("--file", default="policy.json", help="Run file to download")
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Destination path. Default: mimic_policy.json, or jump_policy.json when "
    "the downloaded policy carries a jump_reference block -- so downloading a "
    "jump policy cannot silently overwrite the mixed-gaits policy in the L1 slot.",
  )
  parser.add_argument("--validate-only", type=Path, default=None, help="Validate a local JSON and exit")
  args = parser.parse_args()

  if args.validate_only is not None:
    validate(args.validate_only)
    return 0
  if args.run is None:
    parser.error("a run is required (or use --validate-only)")

  run_path = resolve_run_path(args.run, args.project, args.entity)
  downloads = REPO / ".downloads"
  downloads.mkdir(exist_ok=True)
  downloaded = fetch(run_path, args.file, downloads)
  policy = validate(downloaded)

  output = args.output
  if output is None:
    # Route by kind so each policy lands in its own controller's file rather
    # than clobbering another slot's.
    if "jump_slot" in policy:
      output, kind = REPO / "mixed_jump_policy.json", "mixed gaits + jump (game mode)"
    elif "jump_reference" in policy:
      output, kind = REPO / "jump_policy.json", "jump"
    else:
      output, kind = REPO / "mimic_policy.json", "gait/velocity"
    print(f"\nPolicy kind: {kind} -> {output.name}")

  shutil.copy2(downloaded, output)
  print(f"\nWrote {output}")
  print("Next: python3 install.py")
  return 0


if __name__ == "__main__":
  sys.exit(main())
