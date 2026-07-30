#!/usr/bin/env bash
# One command to put the current policy + controller on the robot and run it.
#
#   ./deploy.sh                    # pull, install, rebuild, launch
#   ./deploy.sh mjlab/pdfzwf3l     # also download that run's policy first
#   ./deploy.sh --no-launch        # stop after the rebuild
#   ./deploy.sh --no-pull          # use the working tree as-is
#
# Anything after -- is passed straight to ros2 launch, e.g.
#   ./deploy.sh -- sim:=True
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

RUN=""
DO_PULL=1
DO_LAUNCH=1
LAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-pull) DO_PULL=0; shift ;;
    --no-launch) DO_LAUNCH=0; shift ;;
    --) shift; LAUNCH_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) RUN="$1"; shift ;;
  esac
done

if [[ $DO_PULL -eq 1 ]]; then
  echo ">>> git pull"
  git pull --ff-only
fi

if [[ -n "$RUN" ]]; then
  echo ">>> downloading policy from $RUN"
  python3 download_policy.py "$RUN"
else
  echo ">>> validating the committed policy"
  python3 download_policy.py --validate-only mimic_policy.json
fi

echo ">>> installing into the monorepo and rebuilding"
python3 install.py

if [[ $DO_LAUNCH -eq 0 ]]; then
  echo ">>> skipping launch (--no-launch)"
  exit 0
fi

echo ">>> ros2 launch neural_controller launch.py ${LAUNCH_ARGS[*]:-}"
echo ">>> press L1 on the joystick to activate the mimic controller"
cd "$HOME"
exec ros2 launch neural_controller launch.py "${LAUNCH_ARGS[@]}"
