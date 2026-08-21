# Pupper mimic-policy deploy

Deploys mjlab's motion-reference ("mimic") Pupper policies — the trot/gallop gait
policy — onto Pupper v3, in the same style as `lab_5_fall_2025`: this repo holds
the files, `install.py` copies them into `pupperv3-monorepo` and rebuilds, and
`deploy.sh` does the whole thing including launch.

```bash
# on the robot
git clone <this repo> ~/pupper_gait_deploy && cd ~/pupper_gait_deploy
./deploy.sh mjlab/<run-id>   # download that run's policy, install, rebuild, launch
# then press L1 on the joystick to activate the mimic controller
```

## Loading a policy from a training run

Pupper training uploads the deploy JSON to its W&B run's **Files**, under the
stable name `policy.json`, when the run ends — both on a normal finish and when
you stop it with Ctrl+C. There is no export step to run by hand and no iteration
number to pick, so pointing the robot at a run id is the whole workflow:

```bash
./deploy.sh mjlab/abc123xy          # download + install + rebuild + launch
python3 download_policy.py mjlab/abc123xy   # download and validate only
```

The run id is the last path component of the W&B run URL. `download_policy.py`
accepts `entity/project/run-id`, `project/run-id`, or a bare `run-id` with
`--project`, and writes `mimic_policy.json` in this repo.

It validates before anything is installed, because every one of these is silent on
hardware right up until the legs move:

- `observation_history × single_observation_size` equals the policy's input shape
- 12 actions out, and `default_joint_pos` / joint limits are 12 long
- `gait_reference` joint names match the controller config's joint order
- the reference tables are `n_samples × 12`
- a 48-dim frame has a `gait_reference` block, and a 36-dim frame does not

It also prints what the controller will do with the policy — frame size and whether
it is a mimic or plain-proprio policy, kp/kd, action scale, and for mimic policies
the gait cadence, gallop threshold, and blend speed. Worth a glance: the gallop
threshold in particular tells you which policy you actually have. A trot-only
policy ships a very large `gallop_speed` so the branch never fires, whereas a
trot/gallop policy shows the real switch speed.

If a run was killed outright rather than interrupted — `SIGKILL`, a node failure,
a cluster preemption — no JSON was written, because those do not reach the training
loop the way Ctrl+C does. Export that run's last checkpoint by hand from the
training machine:

```bash
uv run export-pupper-policy Mjlab-TrotGallop-Flat-Pupper-v3 --wandb-run-path mjlab/<run-id> --upload-wandb
```

Retraining does not require re-running `install.py` unless the controller sources
changed — a new policy is just a new JSON:

```bash
python3 download_policy.py mjlab/<new-run-id>   # refresh mimic_policy.json
python3 install.py                              # copy into the monorepo + rebuild
```

To pin an older checkpoint instead of the latest, convert it on the training
machine and upload under a different name, then pass `--file`:

```bash
# training machine
uv run export-pupper-policy Mjlab-TrotGallop-Flat-Pupper-v3 --wandb-run-path mjlab/<run-id> --output policy_iter9000.json --upload-wandb
# robot
python3 download_policy.py mjlab/<run-id> --file policy_iter9000.json
```

## Why the controller needed changing

The walking policies observe a 36-dim proprioceptive frame:

```
[ang_vel(3), projected_gravity(3), cmd(3), desired_world_z(3), joint_pos - default(12), last_action(12)]
```

A mimic policy also observes **the reference motion it is tracking**: the 12 joint
offsets from the default pose that it should currently be at. So its frame is
48-dim, and — this is the part that cannot come from the checkpoint — the *robot*
has to produce those last 12 numbers itself, every control step, in exactly the
way training produced them.

`neural_controller` previously had the frame size baked in as a compile-time 36 and
rejected any policy whose input shape was not a multiple of it, so the gait policy
could not even load. The changes here:

- **`src/gait_reference.hpp`** (new) — the reference lookup: phase clock, gallop
  switch, time reversal for backward/turning commands, and the speed blend toward
  the default standing pose. No IK on the robot; the phase → joint-angle tables are
  precomputed by mjlab and shipped inside the policy JSON. Fast gaits are
  direction-split: a policy can ship an optional `gallop_back_table` (pre-reversed
  by the exporter) that plays for backward-fast commands instead of the
  time-reversed forward table; policies without it keep the old behavior.
- **`src/neural_controller.{hpp,cpp}`** — frame size is now read from the policy's
  `single_observation_size` (absent ⇒ 36, so **every existing policy keeps loading
  unchanged**), the `gait_reference` block is parsed at load, and the reference is
  written into the newest frame before each inference.

  The controller also runs an **IMU heading hold**, mirroring mjlab's
  command-side loop (the same correction the G1 deploy stack uses): while
  walking with a quiet commanded yaw (|yaw| < 0.1 rad/s, linear command norm >
  0.05 m/s), the yaw command fed to the policy and the gait reference is
  replaced by `clip(kp * heading_error, ±0.3)` toward the IMU heading captured
  when the yaw went quiet. A commanded turn or a stop disengages and re-arms
  it. Default **on** for gait policies (a frozen yaw-rate-tracking policy
  needs no retraining); a `heading_hold` block in the policy JSON — stamped by
  mjlab's exporter for runs that trained with the closed loop — overrides the
  parameters, and `"kp": 0` disables. Plain 36-dim policies are untouched
  unless their JSON asks for it.
- **`config.yaml` / `launch.py`** — register a separate `neural_controller_mimic`
  instance of the same plugin, on joystick button L1. The four existing modes
  (walk / three-legged / parkour / test) are untouched. Both files start from lab
  5's versions, so installing this repo also keeps you on that config; the only
  additions are the mimic controller and an explicit `estop_controller` button
  mapping.

The controller refuses to start on a mismatch: a `gait_reference` block with the
wrong frame size, or an enlarged frame with no reference block, is an error rather
than a silently mis-fed policy.

## Exporting a policy (training machine, not the robot)

The Pi cannot convert an mjlab `.pt` checkpoint — that needs a GPU and MuJoCo-Warp.
The conversion happens on the training machine and is handed off through the W&B run.

**Pupper training does this automatically.** Every checkpoint save also writes and
uploads `policy.json` to the run, overwriting the previous one, so the run always
carries a deployable copy of the latest policy — including if the job is preempted
or killed part way. Nothing needs to be run by hand; go straight to the robot
section below.

The export folds the observation normalization into the first layer, reverses the
history frames into the robot's newest-first order, emits the gait reference tables
for mimic policies, and parity-checks the result against the live actor on real
observations. A failed parity check skips the upload rather than shipping a bad
policy, and never interrupts training.

To convert a specific older checkpoint, or to regenerate the parity fixture, run it
by hand:

```bash
uv run export-pupper-policy Mjlab-Trot-Bumpy-Pupper-v3 --wandb-run-path mjlab/pdfzwf3l --upload-wandb
```

Add `--golden-output test/gait_golden.json` to regenerate the parity fixture, which
you must do whenever the reference tables or the gait parameters change.

On the robot:

```bash
./deploy.sh mjlab/tryieau1          # download that run's policy, install, rebuild, launch
python3 download_policy.py mjlab/tryieau1   # download only
python3 download_policy.py --validate-only mimic_policy.json
```

`download_policy.py` validates the deploy contract before anything is installed —
frame size vs. input shape, joint ordering, table dimensions, action count — because
each of those is silent on hardware right up until the legs move.

`mimic_policy.json` in this repo is the currently deployed policy (mjlab run
`tryieau1` at `model_6500`, the captured-reference MixedGaits trot/gallop,
trained across the full ±1.5 m/s command band), so a plain
`git pull && ./deploy.sh` reproduces a known-good robot.

## Layout

```
config.yaml                   -> neural_controller/launch/config.yaml
launch.py                     -> neural_controller/launch/launch.py
mimic_policy.json             -> neural_controller/launch/mimic_policy.json
src/gait_reference.hpp        -> neural_controller/include/neural_controller/
src/gait_reference_json.hpp   -> neural_controller/include/neural_controller/
src/neural_controller.hpp     -> neural_controller/include/neural_controller/
src/neural_controller.cpp     -> neural_controller/src/
src/CMakeLists.txt            -> neural_controller/CMakeLists.txt
test/test_gait_reference.cpp  -> neural_controller/test/
test/gait_golden.json         -> neural_controller/test/
```

`install.py --dry-run` prints the mapping without touching anything; every
overwritten file is backed up next to itself with a `.backup` suffix.

## Jump policies (one-shot reference, R2 to hop)

A policy exported from `Mjlab-Jump-Flat-Pupper-v3` carries a `jump_reference`
block instead of `gait_reference`: the same 48-dim frame, but the reference slot
plays a *one-shot* table driven by seconds-since-trigger rather than a
command-blended cycle. It gets its own controller instance,
`neural_controller_jump`, loaded from `launch/jump_policy.json` — the mixed
gaits stay in `neural_controller_mimic` on L1, untouched.

**R2 switches to the jump controller and performs a single jump**: the
`estop_controller` switch list maps button 7 (R2 under this pad's
0=x, 1=o, 2=△, 3=□, 4=L1 mapping) to `neural_controller_jump`, which folds to
the crouch through its init/fade (1.5 s + 1.0 s, tighter than the walking
modes), auto-triggers one jump the moment the fade-in completes (the
activation press was the request), and lands back into the crouch. Each
further R2 press while it is active hops again — rising-edge, ignored until
the previous cycle plus a 0.3 s landing margin has passed, so mashing R2
mid-air does nothing. **L1 returns to the mixed gaits.** The jump policy
trained on an all-zero command, so keep the sticks quiet while it is active.

`jump_policy.json` (run `zuy6c85c`, 0.55 m apex in sim) ships in this repo like
`mimic_policy.json` does, so deploying is just:

```bash
# robot
git pull
python3 install.py && cd ~/pupperv3-monorepo/ros2_ws && colcon build --packages-select neural_controller
```

To ship a newer jump run, re-export on the training machine and replace the
file (or use download_policy.py, which routes a jump policy to
jump_policy.json by its `jump_reference` block, so it cannot clobber the
mixed-gaits policy in the L1 slot):

```bash
uv run export-pupper-policy Mjlab-Jump-Flat-Pupper-v3 \
    --wandb-run-path <entity/project/run> --output jump_policy.json \
    --golden-output jump_golden.json
```

If a jump policy was previously downloaded over `mimic_policy.json` (the old
default output), restore the L1 slot with `git checkout -- mimic_policy.json`
or by re-downloading the mixed-gaits run.

Mind the estop: the trained jump pitches ~20-27 deg nose-up in flight, so
`max_body_angle` must stay comfortably above that or the estop fires at the
apex. The shipped `config.yaml` (1.5 rad ~ 86 deg) is fine; the sim policy was
trained with a 45 deg cutoff, so anything in between also works.

## Game mode (MixedGaitsJump, L1+R1)

`mixed_jump_policy.json` (run `kmhi67tp`, the rough-terrain/DR finetune of the
mixed-gaits-plus-jump task) carries the usual `gait_reference` block **plus a
`jump_slot` block**: a captured jump table the controller can splice into the
mixed reference on a button press, with the velocity command carried straight
through the jump. It runs in its own controller instance,
`neural_controller_mixed_jump`.

The controls, "the true game way":

- **L1+R1 (together)** on the default walk controller switches into game mode;
  **L1+R1 again** switches back out to the walk policy. The chord is detected
  by the active controller itself (`chord_partner` in `config.yaml`), which
  asks the `controller_manager` to swap — not by the `estop_controller`.
- **Left stick** walks. The command is capped at the JSON's `walk_speed_cap`
  (±0.49 m/s), just under the 0.5 m/s fast-gait onset, so plain stick input
  can only walk.
- **Hold circle** to run: the cap rises to `run_speed_cap` (±1.5 m/s) and the
  fast gaits become reachable. Release and the cap drops back.
- **X jumps.** A rising edge schedules the jump slot at the next 0.75 s gait
  grid point (the same `request_jump` arithmetic as training, on the fade-in
  clock); pressing X again while one is pending or in flight queues the next
  jump after the busy window (1.0 s) instead of restarting it. The jump keeps
  whatever velocity is commanded — run in with circle held and it jumps at
  speed.

Because game mode claims x, circle and L1, those buttons left the
`estop_controller` single-button switch list. What remains: **triangle =
parkour, square = test, L2 = default walk (initial bring-up), R2 = one-shot
jump policy**. After boot everything spawns inactive, so the flow is: press
**L2** to bring in the walk controller, then **L1+R1** to enter game mode. The
old three-legged (o) and mimic (L1) modes have no button in this config; the
mixed-jump policy supersedes the mimic one.

The new C++ (a `JumpSlot` struct, `compute_mixed_jump_reference_offset`, the
chord service client) builds against `controller_manager_msgs`, which is
already on the robot as part of ros2_control. Deploying is the usual:

```bash
# robot
git pull
python3 install.py && cd ~/pupperv3-monorepo/ros2_ws && colcon build --packages-select neural_controller
```

`download_policy.py` routes a JSON with a `jump_slot` block to
`mixed_jump_policy.json` automatically. To re-export from a newer run:

```bash
uv run export-pupper-policy Mjlab-MixedGaitsJump-Bumpy-Pupper-v3 \
    --wandb-run-path mjlab/<run> --output mixed_jump_policy.json \
    --golden-output mixed_jump_golden.json
```

## Tests

`test/gait_golden.json` holds reference offsets generated from the trained mjlab env
itself, covering standing, trot, gallop, and the time-reversed backward and turning
cases. The C++ has to reproduce them:

```bash
./test/run_host_test.sh      # laptop, no ROS or gtest needed
./test/run_host_test.sh test/mixed_jump_golden.json mixed_jump_policy.json  # game-mode fixture
colcon test --packages-select neural_controller   # on the robot, as part of the build
```

A fixture with a `jump_slot` block additionally checks the composite
gait+jump-slot reference (`slot_cases`: in-slot, cross-fade edges, and
slot-free bit-exactness against the plain mixed reference).

This is the check that matters most. The failure it guards against — a truncating
`%` instead of a floor-modulo when the phase goes negative, which is what a
backward or turn-in-place command produces — is invisible in a code review and
produces a policy being fed an input it has never seen.

## Notes and gotchas

- **Phase clock.** The controller runs at 520 Hz / `repeat_action` 10 = 52 Hz,
  while the policy trained at 50 Hz. The gait phase is driven off wall-clock
  seconds since the policy took over, not a step count, so the foot trajectory
  keeps its physical 0.75 s cadence. It is kept in `double`: the robot stays up
  far longer than a 10 s sim episode, and a `float` visibly quantizes the phase
  after a few minutes of uptime.
- **Teleop range.** `scale_linear.x` is 1.5: full stick is the ±1.5 m/s the
  captured-reference checkpoints (run `tryieau1` onward) actually train on. The
  trot-to-fast switch comes from the policy JSON's `gallop_speed` (0.5 m/s), so
  the fast gait engages at a third of stick deflection in either direction.
- **First run.** `max_body_angle` is 1.5 rad for the mimic mode (vs 0.52 for the
  walking modes) so an early trot attempt does not trip the fall e-stop
  immediately. Tighten it once the gait is trusted.
- **Inference cost.** The first layer grows from 256×720 to 256×960, roughly a
  third more work per step. Watch `~/policy_inference_latency_seconds`; there is
  ~19 ms of budget per policy step.
