#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace neural_controller {

/**
 * Phase-indexed gait reference for policies trained with a motion reference in the
 * observation (mjlab's Pupper "gait" tasks).
 *
 * Those policies observe, on top of the usual 36-dim proprioceptive frame, the
 * 12-dim joint offset from the default pose that they are being asked to track.
 * The reference itself is a precomputed phase -> joint-angle table (the CS 123
 * triangular foot trajectories put through IK), so the controller does no IK: it
 * looks up the table, blends by command speed, and subtracts the default pose.
 *
 * Both tables and every parameter below come from the "gait_reference" block of the
 * deploy JSON, emitted by mjlab's mjlab.tasks.pupper_gait.export. The math here is a
 * direct transcription of that module's reference_offset_numpy(), which is
 * parity-tested against the training-time implementation -- keep the three in step.
 */
struct GaitReference {
  int n_samples = 0;
  int n_joints = 0;
  // Phase cycles per second at the base (trot) cadence.
  double frequency = 0.0;
  // Command speed at which the reference reaches the full gait; below it the
  // reference blends linearly toward the static default pose.
  double blend_speed = 0.0;
  // Commanded |vx| at/above which the reference switches to the gallop table.
  double gallop_speed = 0.0;
  // Phase-clock speedup while galloping.
  double gallop_freq_mult = 1.0;
  // |vx| below which the gait direction falls back to the turn (yaw) sign.
  double dir_threshold = 0.0;
  // Row-major n_samples x n_joints joint-angle tables, already clamped to the
  // joint limits exactly as training clamped them.
  std::vector<double> trot_table;
  std::vector<double> gallop_table;
  // Optional (mixed-gait policies): played instead of trot when the command is
  // not translating (|vx| < dir_threshold), i.e. turning or sidestepping. Empty
  // for older policies, which keep trotting there.
  std::vector<double> lift_table;
  // Optional (direction-split fast gaits): played instead of gallop_table for
  // backward-fast commands. Ships pre-reversed by the exporter, so the shared
  // backward phase reversal below plays the recording forward as captured.
  // Empty for older policies, which time-reverse the single fast table.
  std::vector<double> gallop_back_table;

  bool valid() const {
    const std::size_t expected = static_cast<std::size_t>(n_samples) * n_joints;
    return n_samples > 0 && n_joints > 0 && frequency > 0.0 && blend_speed > 0.0 &&
           trot_table.size() == expected && gallop_table.size() == expected &&
           (lift_table.empty() || lift_table.size() == expected) &&
           (gallop_back_table.empty() || gallop_back_table.size() == expected);
  }
};

namespace detail {

// Fractional part keeping the sign of the argument (matches torch.frac / np.trunc),
// e.g. gait_frac(-0.3) == -0.3.
inline double gait_frac(double x) { return x - std::trunc(x); }

// Floor-modulo. C++ '%' truncates toward zero, so a plain (i % n) would map the
// negative indices produced by a time-reversed phase to the wrong end of the
// table; Python/NumPy (which the training code uses) floor-mods instead.
inline int floor_mod(std::int64_t i, int n) {
  const int m = static_cast<int>(i % n);
  return m < 0 ? m + n : m;
}

}  // namespace detail

/**
 * Compute the reference joint offset the policy expects to see.
 *
 * @param gait              Tables and parameters from the deploy JSON.
 * @param default_joint_pos Default pose, gait.n_joints entries, policy joint order.
 * @param t                 Seconds since the phase clock started. Pass a double
 *                          accumulated from wall-clock time: the robot runs for far
 *                          longer than a sim episode, and a float would quantize the
 *                          phase visibly after a few minutes of uptime.
 * @param vx, vy, yaw       The same velocity command that goes into the observation.
 * @param out               Receives gait.n_joints offsets.
 */
inline void compute_gait_reference_offset(const GaitReference &gait,
                                          const std::vector<double> &default_joint_pos,
                                          double t, double vx, double vy, double yaw,
                                          float *out) {
  const bool galloping = std::abs(vx) >= gait.gallop_speed;
  const double freq = gait.frequency * (galloping ? gait.gallop_freq_mult : 1.0);
  const double phase = detail::gait_frac(t * freq);

  // Forward/backward sets the direction; the turn sign only matters when we are
  // barely translating. A negative direction time-reverses the reference.
  const bool translating = std::abs(vx) >= gait.dir_threshold;
  const double dir_signal = translating ? vx : yaw;
  const double eff_phase = (dir_signal >= 0.0) ? phase : -phase;

  const double pos = eff_phase * gait.n_samples;
  const double pos_floor = std::floor(pos);
  const int i0 = detail::floor_mod(static_cast<std::int64_t>(pos_floor), gait.n_samples);
  const int i1 = (i0 + 1) % gait.n_samples;
  const double alpha = pos - pos_floor;

  // Mixed-gait policies lift in place instead of trotting when not translating,
  // and a direction-split fast gait plays its own backward capture rather than
  // the forward table time-reversed.
  const std::vector<double> &slow_table =
      (translating || gait.lift_table.empty()) ? gait.trot_table : gait.lift_table;
  const std::vector<double> &fast_table =
      (dir_signal >= 0.0 || gait.gallop_back_table.empty()) ? gait.gallop_table
                                                            : gait.gallop_back_table;
  const std::vector<double> &table = galloping ? fast_table : slow_table;

  // Blend toward the static default pose at low command speed (so a zero command
  // means "stand at the default pose", i.e. an all-zero offset).
  // std::min/max rather than std::clamp: RTNeural's CMake drops this package to
  // C++14 in some configurations, and this header has no reason to need 17.
  const double speed = std::sqrt(vx * vx + vy * vy + yaw * yaw);
  const double blend = std::min(std::max(speed / gait.blend_speed, 0.0), 1.0);

  for (int j = 0; j < gait.n_joints; j++) {
    const double lo = table[static_cast<std::size_t>(i0) * gait.n_joints + j];
    const double hi = table[static_cast<std::size_t>(i1) * gait.n_joints + j];
    const double reference = lo * (1.0 - alpha) + hi * alpha;
    out[j] = static_cast<float>(blend * (reference - default_joint_pos[j]));
  }
}

/**
 * Insertable jump slot for MixedGaitsJump policies (mdp/mixed_jump.py).
 *
 * Rides on top of a GaitReference: a trigger schedules a slot start on the
 * grid, and inside the active window the captured jump table plays over
 * playback_s (phase clamped at 1) with a cross_fade_s linear blend to the
 * mixed reference at each edge -- zero exactly at the boundaries, so the
 * composite is continuous by construction. The reference window deliberately
 * ends before touchdown (active_s < playback landing), stitching the running
 * gait while the robot is still descending. Transcribes mjlab's
 * mixed_jump_reference_offset_numpy(); keep the three implementations in step.
 */
struct JumpSlot {
  int n_samples = 0;
  int n_joints = 0;
  double playback_s = 0.0;
  double active_s = 0.0;
  double cross_fade_s = 0.06;
  // Trigger starts snap up to this grid (one base gait cycle).
  double grid_s = 0.75;
  // A new trigger is queued after, not inside, a window this long.
  double busy_s = 1.0;
  // Game-mode controls, tweakable in the JSON without a rebuild.
  int trigger_button = 0;   // x: jump
  int run_button = 1;       // circle, held: lift the speed cap
  double walk_speed_cap = 0.49;
  double run_speed_cap = 1.5;
  std::vector<double> jump_table;

  bool valid() const {
    return n_samples > 0 && n_joints > 0 && playback_s > 0.0 && active_s > 0.0 &&
           cross_fade_s > 0.0 && grid_s > 0.0 && busy_s >= active_s &&
           trigger_button >= 0 && run_button >= 0 &&
           jump_table.size() == static_cast<std::size_t>(n_samples) * n_joints;
  }
};

/**
 * Composite reference: the mixed gaits, overlaid with an active jump slot.
 *
 * @param slot_start Seconds (same clock as t) the pending/active slot starts
 *                   at; pass infinity for none.
 */
inline void compute_mixed_jump_reference_offset(
    const GaitReference &gait, const JumpSlot &slot,
    const std::vector<double> &default_joint_pos, double t, double vx, double vy,
    double yaw, double slot_start, float *out) {
  // Base: the command-driven mixed reference, exactly as without a slot.
  compute_gait_reference_offset(gait, default_joint_pos, t, vx, vy, yaw, out);

  const double t_in = t - slot_start;
  if (!(t_in >= 0.0 && t_in < slot.active_s)) {
    return;
  }
  const double phase = std::min(std::max(t_in / slot.playback_s, 0.0), 1.0);
  const double pos = phase * slot.n_samples;
  const double pos_floor = std::floor(pos);
  const int i0 = detail::floor_mod(static_cast<std::int64_t>(pos_floor), slot.n_samples);
  const int i1 = (i0 + 1) % slot.n_samples;
  const double alpha = pos - pos_floor;
  const double fade =
      std::min(std::max(std::min(t_in, slot.active_s - t_in) / slot.cross_fade_s, 0.0), 1.0);

  for (int j = 0; j < slot.n_joints; j++) {
    const double lo = slot.jump_table[static_cast<std::size_t>(i0) * slot.n_joints + j];
    const double hi = slot.jump_table[static_cast<std::size_t>(i1) * slot.n_joints + j];
    const double jump_off = lo * (1.0 - alpha) + hi * alpha - default_joint_pos[j];
    out[j] = static_cast<float>(fade * jump_off + (1.0 - fade) * static_cast<double>(out[j]));
  }
}

/**
 * One-shot jump reference (mjlab's Pupper "jump" task, mdp/jump.py).
 *
 * Unlike GaitReference this has no blend and no command coupling: the offset is a
 * function of seconds-since-trigger alone. Untriggered (t = 0) the clamp pins the
 * phase at phase_start -- the mid-stance crouch, the pose the policy holds -- and a
 * trigger sweeps exactly one cycle, landing one full wrap later on the same crouch,
 * where any larger t stays. All parameters come from the "jump_reference" block of
 * the deploy JSON; the math transcribes mjlab's jump_reference_offset_numpy(),
 * parity-tested against the training-time term -- keep the three in step.
 */
struct JumpReference {
  int n_samples = 0;
  int n_joints = 0;
  // Phase cycles per second once triggered (the jump gait's 2.5x clock).
  double frequency = 0.0;
  // Seconds the reference holds the crouch after the trigger before launching.
  double crouch_hold_s = 0.0;
  // Where in the cycle the one-shot playback starts and ends (mid-stance).
  double phase_start = 0.0;
  // Joy button index that arms the clock (R2 in the standard PS4/PS5 mapping).
  int trigger_button = 7;
  // Row-major n_samples x n_joints joint angles, clamped as training clamped them.
  std::vector<double> jump_table;

  bool valid() const {
    return n_samples > 0 && n_joints > 0 && frequency > 0.0 && crouch_hold_s >= 0.0 &&
           trigger_button >= 0 &&
           jump_table.size() == static_cast<std::size_t>(n_samples) * n_joints;
  }
};

/**
 * Compute the one-shot jump reference offset the policy expects to see.
 *
 * @param jump              Table and clock from the deploy JSON.
 * @param default_joint_pos Default pose, jump.n_joints entries, policy joint order.
 * @param t                 Seconds since the jump was triggered; pass 0.0 while
 *                          untriggered (the idle crouch hold).
 * @param out               Receives jump.n_joints offsets.
 */
inline void compute_jump_reference_offset(const JumpReference &jump,
                                          const std::vector<double> &default_joint_pos,
                                          double t, float *out) {
  // min/max rather than std::clamp for the same C++14 reason as above.
  const double swept = std::min(std::max((t - jump.crouch_hold_s) * jump.frequency, 0.0), 1.0);
  const double phase = jump.phase_start + swept;

  const double pos = phase * jump.n_samples;
  const double pos_floor = std::floor(pos);
  const int i0 = detail::floor_mod(static_cast<std::int64_t>(pos_floor), jump.n_samples);
  const int i1 = (i0 + 1) % jump.n_samples;
  const double alpha = pos - pos_floor;

  for (int j = 0; j < jump.n_joints; j++) {
    const double lo = jump.jump_table[static_cast<std::size_t>(i0) * jump.n_joints + j];
    const double hi = jump.jump_table[static_cast<std::size_t>(i1) * jump.n_joints + j];
    const double reference = lo * (1.0 - alpha) + hi * alpha;
    out[j] = static_cast<float>(reference - default_joint_pos[j]);
  }
}

}  // namespace neural_controller
