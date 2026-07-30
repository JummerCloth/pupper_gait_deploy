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

  bool valid() const {
    const std::size_t expected = static_cast<std::size_t>(n_samples) * n_joints;
    return n_samples > 0 && n_joints > 0 && frequency > 0.0 && blend_speed > 0.0 &&
           trot_table.size() == expected && gallop_table.size() == expected;
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
  const double dir_signal = (std::abs(vx) >= gait.dir_threshold) ? vx : yaw;
  const double eff_phase = (dir_signal >= 0.0) ? phase : -phase;

  const double pos = eff_phase * gait.n_samples;
  const double pos_floor = std::floor(pos);
  const int i0 = detail::floor_mod(static_cast<std::int64_t>(pos_floor), gait.n_samples);
  const int i1 = (i0 + 1) % gait.n_samples;
  const double alpha = pos - pos_floor;

  const std::vector<double> &table = galloping ? gait.gallop_table : gait.trot_table;

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

}  // namespace neural_controller
