// Cross-language parity test for the on-robot gait reference.
//
// The reference joint offset the controller feeds the policy has to match, to
// within float noise, the one mjlab computed during training -- a phase-wrap or
// modulo slip here shows up on hardware as a policy that has never seen its own
// input. gait_golden.json holds tables and expected offsets generated straight
// from the trained env by mjlab's `export-pupper-policy --golden-output`, covering
// standing, trot, gallop, and the time-reversed backward / turning cases.
//
// argv[1] is the golden fixture; argv[2] is optionally a deploy policy.json, whose
// gait_reference block is then parsed through the same parser the controller uses
// and checked against the fixture's tables.
//
// Deliberately standalone (no gtest): it runs under ctest in the ROS build and
// under test/run_host_test.sh with nothing but a compiler.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"
#include "neural_controller/gait_reference_json.hpp"

namespace {

constexpr double kTolerance = 1e-4;
constexpr int kActionSize = 12;

nlohmann::json read_json(const std::string &path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("could not open " + path);
  }
  nlohmann::json j;
  stream >> j;
  return j;
}

// Parity path for the one-shot jump reference (fixture carries "jump_reference"
// instead of "gait_reference"; cases carry only t -- no command coupling).
int run_jump_test(const nlohmann::json &j, const std::string &policy_path) {
  neural_controller::JumpReference jump;
  try {
    neural_controller::parse_jump_reference(j.at("jump_reference"), kActionSize, jump);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  if (!policy_path.empty()) {
    std::ifstream probe(policy_path);
    const bool present = static_cast<bool>(probe);
    probe.close();
    if (!present) {
      std::printf("No policy at %s -- skipping the deployed-policy check\n",
                  policy_path.c_str());
    } else {
      try {
        const nlohmann::json policy = read_json(policy_path);
        neural_controller::JumpReference from_policy;
        neural_controller::parse_jump_reference(policy.at("jump_reference"), kActionSize,
                                                from_policy);
        const int frame = policy.value("single_observation_size", 36);
        if (frame != 36 + kActionSize) {
          std::fprintf(stderr, "FAIL: %s has jump_reference but frame size %d\n",
                       policy_path.c_str(), frame);
          return 1;
        }
        if (from_policy.jump_table != jump.jump_table) {
          std::fprintf(stderr, "FAIL: %s jump table differs from the golden table\n",
                       policy_path.c_str());
          return 1;
        }
        std::printf("Deploy policy %s: jump_reference parsed, table matches the golden data\n",
                    policy_path.c_str());
      } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: parsing %s: %s\n", policy_path.c_str(), e.what());
        return 1;
      }
    }
  }

  const std::vector<double> default_joint_pos =
      j.at("default_joint_pos").get<std::vector<double>>();
  if (static_cast<int>(default_joint_pos.size()) != jump.n_joints) {
    std::fprintf(stderr, "FAIL: default_joint_pos has %zu entries, expected %d\n",
                 default_joint_pos.size(), jump.n_joints);
    return 1;
  }

  int failures = 0;
  double worst = 0.0;
  int n_cases = 0;
  std::vector<float> out(jump.n_joints, 0.0f);
  for (const auto &c : j.at("cases")) {
    const double t = c.at("t");
    const std::vector<double> expected = c.at("expected").get<std::vector<double>>();
    neural_controller::compute_jump_reference_offset(jump, default_joint_pos, t, out.data());
    double max_err = 0.0;
    for (int i = 0; i < jump.n_joints; i++) {
      max_err = std::max(max_err, std::abs(static_cast<double>(out[i]) - expected[i]));
    }
    worst = std::max(worst, max_err);
    n_cases++;
    if (max_err > kTolerance) {
      std::fprintf(stderr, "FAIL case t=%.3f: max_abs_err=%.3e\n", t, max_err);
      failures++;
    }
  }

  // The idle hold (t=0) must be a crouch, not the default pose: the policy was
  // trained holding this reference, and an all-zero offset would be a different
  // (untrained) input.
  std::vector<float> idle(jump.n_joints);
  neural_controller::compute_jump_reference_offset(jump, default_joint_pos, 0.0, idle.data());
  bool nonzero = false;
  for (int i = 0; i < jump.n_joints; i++) {
    nonzero = nonzero || std::abs(idle[i]) > 1e-3f;
  }
  if (!nonzero) {
    std::fprintf(stderr, "FAIL: idle jump reference is the default pose, not the crouch\n");
    failures++;
  }

  // One full wrap: any time past the cycle end must reproduce the idle crouch
  // exactly (the landing pose IS the launch pose), forever.
  std::vector<float> landed(jump.n_joints), late(jump.n_joints);
  const double cycle_end = jump.crouch_hold_s + 1.0 / jump.frequency;
  neural_controller::compute_jump_reference_offset(jump, default_joint_pos, cycle_end + 1.0,
                                                   landed.data());
  neural_controller::compute_jump_reference_offset(jump, default_joint_pos, 3600.0, late.data());
  for (int i = 0; i < jump.n_joints; i++) {
    if (std::abs(landed[i] - idle[i]) > 1e-5f || std::abs(late[i] - idle[i]) > 1e-5f) {
      std::fprintf(stderr, "FAIL: landing hold differs from the idle crouch at joint %d\n", i);
      failures++;
      break;
    }
  }

  // Mid-cycle must differ from the hold -- a stuck clock would pin the crouch.
  std::vector<float> mid(jump.n_joints);
  neural_controller::compute_jump_reference_offset(
      jump, default_joint_pos, jump.crouch_hold_s + 0.5 / jump.frequency, mid.data());
  bool differs = false;
  for (int i = 0; i < jump.n_joints; i++) {
    differs = differs || std::abs(mid[i] - idle[i]) > 1e-3f;
  }
  if (!differs) {
    std::fprintf(stderr, "FAIL: mid-cycle reference matches the hold -- clock stuck\n");
    failures++;
  }

  if (failures > 0) {
    std::fprintf(stderr, "\n%d check(s) FAILED\n", failures);
    return 1;
  }
  std::printf("OK (jump): %d golden cases matched (worst max_abs_err = %.3e) + 3 property checks\n",
              n_cases, worst);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  const std::string path = argc > 1 ? argv[1] : GAIT_GOLDEN_JSON;
  // Optional: a real deploy policy.json, to confirm its gait_reference block
  // parses and carries the same tables the golden data was generated with.
  const std::string policy_path = argc > 2 ? argv[2] : "";

  nlohmann::json j;
  neural_controller::GaitReference gait;
  try {
    j = read_json(path);
    if (j.contains("jump_reference")) {
      return run_jump_test(j, policy_path);
    }
    neural_controller::parse_gait_reference(j.at("gait_reference"), kActionSize, gait);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  if (!policy_path.empty()) {
    std::ifstream probe(policy_path);
    const bool present = static_cast<bool>(probe);
    probe.close();
    if (!present) {
      // Nothing deployed here yet (download_policy.py has not run); the golden
      // checks below still cover the math.
      std::printf("No policy at %s -- skipping the deployed-policy check\n",
                  policy_path.c_str());
    } else {
      try {
        const nlohmann::json policy = read_json(policy_path);
        neural_controller::GaitReference from_policy;
        neural_controller::parse_gait_reference(policy.at("gait_reference"), kActionSize,
                                                from_policy);
        const int frame = policy.value("single_observation_size", 36);
        if (frame != 36 + kActionSize) {
          std::fprintf(stderr, "FAIL: %s has gait_reference but frame size %d\n",
                       policy_path.c_str(), frame);
          return 1;
        }
        if (from_policy.trot_table != gait.trot_table ||
            from_policy.gallop_table != gait.gallop_table ||
            from_policy.lift_table != gait.lift_table ||
            from_policy.gallop_back_table != gait.gallop_back_table) {
          std::fprintf(stderr, "FAIL: %s tables differ from the golden tables\n",
                       policy_path.c_str());
          return 1;
        }
        std::printf("Deploy policy %s: gait_reference parsed, tables match the golden data\n",
                    policy_path.c_str());
      } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: parsing %s: %s\n", policy_path.c_str(), e.what());
        return 1;
      }
    }
  }

  std::vector<double> default_joint_pos = j.at("default_joint_pos").get<std::vector<double>>();
  if (static_cast<int>(default_joint_pos.size()) != gait.n_joints) {
    std::fprintf(stderr, "FAIL: default_joint_pos has %zu entries, expected %d\n",
                 default_joint_pos.size(), gait.n_joints);
    return 1;
  }

  int failures = 0;
  double worst = 0.0;
  int n_cases = 0;
  std::vector<float> out(gait.n_joints, 0.0f);

  for (const auto &c : j.at("cases")) {
    const double t = c.at("t");
    const double vx = c.at("vx");
    const double vy = c.at("vy");
    const double yaw = c.at("yaw");
    const std::vector<double> expected = c.at("expected").get<std::vector<double>>();

    neural_controller::compute_gait_reference_offset(gait, default_joint_pos, t, vx, vy, yaw,
                                                     out.data());

    double max_err = 0.0;
    for (int i = 0; i < gait.n_joints; i++) {
      max_err = std::max(max_err, std::abs(static_cast<double>(out[i]) - expected[i]));
    }
    worst = std::max(worst, max_err);
    n_cases++;
    if (max_err > kTolerance) {
      std::fprintf(stderr,
                   "FAIL case t=%.3f vx=%+.3f vy=%+.3f yaw=%+.3f: max_abs_err=%.3e\n", t, vx,
                   vy, yaw, max_err);
      failures++;
    }
  }

  // A zero command must produce an all-zero offset: the policy reads that as
  // "stand at the default pose", and it is what the controller's freshly
  // initialized observation history contains.
  neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 3.7, 0.0, 0.0, 0.0,
                                                   out.data());
  for (int i = 0; i < gait.n_joints; i++) {
    if (out[i] != 0.0f) {
      std::fprintf(stderr, "FAIL: zero command gave a nonzero offset at joint %d (%g)\n", i,
                   static_cast<double>(out[i]));
      failures++;
      break;
    }
  }

  // Backward must not equal forward at the same phase: that is the time reversal,
  // and it is exactly what a truncating modulo would silently break.
  std::vector<float> fwd(gait.n_joints), bwd(gait.n_joints);
  neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 0.31, 0.3, 0.0, 0.0,
                                                   fwd.data());
  neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 0.31, -0.3, 0.0, 0.0,
                                                   bwd.data());
  bool differs = false;
  for (int i = 0; i < gait.n_joints; i++) {
    differs = differs || std::abs(fwd[i] - bwd[i]) > 1e-3f;
  }
  if (!differs) {
    std::fprintf(stderr, "FAIL: backward reference matches forward -- time reversal lost\n");
    failures++;
  }

  // Crossing gallop_speed must switch tables.
  std::vector<float> trot(gait.n_joints), gallop(gait.n_joints);
  neural_controller::compute_gait_reference_offset(
      gait, default_joint_pos, 0.4, gait.gallop_speed - 1e-3, 0.0, 0.0, trot.data());
  neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 0.4,
                                                   gait.gallop_speed, 0.0, 0.0, gallop.data());
  differs = false;
  for (int i = 0; i < gait.n_joints; i++) {
    differs = differs || std::abs(trot[i] - gallop[i]) > 1e-3f;
  }
  if (!differs) {
    std::fprintf(stderr, "FAIL: gallop threshold did not switch the reference table\n");
    failures++;
  }

  // Mixed-gait policies: a pure turn must play the lift table, not the trot.
  // Compare against the same gait with the lift table dropped (the legacy path).
  int property_checks = 3;
  if (!gait.lift_table.empty()) {
    property_checks++;
    neural_controller::GaitReference no_lift = gait;
    no_lift.lift_table.clear();
    std::vector<float> lifted(gait.n_joints), trotted(gait.n_joints);
    neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 0.4, 0.0, 0.0,
                                                     0.9, lifted.data());
    neural_controller::compute_gait_reference_offset(no_lift, default_joint_pos, 0.4, 0.0,
                                                     0.0, 0.9, trotted.data());
    differs = false;
    for (int i = 0; i < gait.n_joints; i++) {
      differs = differs || std::abs(lifted[i] - trotted[i]) > 1e-3f;
    }
    if (!differs) {
      std::fprintf(stderr, "FAIL: turning ignored the lift table\n");
      failures++;
    }
  }

  // MixedGaitsJump fixtures: the jump-slot composite cases.
  if (j.contains("jump_slot")) {
    neural_controller::JumpSlot slot;
    try {
      neural_controller::parse_jump_slot(j.at("jump_slot"), kActionSize, slot);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
    property_checks++;
    double slot_worst = 0.0;
    int slot_cases = 0;
    for (const auto &c : j.at("slot_cases")) {
      const double t = c.at("t");
      const std::vector<double> expected = c.at("expected").get<std::vector<double>>();
      neural_controller::compute_mixed_jump_reference_offset(
          gait, slot, default_joint_pos, t, c.at("vx"), c.at("vy"), c.at("yaw"),
          c.at("slot_start"), out.data());
      double max_err = 0.0;
      for (int i = 0; i < gait.n_joints; i++) {
        max_err = std::max(max_err, std::abs(static_cast<double>(out[i]) - expected[i]));
      }
      slot_worst = std::max(slot_worst, max_err);
      slot_cases++;
      if (max_err > kTolerance) {
        std::fprintf(stderr, "FAIL slot case t=%.3f: max_abs_err=%.3e\n", t, max_err);
        failures++;
      }
    }
    // No slot (infinite start) must reproduce the plain mixed reference bit-for-bit.
    std::vector<float> plain(gait.n_joints), composite(gait.n_joints);
    neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 2.3, 0.4, 0.0,
                                                     0.0, plain.data());
    neural_controller::compute_mixed_jump_reference_offset(
        gait, slot, default_joint_pos, 2.3, 0.4, 0.0, 0.0,
        std::numeric_limits<double>::infinity(), composite.data());
    for (int i = 0; i < gait.n_joints; i++) {
      if (plain[i] != composite[i]) {
        std::fprintf(stderr, "FAIL: slot-free composite differs from the mixed reference\n");
        failures++;
        break;
      }
    }
    std::printf("jump_slot: %d composite cases matched (worst %.3e)\n", slot_cases,
                slot_worst);
  }

  if (failures > 0) {
    std::fprintf(stderr, "\n%d check(s) FAILED\n", failures);
    return 1;
  }
  std::printf("OK: %d golden cases matched (worst max_abs_err = %.3e) + %d property checks\n",
              n_cases, worst, property_checks);
  return 0;
}
