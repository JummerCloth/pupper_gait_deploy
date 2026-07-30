#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "neural_controller/gait_reference.hpp"

namespace neural_controller {

/**
 * Fill a GaitReference from the "gait_reference" block of a deploy JSON.
 *
 * Templated on the JSON type so this header needs no include of its own: RTNeural
 * vendors nlohmann/json behind a relative path, so `nlohmann::json` is only
 * reachable to translation units that already pulled in RTNeural.h. The host-side
 * parity test includes the vendored header directly and instantiates this the same
 * way, which is the point -- the controller and the test share one parser.
 *
 * Throws std::runtime_error on anything malformed (a missing field, or a table whose
 * shape disagrees with n_samples) so a bad policy fails at load, not mid-stride.
 *
 * @param block    The "gait_reference" object itself, not the whole policy.
 * @param n_joints Number of actuated joints the controller drives.
 */
template <typename Json>
inline void parse_gait_reference(const Json &block, int n_joints, GaitReference &gait) {
  gait.n_joints = n_joints;
  gait.n_samples = block.at("n_samples");
  gait.frequency = block.at("frequency");
  gait.blend_speed = block.at("blend_speed");
  gait.gallop_speed = block.at("gallop_speed");
  gait.gallop_freq_mult = block.at("gallop_freq_mult");
  gait.dir_threshold = block.at("dir_threshold");

  auto read_table = [&](const std::string &key, std::vector<double> &table) {
    const auto &rows = block.at(key);
    if (static_cast<int>(rows.size()) != gait.n_samples) {
      throw std::runtime_error(key + " has " + std::to_string(rows.size()) +
                               " rows, expected n_samples=" + std::to_string(gait.n_samples));
    }
    table.clear();
    table.reserve(static_cast<std::size_t>(gait.n_samples) * n_joints);
    for (const auto &row : rows) {
      if (static_cast<int>(row.size()) != n_joints) {
        throw std::runtime_error(key + " row has " + std::to_string(row.size()) +
                                 " entries, expected " + std::to_string(n_joints));
      }
      for (const auto &v : row) {
        table.push_back(v);
      }
    }
  };
  read_table("trot_table", gait.trot_table);
  read_table("gallop_table", gait.gallop_table);

  if (!gait.valid()) {
    throw std::runtime_error("gait_reference block is not self-consistent");
  }
}

}  // namespace neural_controller
