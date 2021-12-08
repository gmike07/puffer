#include "exp3_boggart.hh"

#include <assert.h>
#include <cmath>

Exp3Boggart::Exp3Boggart(std::string model_path, double lr) : Exp3(model_path, lr)
{
  reload_model();
}

std::tuple<std::size_t, std::size_t> Exp3Boggart::predict(std::size_t max_conservative,
                                                          std::size_t max_risk,
                                                          std::vector<std::size_t> available_actions_idx)
{
  std::size_t flat_idx = calc_inner_index(max_conservative, max_risk);
  std::size_t arm = contexts_[flat_idx].predict();

  if (arm > available_actions_idx.back())
  {
    arm = available_actions_idx.back();
  }
  else if (arm < available_actions_idx.front())
  {
    arm = available_actions_idx.front();
  }

  return std::make_tuple(arm, flat_idx);
}

std::size_t Exp3Boggart::calc_inner_index(std::size_t max_conservative, std::size_t max_risk)
{
  return max_conservative * num_of_arms_ + max_risk;
}
