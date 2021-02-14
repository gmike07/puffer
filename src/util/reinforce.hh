#ifndef REINFORCE
#define REINFORCE

#include <torch/torch.h>
#include <random>
#include "torch/script.h"

class Reinforce : torch::nn::Module
{
public:
  Reinforce(int64_t num_input, int64_t num_actions);
  torch::Tensor forward(torch::Tensor x);
  size_t get_action(double state[20][64]);
  void update_policy(std::vector<double> rewards, std::vector<double> log_probs);
private:
  torch::nn::Linear fc1_{nullptr};
  int64_t num_input_;
  int64_t num_actions_;
  std::default_random_engine generator_;
  torch::optim::Adam optimizer_;
};

#endif