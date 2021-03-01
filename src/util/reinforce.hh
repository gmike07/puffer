#ifndef REINFORCE
#define REINFORCE

#include <torch/torch.h>
#include <random>
#include "torch/script.h"
#include <tuple>

class Reinforce : torch::nn::Module
{
public:
  Reinforce(int64_t num_input, int64_t num_actions);
  torch::Tensor forward(torch::Tensor x);
  std::tuple<size_t,torch::Tensor> get_action(double state[20][64]);
  void update_policy(std::vector<double> rewards, std::vector<torch::Tensor> log_probs);
private:
  // torch::nn::Linear fc1_{nullptr};
  torch::Tensor w1_{};
  torch::Tensor b1_{};
  int64_t num_input_;
  int64_t num_actions_;
  std::default_random_engine generator_;
  torch::optim::Adam *optimizer_{nullptr};
  torch::nn::Linear fc1_;
};

#endif