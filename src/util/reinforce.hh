#ifndef REINFORCE
#define REINFORCE

#include <torch/extension.h>
#include <random>

class Reinforce : torch::nn::Module
{
public:
  Reinforce(int64_t num_input, int64_t num_actions);
  torch::Tensor forward(torch::Tensor x);
  std::tuple<double, torch::Tensor> get_action(std::vector<double> state);
private:
  torch::nn::Linear fc1_{nullptr};
  int64_t num_input_;
  int64_t num_actions_;
  std::default_random_engine generator_;
  torch::optim::Adam optimizer_;
};

#endif