#include "reinforce.hh"

#include <chrono>
#include <vector>

Reinforce::Reinforce(int64_t num_input, int64_t num_actions) : num_input_(num_input), num_actions_(num_actions), 
    optimizer_(torch::optim::Adam(this->parameters(), 1e-3))
{
    fc1_ = register_module("fc1", torch::nn::Linear(num_input, num_actions));

    // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    // generator_ = std::default_random_engine(seed);
}

torch::Tensor Reinforce::forward(torch::Tensor x)
{
    x = torch::relu(fc1_->forward(x.reshape({x.size(0), num_input_})));
    x = torch::softmax(x, 0.5);
    return x;
}

size_t Reinforce::get_action(double state[20][21])
{
    // assert(state.size() == num_input_);

    torch::Tensor state_tensor = torch::from_blob(state, {20, num_input_});

    torch::Tensor probs = forward(state_tensor);
    probs = probs.squeeze();
    // std::vector<double> data = probs.detach().item<std::vector<double>>();

    // return std::distance(data.begin(), std::max_element(data.begin(), data.end()));
    return 1;
}

void Reinforce::update_policy(std::vector<double> rewards, std::vector<double> log_probs)
{
    std::vector<double> discounted_rewards = rewards;
    // complete dicounted rewards

    std::vector<double> gradients;
    for (int i = 0; i < log_probs.size(); i++)
    {
        gradients.push_back(-log_probs[i] * discounted_rewards[i]);
        gradients.push_back(1);
    }

    optimizer_.zero_grad();
    // torch::tensor gradients_as_tensor = torch::from_blob(gradients.data(), )
    // torch::Tensor policy_gradients = torch::stack(gradients).sum();

    // policy_gradients.backward();
    // optimizer_.step();
}