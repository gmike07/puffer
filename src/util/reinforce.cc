#include "reinforce.hh"

#include <chrono>

Reinforce::Reinforce(int64_t num_input, int64_t num_actions) : num_input_(num_input), num_actions_(num_actions)
{
    fc1_ = register_module("fc1", torch::nn::Linear(num_input, num_actions));

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator_ = std::default_random_engine(seed);

    optimizer_ = torch::optim::Adam(this->parameters, torch::optim::AdamOptions(5e-4).beta1(0.5));
}

torch::Tensor Reinforce::forward(torch::Tensor x)
{
    x = torch::relu(fc1_->forward(x.reshape({x.size(0), num_input_})));
    x = torch::softmax(x, 0.5);
    return x;
}

std::tuple<double, torch::Tensor> Reinforce::get_action(std::vector<double> state)
{
    assert(state.size() == num_input_);
    torch::Tensor state_tensor = torch::from_blob(state.data(), {1, num_input_}).clone();
    torch::Tensor probs = forward(state_tensor);
    probs = probs.squeeze();
    std::vector<double> data = probs.detach().item<std::vector<double>>();
    std::discrete_distribution<> dist(data.begin(), data.end());
    int highest_prob_action = dist(generator_);
    torch::Tensor log_prob = torch::log(probs[highest_prob_action]);
    return std::tuple<double, torch::Tensor>{highest_prob_action, log_prob};
}

void Reinforce::update_policy(std::vector rewards, std::vector log_probs)
{
    std::vector discounted_rewards = rewards;
    // completet dicounted rewards

    std::vector gradients;
    for (int i = 0; i < log_probs.size(); i++)
    {
        gradients.push_pack(-log_probs[i] * discounted_rewards[i]);
    }

    optimizer_->zero_grad();
    torch::Tensor policy_gradient = torch::stack(gradients).sum();
    policy_gradients->backward();
    optimizer_->step();
}