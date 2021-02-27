#include "reinforce.hh"

#include <chrono>
#include <vector>

Reinforce::Reinforce(int64_t num_input, int64_t num_actions) : num_input_(num_input), num_actions_(num_actions), 
    optimizer_(torch::optim::Adam(this->parameters(), 1e-3))
{
    fc1_ = register_module("fc1", torch::nn::Linear(10 * num_input_, num_actions));

    // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    // generator_ = std::default_random_engine(seed);
}

torch::Tensor Reinforce::forward(torch::Tensor x)
{
    auto fc1 = fc1_->forward(x);
    return fc1;
}

size_t Reinforce::get_action(double state[10][64])
{
    // assert(state.size() == num_input_);

    torch::Tensor state_tensor = torch::from_blob(state, {1, 10 * num_input_});

    torch::Tensor preds = forward(torch::softmax(state_tensor, 1));
    preds = preds.detach().squeeze();

    std::vector<double> preds_vec;
    for (size_t j = 0; j < num_actions_; j++) {
        preds_vec.push_back(preds[j].item<double>());
    }

    size_t highest_prob_action = std::distance(preds_vec.begin(), std::max_element(preds_vec.begin(), preds_vec.end()));

    return highest_prob_action;
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