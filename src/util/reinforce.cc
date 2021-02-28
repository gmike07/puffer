#include "reinforce.hh"

#include <chrono>
#include <vector>
#include <math.h> 

Reinforce::Reinforce(int64_t num_input, int64_t num_actions) : torch::nn::Module(), num_input_(num_input), num_actions_(num_actions), 
    optimizer_(torch::optim::Adam(this->parameters(), 1e-3))
{
    fc1_ = torch::nn::Linear(10 * 64, num_actions);
    fc1_->weight = fc1_->weight.to(torch::kDouble);
    fc1_->bias = fc1_->bias.to(torch::kDouble); 
}

torch::Tensor Reinforce::forward(torch::Tensor x)
{
    auto fc1 = fc1_->forward(x);
    return fc1;
}

std::tuple<size_t,double> Reinforce::get_action(double state[20][64])
{
    std::vector<double> v_state(state[0], state[0] + 10 * 64);

    torch::Tensor weights = fc1_->weight;
    // std::cout << weights << std::endl;

    auto opts = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor state_tensor = torch::from_blob(v_state.data(), {10 * 64}, opts);
    // std::cout << state_tensor << std::endl;
    
    // torch::Tensor softmax_state = torch::softmax(state_tensor, 0);

    torch::Tensor preds = forward(state_tensor.unsqueeze(0));
    preds = preds.detach().squeeze();

    std::vector<double> preds_vec;
    for (size_t j = 0; j < num_actions_; j++) {
        preds_vec.push_back(preds[j].item<double>());
    }

    double maximum_val = *std::max_element(preds_vec.begin(), preds_vec.end());
    if (std::isnan(maximum_val)){
        maximum_val = 0;
    }

    size_t highest_prob_action = std::distance(preds_vec.begin(), std::max_element(preds_vec.begin(), preds_vec.end()));

    return std::make_tuple(highest_prob_action, log(maximum_val));
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
    torch::Tensor gradients_as_tensor = torch::from_blob(gradients.data(), gradients.size());
    torch::Tensor policy_gradients = torch::stack(gradients_as_tensor).sum();

    policy_gradients.backward();
    optimizer_.step();
}