#include "reinforce.hh"

#include <chrono>
#include <vector>
#include <math.h> 

Reinforce::Reinforce(int64_t num_input, int64_t num_actions) : 
    // torch::nn::Module(), 
    num_input_(num_input), 
    num_actions_(num_actions),
    fc1_(torch::nn::Linear(64, 10))
{
    // auto opts = torch::TensorOptions().dtype(torch::kDouble).requires_grad(true);
    // w1_ = register_parameter("w1", torch::randn({1 * 64, num_actions}, opts));
    // b1_ = register_parameter("b1", torch::randn(num_actions, opts));
    fc1_->to(torch::kDouble);
    register_module("fc1", fc1_);
    optimizer_ = new torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(1e2));
}

torch::Tensor Reinforce::forward(torch::Tensor x)
{
    // auto fc1 = torch::addmm(b1_, x, w1_);
    torch::Tensor fc1 = fc1_->forward(x);
    return fc1;
}

std::tuple<size_t,torch::Tensor> Reinforce::get_action(double state[20][64])
{
    std::vector<double> v_state(state[0], state[0] + 1 * 64);

    torch::Tensor weights = b1_;
    // std::cout << weights << std::endl;

    auto opts = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor state_tensor = torch::from_blob(v_state.data(), {1 * 64}, opts);
    // std::cout << state_tensor << std::endl;
    
    torch::Tensor softmax_state = torch::softmax(state_tensor, 0);

    torch::Tensor preds = forward(softmax_state.unsqueeze(0));
    preds = preds.squeeze();

    torch::Tensor max_tensor = torch::max_values(preds, 0);
    // std::cout << "max_tensor " << max_tensor << std::endl;  
    // std::cout << "forward output " << preds << std::endl;   

    preds = preds.detach();
    
    std::vector<double> preds_vec;
    for (size_t j = 0; j < num_actions_; j++) {
        preds_vec.push_back(preds[j].item<double>());
    }

    double maximum_val = *std::max_element(preds_vec.begin(), preds_vec.end());
    // if (std::isnan(maximum_val)){
    //     maximum_val = 0;
    // }

    size_t highest_prob_action = std::distance(preds_vec.begin(), std::max_element(preds_vec.begin(), preds_vec.end()));

    return std::make_tuple(highest_prob_action, torch::log(max_tensor));
}

void Reinforce::update_policy(std::vector<double> rewards, std::vector<torch::Tensor> log_probs)
{
    optimizer_->zero_grad();

    torch::Tensor gradients = torch::zeros_like(log_probs[0]);
    for (int i = 0; i < log_probs.size(); i++)
    {
        gradients = gradients.add(-log_probs[i] * 2);
    }
    torch::Tensor sum_gradients = gradients.sum();

    sum_gradients.backward();

    std::cout << "grads" << std::endl;  
    std::cout << gradients << std::endl;
    // std::cout << sum_gradients << std::endl;
    // std::cout << fc1_->bias << std::endl;   
    std::cout << fc1_->bias.grad() << std::endl;   
    
    optimizer_->step();
}