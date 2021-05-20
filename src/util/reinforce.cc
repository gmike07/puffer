#include "reinforce.hh"

#include <chrono>
#include <vector>
#include <math.h> 

Reinforce::Reinforce(int64_t num_input, int64_t num_actions) : 
    save_round_(0),
    num_input_(num_input), 
    num_actions_(num_actions),
    fc1_(torch::nn::Linear(10 * 64, 10)),
    fc2_(torch::nn::Linear(10 * 64, 10))
{
    fc1_->to(torch::kDouble);
    register_module("fc1", fc1_);
    fc2_->to(torch::kDouble);
    register_module("fc2", fc2_);
    optimizer_ = new torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(1e-2));
}

torch::Tensor Reinforce::forward(torch::Tensor x)
{
    torch::Tensor fc1 = fc1_->forward(x);
    // torch::Tensor fc2 = fc2_->forward(fc1);
    return fc1;
}

std::tuple<size_t,torch::Tensor> Reinforce::get_action(double state[20][64])
{
    std::vector<double> v_state(state[0], state[0] + 10 * 64);

    auto opts = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor state_tensor = torch::from_blob(v_state.data(), {10 * 64}, opts);
    
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

    // std::cout << "highest_prob_action " << preds_vec << std::endl;

    return std::make_tuple(highest_prob_action, torch::log(max_tensor));
}

void Reinforce::update_policy(std::vector<double> rewards, std::vector<torch::Tensor> log_probs)
{
    // calc discounted rewards
    const int GAMMA = 0.5;
    std::vector<double> discounted_rewards;
    for (std::size_t t = 0; t < rewards.size(); t++){
        int Gt = 0;
        int pw = 0;
        for (std::size_t r = t; r < rewards.size(); r++){
            Gt += pow(GAMMA, pw) * r;
            pw++;
        }
        discounted_rewards.push_back(Gt);
    }

    int size = discounted_rewards.size();
    torch::Tensor discounted_rewards_tensor = 
        torch::from_blob(discounted_rewards.data(), { size }, torch::kDouble);

    discounted_rewards_tensor = (discounted_rewards_tensor - discounted_rewards_tensor.mean()) / 
                                (discounted_rewards_tensor.std() + 1e-9);

    // update parameters
    optimizer_->zero_grad();

    torch::Tensor gradients = torch::zeros_like(log_probs[0]);
    for (int i = 0; i < log_probs.size(); i++)
    {
        gradients = gradients.add(-log_probs[i] * discounted_rewards_tensor[i]);
    }

    gradients.backward();

    // std::cout << "grads" << std::endl;  
    // std::cout << gradients << std::endl;
    // std::cout << sum_gradients << std::endl;
    // std::cout << fc1_->bias << std::endl;   
    // std::cout << fc1_->bias.grad() << std::endl;   
    
    optimizer_->step();

    save_round_++;
    std::cout << "save round " << save_round_ << std::endl;
    
    if (save_round_ % 60*10 == 0){
        std::cout << "saving point " << save_round_ << std::endl;

        string model_path = "/home/csuser/puffer/ttp/policy/model" + std::to_string(save_round_) + ".pt";
        torch::serialize::OutputArchive output_archive;
        this->save(output_archive);
        output_archive.save_to(model_path);
    }
}