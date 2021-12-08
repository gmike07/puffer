// #include "reinforce_policy.hh"

// #include <fstream>
// #include <memory>
// #include <tuple>


// #include "ws_client.hh"
// #include "json.hpp"

// using namespace std;
// using json = nlohmann::json;

// ReinforcePolicy::ReinforcePolicy(const WebSocketClient & client,
//                const string & abr_name, const YAML::Node & abr_config)
//   : ABRAlgo(client, abr_name), rl_model_(Reinforce(HIDDEN_SIZE, 10))
// {
//   if (abr_config["max_lookahead_horizon"]) {
//     max_lookahead_horizon_ = min(
//       max_lookahead_horizon_,
//       abr_config["max_lookahead_horizon"].as<size_t>());
//   }

//   if (abr_config["rebuffer_length_coeff"]) {
//     rebuffer_length_coeff_ = abr_config["rebuffer_length_coeff"].as<double>();
//   }

//   if (abr_config["ssim_diff_coeff"]) {
//     ssim_diff_coeff_ = abr_config["ssim_diff_coeff"].as<double>();
//   }

//   if (abr_config["training_mode"]) {
//     training_mode_ = abr_config["training_mode"].as<bool>();
//   }

//   dis_buf_length_ = min(dis_buf_length_,
//                         discretize_buffer(WebSocketClient::MAX_BUFFER_S));

//   if (abr_config["policy_model_dir"]) {
//     policy_model_dir_ = abr_config["policy_model_dir"].as<string>();
//   }
//   load_weights();
// }

// void ReinforcePolicy::load_weights()
// {
//   vector<fs::path> files;
//   for (const auto & entry: fs::directory_iterator(policy_model_dir_)) {
//     files.push_back(entry.path());
//   }

//   sort(files.begin(), files.end(), [](const fs::path f1, const fs::path f2)
//   {
//     std::string s1 = f1.c_str();
//     std::string s2 = f2.c_str();
//     int n1 = stoi(s1.substr(s1.find_last_of('_') + 1, s1.find_last_of('_')));
//     int n2 = stoi(s2.substr(s2.find_last_of('_') + 1, s2.find_last_of('_')));
//     return n1 < n2;
//   });

//   string path_str = files.back().c_str();
//   std::cout << files << std::endl;
//   policy_ = torch::jit::load(path_str);

//   version_ = stoi(path_str.substr(path_str.find_last_of('_') + 1, path_str.find_last_of('_')));
//   version_++;

//   cout << "version " << version_ << endl;
// }

// void ReinforcePolicy::video_chunk_acked(Chunk && c)
// {
//   past_chunks_.push_back(c);
//   if (past_chunks_.size() > max_num_past_chunks_) {
//     past_chunks_.pop_front();
//   }

//   if (past_chunks_.size() < 2){
//     return;
//   }

//   double qoe = calc_qoe();
//   send_chunk_statistics(qoe);
// }

// double ReinforcePolicy::calc_qoe()
// {
//   Chunk curr_chunk = past_chunks_.back();
//   Chunk prev_chunk = past_chunks_.end()[-2];

//   double qoe = ssim_db(curr_chunk.ssim);
//   qoe -= ssim_diff_coeff_ * fabs(ssim_db(curr_chunk.ssim) - ssim_db(prev_chunk.ssim));

//   int rebuffer = max(curr_chunk.trans_time - curr_buffer_, (unsigned long)0);
//   qoe -= rebuffer_length_coeff_ * rebuffer;
  
//   std::cout << qoe << std::endl;

//   return qoe;
// }

// VideoFormat ReinforcePolicy::select_video_format()
// {
//   reinit();

//   // getting only the next chunk sending time porb
//   auto& next_sending_time = sending_time_prob_[1];

//   std::tuple<size_t, torch::Tensor> result = this->get_action(next_sending_time);
//   torch::Tensor log_prob = std::get<1>(result);
//   log_probs_.push_back(log_prob);
//   if (log_probs_.size() > DONE){
//     log_probs_.pop_front();
//   }

//   size_t format = std::get<0>(result);
//   return client_.channel()->vformats()[format];
// }

// std::tuple<size_t,torch::Tensor> ReinforcePolicy::get_action(double state[20][64])
// {
//     std::vector<torch::jit::IValue> torch_inputs;

//     torch_inputs.push_back(torch::from_blob(state, {20 * 64}, torch::kDouble).unsqueeze(0));
    
//     torch::Tensor preds = policy_->forward(torch_inputs).toTensor();
//     preds = preds.squeeze();

//     torch::Tensor max_tensor = torch::max_values(preds, 0);
    
//     preds = preds.detach();
//     preds = torch::softmax(preds, 0);
    
//     std::vector<double> preds_vec;
//     for (size_t j = 0; j < 10; j++) {
//         preds_vec.push_back(preds[j].item<double>());
//     }

//     size_t highest_prob_action = std::distance(preds_vec.begin(), std::max_element(preds_vec.begin(), preds_vec.end()));

//     return std::make_tuple(highest_prob_action, torch::log(max_tensor));
// }

// void ReinforcePolicy::reinit()
// {
//   curr_round_++;

//   const auto & channel = client_.channel();
//   const auto & vformats = channel->vformats();
//   const unsigned int vduration = channel->vduration();
//   const uint64_t next_ts = client_.next_vts().value();

//   dis_chunk_length_ = discretize_buffer((double) vduration / channel->timescale());
//   num_formats_ = vformats.size();

//   /* initialization failed if there is no ready chunk ahead */
//   if (channel->vready_frontier().value() < next_ts || num_formats_ == 0) {
//     throw runtime_error("no ready chunk ahead");
//   }

//   lookahead_horizon_ = min(
//     max_lookahead_horizon_,
//     (channel->vready_frontier().value() - next_ts) / vduration + 1);

//   curr_buffer_ = min(dis_buf_length_,
//                      discretize_buffer(client_.video_playback_buf()));

//   /* init curr_ssims */
//   if (past_chunks_.size() > 0) {
//     is_init_ = false;
//     curr_ssims_[0][0] = ssim_db(past_chunks_.back().ssim);
//   } else {
//     is_init_ = true;
//   }

//   for (size_t i = 1; i <= lookahead_horizon_; i++) {
//     const auto & data_map = channel->vdata(next_ts + vduration * (i - 1));

//     for (size_t j = 0; j < num_formats_; j++) {

//       try {
//         curr_ssims_[i][j] = ssim_db(
//             channel->vssim(vformats[j], next_ts + vduration * (i - 1)));
//       } catch (const exception & e) {
//         cerr << "Error occurs when getting the ssim of "
//              << next_ts + vduration * (i - 1) << " " << vformats[j] << endl;
//         curr_ssims_[i][j] = MIN_SSIM;
//       }

//       try {
//         curr_sizes_[i][j] = get<1>(data_map.at(vformats[j]));
//       } catch (const exception & e) {
//         cerr << "Error occurs when getting the sizes of "
//              << next_ts + vduration * (i - 1) << " " << vformats[j] << endl;
//         curr_sizes_[i][j] = -1;
//       }
//     }
//   }

//   /* init sending time */
//   reinit_sending_time();
// }

// void ReinforcePolicy::deal_all_ban(size_t i)
// {
//   double min_v = 0;
//   size_t min_id = num_formats_;

//   for (size_t j = 0; j < num_formats_; j++) {
//     double tmp = curr_sizes_[i][j];
//     if (tmp > 0 and (min_id == num_formats_ or min_v > tmp)) {
//       min_v = curr_sizes_[i][j];
//       min_id = j;
//     }
//   }

//   if (min_id == num_formats_) {
//     min_id = 0;
//   }

//   is_ban_[i][min_id] = false;
//   for (size_t k = 0; k < dis_sending_time_; k++) {
//      sending_time_prob_[i][min_id][k] = 0;
//   }

//   sending_time_prob_[i][min_id][dis_sending_time_] = 1;
// }

// size_t ReinforcePolicy::discretize_buffer(double buf)
// {
//   return (buf + unit_buf_length_ * 0.5) / unit_buf_length_;
// }
