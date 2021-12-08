// #include <fstream>
// #include <memory>
// #include <tuple>
// #include <thread>
// #include <math.h>

// #include "exp3_policy.hh"
// #include "exp3_kmeans.hh"
// #include "exp3_boggart.hh"
// #include "ws_client.hh"
// #include "json.hpp"
// #include "timestamp.hh"

// using namespace std;
// using json = nlohmann::json;

// Exp3Policy::Exp3Policy(const WebSocketClient &client,
//                        const string &abr_name, const YAML::Node &abr_config)
//     : ABRAlgo(client, abr_name)
// {
//   if (abr_config["max_lookahead_horizon"])
//   {
//     max_lookahead_horizon_ = min(
//         max_lookahead_horizon_,
//         abr_config["max_lookahead_horizon"].as<size_t>());
//   }

//   if (abr_config["rebuffer_length_coeff"])
//   {
//     rebuffer_length_coeff_ = abr_config["rebuffer_length_coeff"].as<double>();
//   }

//   if (abr_config["ssim_diff_coeff"])
//   {
//     ssim_diff_coeff_ = abr_config["ssim_diff_coeff"].as<double>();
//   }

//   if (abr_config["training_mode"])
//   {
//     training_mode_ = abr_config["training_mode"].as<bool>();
//   }

//   if (abr_config["use_boggart"])
//   {
//     use_boggart_ = abr_config["use_boggart"].as<bool>();
//   }

//   if (abr_config["exp3_dir"])
//   {
//     if (use_boggart_)
//     {
//       exp3_agent_ = new Exp3Boggart(abr_config["exp3_dir"].as<string>(),
//                                     abr_config["learning_rate"].as<double>());
//     }
//     else
//     {
//       exp3_agent_ = new Exp3Kmeans(abr_config["exp3_dir"].as<string>(),
//                                    abr_config["kmeans_dir"].as<string>(),
//                                    abr_config["delta"].as<double>(),
//                                    abr_config["learning_rate"].as<double>());
//     }
//   }

//   dis_buf_length_ = min(dis_buf_length_,
//                         discretize_buffer(WebSocketClient::MAX_BUFFER_S));
// }

// Exp3Policy::~Exp3Policy()
// {
//   delete exp3_agent_;
// }

// double Exp3Policy::get_qoe(double curr_ssim,
//                            double prev_ssim,
//                            uint64_t curr_trans_time,
//                            std::size_t curr_buffer)
// {
//   double qoe = ssim_db(curr_ssim);
//   qoe -= ssim_diff_coeff_ * fabs(ssim_db(curr_ssim) - ssim_db(prev_ssim));

//   double rebuffer = rebuffer_length_coeff_ * max(curr_trans_time * 0.001 - curr_buffer * unit_buf_length_, 0.0);
//   qoe -= rebuffer;

//   // std::cout << "rebuf: " << rebuffer << std::endl;

//   return qoe;
// }

// double Exp3Policy::normalize_reward()
// {
//   if (past_chunks_.size() < 2)
//   {
//     return 0;
//   }

//   Chunk curr_chunk = past_chunks_.back();
//   Chunk prev_chunk = *(past_chunks_.end() - 2);

//   double reward = this->get_qoe(curr_chunk.ssim, prev_chunk.ssim, curr_chunk.trans_time, curr_buffer_);

//   const VideoFormat &min_vformat = sorted_vformats_.front();
//   const VideoFormat &max_vformat = sorted_vformats_.back();

//   const unsigned int vduration = client_.channel()->vduration();
//   uint64_t curr_vts = client_.client_next_vts().value() - vduration;
//   double min_ssim = client_.channel()->vssim(curr_vts).at(min_vformat);
//   double max_ssim = client_.channel()->vssim(curr_vts).at(max_vformat);

//   if (!(min_ssim <= curr_chunk.ssim && curr_chunk.ssim <= max_ssim))
//   {
//     std::cout << "something is wrong with vts or normalization " << min_ssim << ", " << curr_chunk.ssim << ", " << max_ssim << std::endl;
//   }
  
//   // local normalization
//   double min_reward = this->get_qoe(min_ssim, max_ssim, 5000, 0);
//   double max_reward = ssim_db(max_ssim);

//   // global normalization
//   // double min_reward = this->get_qoe(0, 1, 5000, 0);
//   // double max_reward = this->get_qoe(1, 1, 0, 0);

//   double normalized_reward = (reward - min_reward) / (max_reward - min_reward);

//   // std::cout << "rewards (max,min,curr, normalized): " << max_reward << ", " << min_reward << ", " << reward << ", " << normalized_reward << std::endl;
//   // std::cout << "logs transtime& buf: " << curr_chunk.trans_time << "," <<curr_buffer_ << std::endl;

//   return normalized_reward;
// }

// void Exp3Policy::video_chunk_acked(Chunk &&c)
// {
//   past_chunks_.push_back(c);
//   if (past_chunks_.size() > max_num_past_chunks_)
//   {
//     past_chunks_.pop_front();
//   }

//   if (!training_mode_ || is_init_)
//   {
//     return;
//   }

//   // auto before_ts = timestamp_ms();

//   std::thread([&]()
//               {
//                 auto [action_idx, context_idx] = past_datapoints_[curr_ack_round_];

//                 json data;
//                 data["context_idx"] = context_idx;
//                 data["arm"] = action_idx;
//                 data["reward"] = this->normalize_reward();
//                 data["version"] = exp3_agent_->version_;

//                 long status = sender_.post(data, "update");
//               })
//       .detach();

//   curr_ack_round_++;

//   if (exp3_agent_->should_reload())
//   {
//     exp3_agent_->reload_model();
//     past_datapoints_.clear();
//     curr_ack_round_ = 0;
//     last_format_idx_ = 0;
//     last_sorted_format_idx_ = 0;
//     throw logic_error("weights updated, reinit channel");
//   }

//   // auto after = timestamp_ms() - before_ts;
//   // std::cout <<  "diff time:" << after << std::endl;

//   // last_buffer_formats_.pop_front();
// }

// VideoFormat Exp3Policy::select_video_format()
// {
//   // auto before_ts = timestamp_ms(); // TODO: check time

//   reinit();

//   std::vector<std::size_t> available_actions_idx;
//   for (std::size_t i = 0; i < actions_.size(); i++)
//   {
//     int action = actions_.at(i);
//     int sorted_format_idx = last_sorted_format_idx_ + action;
//     if (sorted_format_idx >= 0 && sorted_format_idx < sorted_vformats_.size())
//     {
//       available_actions_idx.push_back(i);
//     }
//   }

//   // std::cout << "available_actions_idx ";
//   // for (auto &v : available_actions_idx)
//   // {
//   //   std::cout << v << " ";
//   // }
//   // std::cout << std::endl;

//   std::tuple<std::size_t, std::size_t> result;
//   if (use_boggart_)
//   {
//     auto [max_conservative, max_risk] = find_optimal_actions(available_actions_idx);
//     result = ((Exp3Boggart *)exp3_agent_)->predict(max_conservative, max_risk, available_actions_idx);
//   }
//   else
//   {
//     // Clustering predict
//     double real_rebuffer = curr_buffer_ * unit_buf_length_;
//     result = ((Exp3Kmeans *)exp3_agent_)->predict(inputs_.back(), real_rebuffer, last_format_idx_, available_actions_idx);
//     inputs_.pop_back();
//   }

//   past_datapoints_.push_back(result);
//   std::size_t action_idx = std::get<0>(result);
//   std::size_t selected_format_in_sorted = last_sorted_format_idx_ + actions_.at(action_idx);

//   auto &vformats = client_.channel()->vformats();
//   auto it = find(vformats.begin(), vformats.end(), sorted_vformats_.at(selected_format_in_sorted));
//   last_format_idx_ = it - vformats.begin();

//   // auto after = timestamp_ms() - before_ts;
//   // std::cout << "diff time:" << after << std::endl;

//   return sorted_vformats_.at(selected_format_in_sorted); // return client_.channel()->vformats()[last_format_idx_];
// }

// std::tuple<std::size_t, std::size_t> Exp3Policy::find_optimal_actions(std::vector<std::size_t> available_actions_idx)
// {
//   // calc expected transmission time for each vformat
//   double expected_trans_time[MAX_LOOKAHEAD_HORIZON + 1][MAX_NUM_FORMATS];
//   for (size_t i = 1; i <= max_lookahead_horizon_; i++)
//   {
//     for (size_t j = 0; j < num_formats_; j++)
//     {
//       expected_trans_time[i][j] = 0;
//       for (size_t k = 0; k < dis_sending_time_; k++)
//       {
//         expected_trans_time[i][j] += sending_time_prob_[i][j][k] * (BINS[k] + BINS[k + 1]) / 2;
//       }
//     }
//   }

//   // std::cout << "buf: " << curr_buffer_ << std::endl;
//   // std::cout << "throughput of last: ";
//   // for (size_t j = 0; j < dis_sending_time_; j++)
//   // {
//   //   std::cout << "," << sending_time_prob_[1][8][j];
//   //   // std::cout << "," << curr_sizes_[1][j];
//   // }
//   // std::cout << std::endl;

//   size_t max_conservative = 0;
//   size_t max_risk = 0;
//   const unsigned int vduration = client_.channel()->vduration();

//   for (size_t i = 1; i <= 1; i++) // lookahead_horizon_
//   {
//     for (size_t action_idx : available_actions_idx)
//     {
//       int action = actions_.at(action_idx);
//       int sorted_format_idx = last_sorted_format_idx_ + action;

//       auto &vformats = client_.channel()->vformats();
//       auto it = find(vformats.begin(), vformats.end(), sorted_vformats_.at(sorted_format_idx));
//       int format_idx = it - vformats.begin();

//       // get highest format chunk can be downloaded less than chunk duration
//       if (expected_trans_time[i][format_idx] < dis_chunk_length_ * unit_buf_length_)
//       {
//         if (curr_ssims_[i][max_conservative] < curr_ssims_[i][format_idx])
//         {
//           max_conservative = action_idx;
//         }
//       }

//       // get highest format chunk can be downloaded less than chunk duration + buffer size
//       if (expected_trans_time[i][format_idx] < curr_buffer_ * unit_buf_length_)
//       {
//         if (curr_ssims_[i][max_risk] < curr_ssims_[i][format_idx])
//         {
//           max_risk = action_idx;
//         }
//       }
//     }
//   }

//   return std::make_tuple(max_conservative, max_risk);
// }

// void Exp3Policy::reinit()
// {
//   curr_round_++;

//   const auto &channel = client_.channel();
//   const auto &vformats = channel->vformats();
//   const unsigned int vduration = channel->vduration();
//   const uint64_t next_ts = client_.next_vts().value();

//   dis_chunk_length_ = discretize_buffer((double)vduration / channel->timescale());
//   num_formats_ = vformats.size();

//   /* initialization failed if there is no ready chunk ahead */
//   if (channel->vready_frontier().value() < next_ts || num_formats_ == 0)
//   {
//     throw runtime_error("no ready chunk ahead");
//   }

//   lookahead_horizon_ = min(
//       max_lookahead_horizon_,
//       (channel->vready_frontier().value() - next_ts) / vduration + 1);

//   curr_buffer_ = min(dis_buf_length_,
//                      discretize_buffer(client_.video_playback_buf()));

//   /* init curr_ssims */
//   if (past_chunks_.size() > 0)
//   {
//     is_init_ = false;
//     curr_ssims_[0][0] = ssim_db(past_chunks_.back().ssim);
//   }
//   else
//   {
//     is_init_ = true;
//   }

//   for (size_t i = 1; i <= lookahead_horizon_; i++)
//   {
//     const auto &data_map = channel->vdata(next_ts + vduration * (i - 1));

//     for (size_t j = 0; j < num_formats_; j++)
//     {
//       try
//       {
//         curr_ssims_[i][j] = ssim_db(
//             channel->vssim(vformats[j], next_ts + vduration * (i - 1)));
//       }
//       catch (const exception &e)
//       {
//         cerr << "Error occurs when getting the ssim of "
//              << next_ts + vduration * (i - 1) << " " << vformats[j] << endl;
//         curr_ssims_[i][j] = MIN_SSIM;
//       }

//       try
//       {
//         curr_sizes_[i][j] = get<1>(data_map.at(vformats[j]));
//       }
//       catch (const exception &e)
//       {
//         cerr << "Error occurs when getting the sizes of "
//              << next_ts + vduration * (i - 1) << " " << vformats[j] << endl;
//         curr_sizes_[i][j] = -1;
//       }
//     }
//   }

//   /* init sending time */
//   reinit_sending_time();

//   int vformats_size = (int)num_formats_;
//   actions_ = {-(vformats_size - 1), -(int)floor(sqrt(vformats_size - 1)), -1, 0, 1, (int)floor(sqrt(vformats_size - 1)), vformats_size - 1};

//   uint64_t curr_vts = client_.next_vts().value();
//   sorted_vformats_ = client_.channel()->vformats();
//   std::sort(sorted_vformats_.begin(),
//             sorted_vformats_.end(),
//             [&](VideoFormat format1, VideoFormat format2)
//             {
//               return client_.channel()->vssim(curr_vts).at(format1) < client_.channel()->vssim(curr_vts).at(format2);
//             });

//   auto it = find(sorted_vformats_.begin(), sorted_vformats_.end(), vformats.at(last_format_idx_));
//   last_sorted_format_idx_ = it - sorted_vformats_.begin();
// }

// void Exp3Policy::deal_all_ban(size_t i)
// {
//   double min_v = 0;
//   size_t min_id = num_formats_;

//   for (size_t j = 0; j < num_formats_; j++)
//   {
//     double tmp = curr_sizes_[i][j];
//     if (tmp > 0 and (min_id == num_formats_ or min_v > tmp))
//     {
//       min_v = curr_sizes_[i][j];
//       min_id = j;
//     }
//   }

//   if (min_id == num_formats_)
//   {
//     min_id = 0;
//   }

//   is_ban_[i][min_id] = false;
//   for (size_t k = 0; k < dis_sending_time_; k++)
//   {
//     sending_time_prob_[i][min_id][k] = 0;
//   }

//   sending_time_prob_[i][min_id][dis_sending_time_] = 1;
// }

// size_t Exp3Policy::discretize_buffer(double buf)
// {
//   return (buf + unit_buf_length_ * 0.5) / unit_buf_length_;
// }
