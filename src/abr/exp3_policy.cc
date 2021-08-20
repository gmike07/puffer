#include "exp3_policy.hh"

#include <fstream>
#include <memory>
#include <tuple>
#include <thread>

#include "ws_client.hh"
#include "json.hpp"
#include "timestamp.hh"

using namespace std;
using json = nlohmann::json;

Exp3Policy::Exp3Policy(const WebSocketClient & client,
               const string & abr_name, const YAML::Node & abr_config)
  : ABRAlgo(client, abr_name)
{
  if (abr_config["max_lookahead_horizon"]) {
    max_lookahead_horizon_ = min(
      max_lookahead_horizon_,
      abr_config["max_lookahead_horizon"].as<size_t>());
  }

  if (abr_config["rebuffer_length_coeff"]) {
    rebuffer_length_coeff_ = abr_config["rebuffer_length_coeff"].as<double>();
  }

  if (abr_config["ssim_diff_coeff"]) {
    ssim_diff_coeff_ = abr_config["ssim_diff_coeff"].as<double>();
  }

  if (abr_config["training_mode"]) {
    training_mode_ = abr_config["training_mode"].as<bool>();
  }

  if (abr_config["exp3_dir"]) {
    exp3_agent_ = Exp3(abr_config["exp3_dir"].as<string>(), 
                       abr_config["normalization_dir"].as<string>(), 
                       abr_config["learning_rate"].as<double>(),
                       abr_config["delta"].as<double>());
  }

  dis_buf_length_ = min(dis_buf_length_,
                        discretize_buffer(WebSocketClient::MAX_BUFFER_S));
}

double Exp3Policy::get_qoe(double curr_ssim, 
                          double prev_ssim, 
                          uint64_t curr_trans_time,
                          std::size_t curr_buffer)
{
  double qoe = ssim_db(curr_ssim);
  qoe -= ssim_diff_coeff_ * fabs(ssim_db(curr_ssim) - ssim_db(prev_ssim));

  double rebuffer = max(curr_trans_time*0.001 - curr_buffer*unit_buf_length_, 0.0);
  qoe -= rebuffer;

  // std::cout << "qoe:" <<  qoe << ",ssim " << ssim_db(curr_ssim) << ", jitter: " << ssim_diff_coeff_ * fabs(ssim_db(curr_ssim) - ssim_db(prev_ssim)) << ", rebuf " << rebuffer << std::endl;

  // std::cout << ", rebuf " << curr_trans_time << ", " << curr_buffer*unit_buf_length_ << std::endl;
  // std::cout << "calc qoe " << ssim_db(curr_ssim) << ", jitter: " << ssim_diff_coeff_ * fabs(ssim_db(curr_ssim) - ssim_db(prev_ssim)) << ", rebuf " << rebuffer_length_coeff_ * rebuffer << std::endl;

  return qoe;
}

double Exp3Policy::normalize_reward()
{
  if (past_chunks_.size() < 2) {
    return 0;
  }

  Chunk curr_chunk = past_chunks_.back();
  Chunk prev_chunk = *(past_chunks_.end()-2);

  double reward = this->get_qoe(curr_chunk.ssim, prev_chunk.ssim, curr_chunk.trans_time, curr_buffer_);

  uint64_t next_vts = client_.next_vts().value() - 180180*(last_buffer_formats_.size()-curr_ack_round_); // get current_vts - by subtract video chunk length
  const VideoFormat & min_vformat = *std::min_element(client_.channel()->vformats().begin(), 
                                                        client_.channel()->vformats().end(), 
                                                        [&](VideoFormat format1, VideoFormat format2){
    return client_.channel()->vssim(next_vts).at(format1) < client_.channel()->vssim(next_vts).at(format2);
  });
  const VideoFormat & max_vformat = *std::max_element(client_.channel()->vformats().begin(), 
                                                        client_.channel()->vformats().end(), 
                                                        [&](VideoFormat format1, VideoFormat format2){
    return client_.channel()->vssim(next_vts).at(format1) < client_.channel()->vssim(next_vts).at(format2);
  });
  

  double min_ssim = client_.channel()->vssim(next_vts).at(min_vformat);
  double max_ssim = client_.channel()->vssim(next_vts).at(max_vformat);

  // auto [buffer, last_format, format] = last_buffer_formats_[curr_ack_round_];
  // std::cout << "validate equal " << curr_chunk.format << ", " << prev_chunk.format << ", " << client_.channel()->vformats()[format] << std::endl;


  if (!(min_ssim <= curr_chunk.ssim && curr_chunk.ssim <= max_ssim)){
    std::cout << "something is wrong with vts or normalization " << min_ssim << ", " << curr_chunk.ssim << ", " << max_ssim << std::endl;
  }

  double min_reward = this->get_qoe(min_ssim, max_ssim, 5000, 0);
  double max_reward = ssim_db(max_ssim);

  // max/min global normalization
  // double min_reward = this->get_qoe(0, 1, 5000, 0);
  // double max_reward = ssim_db(1);

  double normalized_reward = (reward - min_reward) / (max_reward - min_reward);
  // std::cout << "rewards (max,min,curr, normalized): " << max_reward << ", " << min_reward << ", " << reward << ", " << normalized_reward << std::endl;
   
  return normalized_reward;
}

void Exp3Policy::video_chunk_acked(Chunk && c)
{
  past_chunks_.push_back(c);
  if (past_chunks_.size() > max_num_past_chunks_) {
    past_chunks_.pop_front();
  }

  if (!training_mode_) {
    return;
  }

  // std::thread([&](){ 
  std::size_t context_idx = contexts_[curr_ack_round_];
  std::vector<double> last_input = inputs_[curr_ack_round_];
  auto [buffer, last_format, format] = last_buffer_formats_[curr_ack_round_];

  json data;
  data["context_idx"] = context_idx;
  data["datapoint"] = last_input;
  data["buffer_size"] = buffer;
  data["last_format"] = last_format;
  data["arm"] = format;
  data["reward"] = this->normalize_reward();
  data["version"] = exp3_agent_.version_;
  
  long status = sender_.post(data, "update"); 

  if (status == 406) {
    exp3_agent_.reload_model();
    contexts_.clear();
    inputs_.clear();
    last_buffer_formats_.clear();
    curr_ack_round_ = 0;
    last_format_ = 0; 
    throw logic_error("weights updated, reinit channel");
  }

  curr_ack_round_++;
    // inputs_.pop_front();
    // last_buffer_formats_.pop_front();
  // }).detach();
}

VideoFormat Exp3Policy::select_video_format()
{
  // auto before_ts = timestamp_ms();
  reinit();
  
  auto [format, context_idx] = exp3_agent_.predict(inputs_.back(), curr_buffer_, last_format_);
  last_buffer_formats_.push_back(std::tuple<size_t,size_t,size_t>{curr_buffer_, last_format_, format});
  contexts_.push_back(context_idx);
  last_format_ = format;

  // std::cout << "sizes:" << last_buffer_formats_.size() << "," << contexts_.size() << "," << inputs_.size() << std::endl;

  // auto after = timestamp_ms() - before_ts;
  // std::cout <<  "diff time:" << after << std::endl;

  return client_.channel()->vformats()[format];
}

void Exp3Policy::reinit()
{
  curr_round_++;

  const auto & channel = client_.channel();
  const auto & vformats = channel->vformats();
  const unsigned int vduration = channel->vduration();
  const uint64_t next_ts = client_.next_vts().value();

  dis_chunk_length_ = discretize_buffer((double) vduration / channel->timescale());
  num_formats_ = vformats.size();

  /* initialization failed if there is no ready chunk ahead */
  if (channel->vready_frontier().value() < next_ts || num_formats_ == 0) {
    throw runtime_error("no ready chunk ahead");
  }

  lookahead_horizon_ = min(
    max_lookahead_horizon_,
    (channel->vready_frontier().value() - next_ts) / vduration + 1);

  curr_buffer_ = min(dis_buf_length_,
                     discretize_buffer(client_.video_playback_buf()));

  /* init curr_ssims */
  if (past_chunks_.size() > 0) {
    is_init_ = false;
    curr_ssims_[0][0] = ssim_db(past_chunks_.back().ssim);
  } else {
    is_init_ = true;
  }

  for (size_t i = 1; i <= lookahead_horizon_; i++) {
    const auto & data_map = channel->vdata(next_ts + vduration * (i - 1));

    for (size_t j = 0; j < num_formats_; j++) {

      try {
        curr_ssims_[i][j] = ssim_db(
            channel->vssim(vformats[j], next_ts + vduration * (i - 1)));
      } catch (const exception & e) {
        cerr << "Error occurs when getting the ssim of "
             << next_ts + vduration * (i - 1) << " " << vformats[j] << endl;
        curr_ssims_[i][j] = MIN_SSIM;
      }

      try {
        curr_sizes_[i][j] = get<1>(data_map.at(vformats[j]));
      } catch (const exception & e) {
        cerr << "Error occurs when getting the sizes of "
             << next_ts + vduration * (i - 1) << " " << vformats[j] << endl;
        curr_sizes_[i][j] = -1;
      }
    }
  }

  /* init sending time */
  reinit_sending_time();
}

size_t Exp3Policy::discretize_buffer(double buf)
{
  return (buf + unit_buf_length_ * 0.5) / unit_buf_length_;
}
