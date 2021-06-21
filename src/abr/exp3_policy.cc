#include "exp3_policy.hh"

#include <fstream>
#include <memory>
#include <tuple>

#include "ws_client.hh"
#include "json.hpp"

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

  if (abr_config["use_puffer"]) {
    training_mode_ = abr_config["use_puffer"].as<bool>();
  }

  dis_buf_length_ = min(dis_buf_length_,
                        discretize_buffer(WebSocketClient::MAX_BUFFER_S));
}

void Exp3Policy::video_chunk_acked(Chunk && c)
{
  past_chunks_.push_back(c);
  if (past_chunks_.size() > max_num_past_chunks_) {
    past_chunks_.pop_front();
  }
}

VideoFormat Exp3Policy::select_video_format()
{
  reinit();

  size_t format = this->get_bitrate();
  inputs_.clear();
  last_format_ = format;

  return client_.channel()->vformats()[format];
}

size_t Exp3Policy::get_bitrate()
{
    auto& state = sending_time_prob_[1];

    json data;
    data["datapoint"] = inputs_;
    data["buffer_size"] = curr_buffer_;
    data["last_format"] = last_format_;
    
    std::string response = sender_.post(data, "get-bitrate");
    size_t format = stoi(response);

    std::cout << "format: " << format << std::endl;

    return format;
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
