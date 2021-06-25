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

  if (abr_config["use_puffer"]) {
    training_mode_ = abr_config["use_puffer"].as<bool>();
  }

  if (abr_config["exp3_dir"]) {
    exp3_agent_ = Exp3(abr_config["exp3_dir"].as<string>());
  }

  dis_buf_length_ = min(dis_buf_length_,
                        discretize_buffer(WebSocketClient::MAX_BUFFER_S));
}

double Exp3Policy::calc_qoe()
{
  if (past_chunks_.size() < 2) {
    return 0;
  }

  Chunk curr_chunk = past_chunks_.back();
  Chunk prev_chunk = *(past_chunks_.end()-2);

  double qoe = ssim_db(curr_chunk.ssim);
  qoe -= ssim_diff_coeff_ * fabs(ssim_db(curr_chunk.ssim) - ssim_db(prev_chunk.ssim));

  double rebuffer = max(curr_chunk.trans_time*0.001 - curr_buffer_*unit_buf_length_, 0.0);
  qoe -= rebuffer_length_coeff_ * rebuffer;
  
  // std::cout <<  "rebuffer " << rebuffer << ",tans: " << curr_chunk.trans_time << "curr: " << curr_buffer_*unit_buf_length_ << ",cum: "<< client_.cum_rebuffer() << std::endl;

  return qoe;
}

void Exp3Policy::video_chunk_acked(Chunk && c)
{
  past_chunks_.push_back(c);
  if (past_chunks_.size() > max_num_past_chunks_) {
    past_chunks_.pop_front();
  }

  std::thread([&](){ 
    std::vector<double> last_input = inputs_.front();
    auto [buffer, last_format, format] = last_buffer_formats_.front();
    double qoe = this->calc_qoe();
    
    json data;
    data["datapoint"] = last_input;
    data["buffer_size"] = buffer;
    data["last_format"] = last_format;
    data["arm"] = format;
    data["reward"] = qoe;
    
    sender_.post(data, "update"); 

    inputs_.pop_front();
    last_buffer_formats_.pop_front();
  }).detach();
}

VideoFormat Exp3Policy::select_video_format()
{
  auto before_ts = timestamp_ms();
  reinit();
  
  size_t format = exp3_agent_.predict(inputs_.front()); //this->get_bitrate();
  last_buffer_formats_.push_back(std::tuple<size_t,size_t,size_t>{curr_buffer_, last_format_, format});
  last_format_ = format;

  auto after = timestamp_ms() - before_ts;
  // std::cout <<  "diff time:" << after << std::endl;


  return client_.channel()->vformats()[9]; //todo: change to format 
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
