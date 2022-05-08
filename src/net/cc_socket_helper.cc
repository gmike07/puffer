#include "cc_socket_helper.hh"
#include "algorithm"

template<typename T>
T get_attribute(YAML::Node& cc_config, std::string atribute_name, T default_value)
{
  if(cc_config && cc_config[atribute_name])
  {
    return cc_config[atribute_name].as<T>();
  }
  return default_value;
}

SocketHelper::SocketHelper(const WebSocketClient& client, TCPSocket& socket, YAML::Node& ccs, YAML::Node& cc_config, int server_id_int): client_(client), sock(socket)
{
  history_size = get_attribute(cc_config, "history_size", 40);
  sample_size = get_attribute(cc_config, "random_sample_size", 5);
  abr_time = get_attribute(cc_config, "abr_time", false);
  nn_roundup = get_attribute(cc_config, "nn_roundup", 1000);
  buffer_length_coef = get_attribute(cc_config, "buffer_length_coef", 100.0);
  quality_change_qoef = get_attribute(cc_config, "quality_change_qoef", 1.0);
  scoring_type = get_attribute<std::string>(cc_config, "scoring_function_type", "ssim");
  stateless = get_attribute<bool>(cc_config, "start_stateless", false);

  server_path = get_attribute<std::string>(cc_config, "server_path", "");
  server_id = server_id_int;
  dis_buf_length_ = std::min(dis_buf_length_, discretize_buffer(WebSocketClient::MAX_BUFFER_S));

  for (const auto & cc : ccs) {
    supported_ccs.push_back(cc.as<std::string>());
  }
  std::string cc_string = "\nsupported ccs:\n";
  for(const auto& cc: supported_ccs)
  {
    cc_string += cc + "\n";
  }
  std::cout << cc_string << std::endl;
  reinit();
}

double SocketHelper::get_qoe()
{
  if (past_chunks_.size() < 2)
  {
    return 0;
  }

  ABRAlgo::Chunk curr_chunk = past_chunks_.back();
  ABRAlgo::Chunk prev_chunk = *(past_chunks_.end() - 2);

  return get_qoe(curr_chunk.ssim, prev_chunk.ssim, curr_chunk.trans_time, curr_buffer_);
}


std::vector<double> SocketHelper::get_qoe_vector()
{
  ABRAlgo::Chunk curr_chunk = past_chunks_.back();
  ABRAlgo::Chunk prev_chunk = *(past_chunks_.end() - 2);

  double curr_ssim = ssim_db(curr_chunk.ssim);
  double change_ssim = std::fabs(ssim_db(curr_chunk.ssim) - ssim_db(prev_chunk.ssim));
  double rebuffer =  std::max(curr_chunk.trans_time * 0.001 - curr_buffer_ * UNIT_BUF_LENGTH, 0.0);
  return {curr_buffer_ / 100.0, get_normalized_qoe(), curr_ssim / MAX_SSIM, change_ssim * quality_change_qoef / MAX_SSIM, rebuffer * buffer_length_coef / 1000};
}

double SocketHelper::get_rebuffer()
{
  ABRAlgo::Chunk curr_chunk = past_chunks_.back();
  return std::max(curr_chunk.trans_time * 0.001 - curr_buffer_ * UNIT_BUF_LENGTH, 0.0);
}


double SocketHelper::get_ssim()
{
  ABRAlgo::Chunk curr_chunk = past_chunks_.back();
  return ssim_db(curr_chunk.ssim);
}


double SocketHelper::get_change_ssim()
{
  ABRAlgo::Chunk curr_chunk = past_chunks_.back();
  ABRAlgo::Chunk prev_chunk = *(past_chunks_.end() - 2);
  return std::fabs(ssim_db(curr_chunk.ssim) - ssim_db(prev_chunk.ssim));
}


std::vector<uint64_t> SocketHelper::get_tcp_full_vector()
{
    tcp_info info = sock.get_tcp_full_info();
    return 
        {info.tcpi_sndbuf_limited, info.tcpi_rwnd_limited, info.tcpi_busy_time, info.tcpi_delivery_rate,
        info.tcpi_data_segs_out, info.tcpi_data_segs_in, info.tcpi_min_rtt, info.tcpi_notsent_bytes,
        info.tcpi_segs_in, info.tcpi_segs_out, info.tcpi_bytes_received, info.tcpi_bytes_acked, 
        info.tcpi_total_retrans, info.tcpi_rcv_space, info.tcpi_rcv_rtt,
        info.tcpi_snd_cwnd, info.tcpi_snd_ssthresh, info.tcpi_rttvar,
        info.tcpi_rtt, info.tcpi_rcv_ssthresh, info.tcpi_last_ack_recv, info.tcpi_last_data_recv,
        info.tcpi_last_data_sent, info.tcpi_retrans, info.tcpi_lost, info.tcpi_sacked, 
        info.tcpi_unacked, info.tcpi_rcv_mss, info.tcpi_ato, info.tcpi_rto, 
        info.tcpi_backoff, info.tcpi_ca_state};
         // info.tcpi_reordering, info.tcpi_advmss, info.tcpi_probes, info.tcpi_pmtu, info.tcpi_fackets, info.tcpi_max_pacing_rate, info.tcpi_snd_mss - don't change
}


std::vector<double> SocketHelper::get_custom_cc_state()
{
  auto curr_tcp_info = sock.get_tcp_info();
  return {
      (double) curr_tcp_info.delivery_rate / PKT_BYTES,
      (double) curr_tcp_info.cwnd,
      (double) curr_tcp_info.in_flight,
      (double) curr_tcp_info.min_rtt / MILLION,
      (double) curr_tcp_info.rtt / MILLION,
    };
}

std::vector<double> SocketHelper::get_tcp_full_normalized_vector(uint64_t delta_time)
{
    std::vector<uint64_t> info = get_tcp_full_vector();
    info.push_back(delta_time);
    return get_tcp_full_normalized_vector(info);
}

std::vector<double> SocketHelper::get_tcp_full_normalized_vector(const std::vector<uint64_t>& vec)
{
    std::vector<double> state(0);
    for(const auto& num : vec)
    {
      state.push_back(num / (10 * MILLION));
    }
    return state;
}




int SocketHelper::get_congestion_control_index()
{
  const auto& cc = get_congestion_control();
  for(unsigned int i = 0; i < supported_ccs.size(); i++)
  {
    if(cc == supported_ccs[i])
    {
      return i;
    }
  }
  return -1;
}



void SocketHelper::init_vformats()
{
  uint64_t curr_vts = client_.next_vts().value();
  sorted_vformats_ = client_.channel()->vformats();
  std::sort(sorted_vformats_.begin(),
            sorted_vformats_.end(),
            [&](VideoFormat format1, VideoFormat format2)
            {
              return client_.channel()->vssim(curr_vts).at(format1) < client_.channel()->vssim(curr_vts).at(format2);
            });
}



double SocketHelper::get_qoe(double curr_ssim, double prev_ssim,  uint64_t curr_trans_time, std::size_t curr_buffer)
{
  double qoe = ssim_db(curr_ssim);
  qoe -= quality_change_qoef * std::fabs(ssim_db(curr_ssim) - ssim_db(prev_ssim));

  double rebuffer = buffer_length_coef * std::max(curr_trans_time * 0.001 - curr_buffer * UNIT_BUF_LENGTH, 0.0);
  qoe -= rebuffer;

  return qoe;
}



double normalize(double x, double a, double b)
{
  const static double PI = std::atan(1.0) * 4;
  //return std::min(std::max((x-a) / (b-a), 0.0), 1.0);
  return std::atan(x) / PI + 0.5;
}

double SocketHelper::get_normalized_qoe()
{
  if (past_chunks_.size() < 2)
  {
    return 0;
  }

  ABRAlgo::Chunk curr_chunk = past_chunks_.back();
  ABRAlgo::Chunk prev_chunk = *(past_chunks_.end() - 2);

  double reward = get_qoe(curr_chunk.ssim, prev_chunk.ssim, curr_chunk.trans_time, curr_buffer_);

  const VideoFormat &min_vformat = sorted_vformats_.front();
  const VideoFormat &max_vformat = sorted_vformats_.back();

  const unsigned int vduration = client_.channel()->vduration();
  uint64_t curr_vts = client_.client_next_vts().value() - vduration;
  double min_ssim = client_.channel()->vssim(curr_vts).at(min_vformat);
  double max_ssim = client_.channel()->vssim(curr_vts).at(max_vformat);

  if (!(min_ssim <= curr_chunk.ssim && curr_chunk.ssim <= max_ssim))
  {
    std::cout << "something is wrong with vts or normalization " << min_ssim << ", " << curr_chunk.ssim << ", " << max_ssim << std::endl;
  }
  
  // local normalization
  double min_reward = this->get_qoe(min_ssim, max_ssim, 10000, 0);
  double max_reward = ssim_db(max_ssim);
  return normalize(reward, min_reward, max_reward);
}


size_t SocketHelper::discretize_buffer(double buf)
{
  return (buf + UNIT_BUF_LENGTH * 0.5) / UNIT_BUF_LENGTH;
}


void SocketHelper::init_ssim(const std::shared_ptr<Channel>& channel, const std::vector<VideoFormat> & vformats, const unsigned int vduration, const uint64_t next_ts)
{
  /* init curr_ssims */
  if (past_chunks_.size() > 0)
  {
    is_init_ = false;
    curr_ssims_[0][0] = ssim_db(past_chunks_.back().ssim);
  }
  else
  {
    is_init_ = true;
  }

  for (size_t i = 1; i <= lookahead_horizon_; i++)
  {
    for (size_t j = 0; j < num_formats_; j++)
    {
      try
      {
        curr_ssims_[i][j] = ssim_db(channel->vssim(vformats[j], next_ts + vduration * (i - 1)));
      }
      catch (const std::exception &e)
      {
        std::cerr << "Error occurs when getting the ssim of " << next_ts + vduration * (i - 1) << " " << vformats[j] << std::endl;
        curr_ssims_[i][j] = MIN_SSIM;
      }
    }
  }
}

void SocketHelper::reinit()
{
  const auto &channel = client_.channel();
  const auto &vformats = channel->vformats();
  const unsigned int vduration = channel->vduration();
  const uint64_t next_ts = client_.next_vts().value();

  dis_chunk_length_ = discretize_buffer((double)vduration / channel->timescale());
  num_formats_ = vformats.size();

  /* initialization failed if there is no ready chunk ahead */
  if (channel->vready_frontier().value() < next_ts || num_formats_ == 0)
  {
    throw std::runtime_error("no ready chunk ahead");
  }

  lookahead_horizon_ = std::min(max_lookahead_horizon_, (channel->vready_frontier().value() - next_ts) / vduration + 1);

  curr_buffer_ = std::min(dis_buf_length_, discretize_buffer(client_.video_playback_buf()));

  init_ssim(channel, vformats, vduration, next_ts);
  int vformats_size = (int)num_formats_;
  actions_ = {-(vformats_size - 1), -(int)floor(sqrt(vformats_size - 1)), -1, 0, 1, (int)floor(sqrt(vformats_size - 1)), vformats_size - 1};

  init_vformats();

  auto it = find(sorted_vformats_.begin(), sorted_vformats_.end(), vformats.at(last_format_idx_));
  last_sorted_format_idx_ = it - sorted_vformats_.begin();
}


void SocketHelper::add_chunk(ABRAlgo::Chunk &&c)
{
  past_chunks_.push_back(c);
  if (past_chunks_.size() > MAX_NUM_PAST_CHUNKS)
  {
    past_chunks_.pop_front();
  }
  reinit();
}

void SocketHelper::select_new_cc(double new_ssim)
{
  if(past_chunks_.size() <= 1)
  {
    return;
  }
  chosen_ssim = ssim_db(new_ssim);
  is_new_chunk_scoring = true;
  is_new_chunk_model = true;
}