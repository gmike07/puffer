#ifndef CC_SOCKET_HELPER_HH
#define CC_SOCKET_HELPER_HH

#include "socket.hh"
#include "abr_algo.hh"
#include "algorithm"
#include "ws_client.hh"

class SocketHelper
{
private:
  static constexpr size_t MAX_NUM_PAST_CHUNKS = 8;
  static constexpr double UNIT_BUF_LENGTH = 0.5;
  static constexpr size_t MAX_LOOKAHEAD_HORIZON = 5;
  static constexpr size_t NUM_OF_ACTIONS = 7;
  static constexpr size_t MAX_NUM_FORMATS = 20;
  static constexpr size_t MAX_DIS_BUF_LENGTH = 100;
  static constexpr size_t HIDDEN_SIZE = 64;
  static constexpr size_t MAX_DIS_SENDING_TIME = 20;
  static constexpr double BINS[MAX_DIS_SENDING_TIME + 2] = {0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 30};

  std::vector<VideoFormat> sorted_vformats_{};
  std::deque<ABRAlgo::Chunk> past_chunks_{};
  const WebSocketClient & client_;
  /* for the current buffer length */
  size_t curr_buffer_ = 0;
  std::size_t last_format_idx_ = 0;
  std::array<int, NUM_OF_ACTIONS> actions_{};
  /* the ssim and size of the chunk given the timestamp and format */
  double curr_ssims_[MAX_LOOKAHEAD_HORIZON + 1][MAX_NUM_FORMATS]{0};
  /* whether the current chunk is the first chunk */
  bool is_init_ = false;
  /* the estimation of sending time given the timestamp and format */
  double sending_time_prob_[MAX_LOOKAHEAD_HORIZON + 1][MAX_NUM_FORMATS]
                           [HIDDEN_SIZE]{0};
  std::size_t last_sorted_format_idx_ = 0;


  /* all the time durations are measured in sec */
  size_t max_lookahead_horizon_{MAX_LOOKAHEAD_HORIZON};
  size_t lookahead_horizon_= 0;
  size_t dis_chunk_length_= 0;
  size_t dis_buf_length_{MAX_DIS_BUF_LENGTH};
  size_t dis_sending_time_{MAX_DIS_SENDING_TIME};
  size_t num_formats_ = 0;

private:
  std::vector<std::string> supported_ccs{};
  //scoring data
  std::string scoring_type = "ssim";
  std::vector<std::string> scoring_types = {"ssim"}; //"bit_rate"
  TCPSocket& sock;

  static constexpr double MILLION = 1000000;
  static constexpr double PKT_BYTES = 1500;

public:
  int server_id = -1;
  //finished initializing all variables

  //new chunk booleans
  bool is_new_chunk_scoring = false;
  bool is_new_chunk_model = false;
  double chosen_ssim = 0.0;

  //paths
  std::string scoring_path = "";
  std::string server_path = "";


  //data for model
  int history_size = 40;
  int sample_size = 7;
  bool abr_time = false;
  int nn_roundup = 1000;

  bool stateless = false;

  double quality_change_qoef = 1.0;
  double buffer_length_coef = 1.0;
  double get_qoe(double curr_ssim, double prev_ssim,  uint64_t curr_trans_time, std::size_t curr_buffer);

  void init_vformats();

  void reinit();

  void init_ssim(const std::shared_ptr<Channel>& channel, const std::vector<VideoFormat> & vformats, const unsigned int vduration, const uint64_t next_ts);

  size_t discretize_buffer(double buf);

  std::vector<uint64_t> get_tcp_full_vector();

  std::vector<double> get_tcp_full_normalized_vector(const std::vector<uint64_t>& vec);

public:
  SocketHelper(const WebSocketClient& client, TCPSocket& socket, YAML::Node& ccs, YAML::Node& cc_config, int server_id_int);  

  SocketHelper(const SocketHelper&)=delete;
  SocketHelper& operator=(const SocketHelper&)=delete;

  void add_chunk(ABRAlgo::Chunk &&c);

  double get_normalized_qoe();

  std::vector<double> get_qoe_vector();

  std::vector<double> get_tcp_full_normalized_vector(uint64_t delta_time);

  std::vector<double> get_custom_cc_state();

  double get_qoe();

  bool is_valid_score_type() const
  {
      return std::find(scoring_types.begin(), scoring_types.end(), scoring_type) != scoring_types.end();
  }

  std::vector<std::string>& get_supported_cc() {return supported_ccs;}

  std::string get_congestion_control(){return sock.get_congestion_control();}

  void set_congestion_control(const std::string & cc){sock.set_congestion_control(cc);}

  void finish_creating_socket() {sock.created_socket = true;}

  double get_change_ssim();

  double get_ssim();

  double get_rebuffer();

  int get_congestion_control_index();

  void select_new_cc(double new_ssim);

};

#endif /* CC_SOCKET_HELPER_HH */
