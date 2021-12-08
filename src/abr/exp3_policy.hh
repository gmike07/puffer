// #ifndef EXP3_POLICY_HH
// #define EXP3_POLICY_HH

// #include "abr_algo.hh"
// #include <deque>
// #include <vector>
// #include "filesystem.hh"
// #include "sender.hh"
// #include "exp3.hh"
// #include "exp3_kmeans.hh"
// #include "exp3_boggart.hh"

// class Exp3Policy : public ABRAlgo
// {
// public:
//   Exp3Policy(const WebSocketClient &client,
//              const std::string &abr_name, const YAML::Node &abr_config);
//   ~Exp3Policy();

//   void video_chunk_acked(Chunk &&c) override;
//   VideoFormat select_video_format() override;

// protected:
//   static constexpr size_t MAX_NUM_PAST_CHUNKS = 8;
//   static constexpr size_t MAX_LOOKAHEAD_HORIZON = 5;
//   static constexpr size_t MAX_DIS_BUF_LENGTH = 100;
//   static constexpr double REBUFFER_LENGTH_COEFF = 20;
//   static constexpr double SSIM_DIFF_COEFF = 1;
//   static constexpr size_t MAX_NUM_FORMATS = 20;
//   static constexpr double UNIT_BUF_LENGTH = 0.5;
//   static constexpr size_t MAX_DIS_SENDING_TIME = 20;
//   static constexpr double ST_PROB_EPS = 1e-5;
//   static constexpr size_t HIDDEN_SIZE = 64;
//   static constexpr double BINS[MAX_DIS_SENDING_TIME + 2] = {0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 30};
//   static constexpr size_t NUM_OF_ACTIONS = 7;
//   static constexpr size_t PKT_BYTES = 1500;

//   /* past chunks and max number of them */
//   size_t max_num_past_chunks_{MAX_NUM_PAST_CHUNKS};
//   std::deque<Chunk> past_chunks_{};

//   /* all the time durations are measured in sec */
//   size_t max_lookahead_horizon_{MAX_LOOKAHEAD_HORIZON};
//   size_t lookahead_horizon_{};
//   size_t dis_chunk_length_{};
//   size_t dis_buf_length_{MAX_DIS_BUF_LENGTH};
//   size_t dis_sending_time_{MAX_DIS_SENDING_TIME};
//   double unit_buf_length_{UNIT_BUF_LENGTH};
//   size_t num_formats_{};
//   double rebuffer_length_coeff_{REBUFFER_LENGTH_COEFF};
//   double ssim_diff_coeff_{SSIM_DIFF_COEFF};
//   double st_prob_eps_{ST_PROB_EPS};

//   /* whether the current chunk is the first chunk */
//   bool is_init_{};

//   /* for the current buffer length */
//   size_t curr_buffer_{};

//   /* for storing the value function */
//   uint64_t flag_[MAX_LOOKAHEAD_HORIZON + 1][MAX_DIS_BUF_LENGTH + 1][MAX_NUM_FORMATS]{};
//   double v_[MAX_LOOKAHEAD_HORIZON + 1][MAX_DIS_BUF_LENGTH + 1][MAX_NUM_FORMATS]{};

//   /* record the current round of DP */
//   uint64_t curr_round_{};

//   /* the ssim and size of the chunk given the timestamp and format */
//   double curr_ssims_[MAX_LOOKAHEAD_HORIZON + 1][MAX_NUM_FORMATS]{};
//   int curr_sizes_[MAX_LOOKAHEAD_HORIZON + 1][MAX_NUM_FORMATS]{};

//   /* the estimation of sending time given the timestamp and format */
//   double sending_time_prob_[MAX_LOOKAHEAD_HORIZON + 1][MAX_NUM_FORMATS]
//                            [HIDDEN_SIZE]{};

//   /* denote whether a chunk is abandoned */
//   bool is_ban_[MAX_LOOKAHEAD_HORIZON + 1][MAX_NUM_FORMATS]{};

//   void reinit();
//   virtual void reinit_sending_time(){};

//   /* discretize the buffer length */
//   size_t discretize_buffer(double buf);
//   void deal_all_ban(size_t i);

//   // my code
//   bool use_boggart_ = false;
//   bool training_mode_ = false;

//   Sender sender_{};
//   uint64_t curr_ack_round_{};
//   std::deque<std::tuple<size_t, size_t>> past_datapoints_;

//   double get_qoe(double curr_ssim,
//                  double prev_ssim,
//                  uint64_t curr_trans_time,
//                  std::size_t curr_buffer);
//   double normalize_reward();

//   Exp3 *exp3_agent_;
//   std::size_t last_format_idx_;
//   std::deque<std::vector<double>> inputs_;

//   std::tuple<std::size_t, std::size_t> find_optimal_actions(std::vector<std::size_t> available_actions_idx);
//   std::array<int, NUM_OF_ACTIONS> actions_;

//   std::vector<VideoFormat> sorted_vformats_{};
//   std::size_t last_sorted_format_idx_;
// };

// #endif /* EXP3_POLICY_HH */
