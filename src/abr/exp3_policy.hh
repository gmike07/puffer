#ifndef EXP3_POLICY_HH
#define EXP3_POLICY_HH

#include "abr_algo.hh"
#include <deque>
#include <vector>
#include "filesystem.hh"
#include "sender.hh"
#include "exp3.hh"

class Exp3Policy : public ABRAlgo
{
public:
  Exp3Policy(const WebSocketClient & client,
         const std::string & abr_name, const YAML::Node & abr_config);

  void video_chunk_acked(Chunk && c) override;
  VideoFormat select_video_format() override;

protected:
  static constexpr size_t MAX_NUM_PAST_CHUNKS = 8;
  static constexpr size_t MAX_LOOKAHEAD_HORIZON = 5;
  static constexpr size_t MAX_DIS_BUF_LENGTH = 100;
  static constexpr double REBUFFER_LENGTH_COEFF = 20;
  static constexpr double SSIM_DIFF_COEFF = 1;
  static constexpr size_t MAX_NUM_FORMATS = 20;
  static constexpr double UNIT_BUF_LENGTH = 0.5;
  static constexpr size_t MAX_DIS_SENDING_TIME = 20;
  static constexpr double ST_PROB_EPS = 1e-5;
  static constexpr size_t HIDDEN_SIZE = 64;

  /* past chunks and max number of them */
  size_t max_num_past_chunks_ {MAX_NUM_PAST_CHUNKS};
  std::deque<Chunk> past_chunks_ {};

  /* all the time durations are measured in sec */
  size_t max_lookahead_horizon_ {MAX_LOOKAHEAD_HORIZON};
  size_t lookahead_horizon_ {};
  size_t dis_chunk_length_ {};
  size_t dis_buf_length_ {MAX_DIS_BUF_LENGTH};
  size_t dis_sending_time_ {MAX_DIS_SENDING_TIME};
  double unit_buf_length_ {UNIT_BUF_LENGTH};
  size_t num_formats_ {};
  double rebuffer_length_coeff_ {REBUFFER_LENGTH_COEFF};
  double ssim_diff_coeff_ {SSIM_DIFF_COEFF};
  double st_prob_eps_ {ST_PROB_EPS};

  /* whether the current chunk is the first chunk */
  bool is_init_ {};

  /* for the current buffer length */
  size_t curr_buffer_ {};

  /* for storing the value function */
  uint64_t flag_[MAX_LOOKAHEAD_HORIZON + 1][MAX_DIS_BUF_LENGTH + 1][MAX_NUM_FORMATS] {};
  double v_[MAX_LOOKAHEAD_HORIZON + 1][MAX_DIS_BUF_LENGTH + 1][MAX_NUM_FORMATS] {};

  /* record the current round of DP */
  uint64_t curr_round_ {};

  /* the ssim and size of the chunk given the timestamp and format */
  double curr_ssims_[MAX_LOOKAHEAD_HORIZON + 1][MAX_NUM_FORMATS] {};
  int curr_sizes_[MAX_LOOKAHEAD_HORIZON + 1][MAX_NUM_FORMATS] {};

  /* the estimation of sending time given the timestamp and format */
  double sending_time_prob_[MAX_LOOKAHEAD_HORIZON + 1][MAX_NUM_FORMATS]
                           [HIDDEN_SIZE] {};

  /* denote whether a chunk is abandoned */
  bool is_ban_[MAX_LOOKAHEAD_HORIZON + 1][MAX_NUM_FORMATS] {};

  void reinit();
  virtual void reinit_sending_time() {};

  /* discretize the buffer length */
  size_t discretize_buffer(double buf);

  bool training_mode_ = false;
  bool use_puffer_ = false;
  Sender sender_ {};
  size_t last_format_;

  std::deque<std::tuple<size_t,size_t,size_t>> last_buffer_formats_;
  std::deque<std::vector<double>> inputs_;
  std::vector<double> hidden2_;

  double calc_qoe();

  Exp3 exp3_agent_ {};
  std::size_t version_;
};

#endif /* EXP3_POLICY_HH */
