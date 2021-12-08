// #ifndef PUFFER_EXP3_HH
// #define PUFFER_EXP3_HH

// #include "exp3_policy.hh"
// #include "sender.hh"
// #include "torch/script.h"

// #include <deque>

// class PufferExp3 : public Exp3Policy
// {
// public:
//   PufferExp3(const WebSocketClient & client,
//             const std::string & abr_name, const YAML::Node & abr_config);
// private:
//   static constexpr double BAN_PROB_ = 0.5;
//   static constexpr size_t TTP_INPUT_DIM = 62;
//   static constexpr size_t TTP_CURR_DIFF_POS = 5;
//   static constexpr size_t PKT_BYTES = 1500;
//   static constexpr size_t MILLION = 1000000;
//   static constexpr size_t THOUSAND = 1000;

//   double ban_prob_ {BAN_PROB_};

//   std::shared_ptr<torch::jit::script::Module> ttp_modules_[MAX_LOOKAHEAD_HORIZON];
//   std::shared_ptr<torch::jit::script::Module> hidden2_ttp_modules_[MAX_LOOKAHEAD_HORIZON];


//   /* stats of training data used for normalization */
//   std::vector<double> obs_mean_[MAX_LOOKAHEAD_HORIZON];
//   std::vector<double> obs_std_[MAX_LOOKAHEAD_HORIZON];

//   size_t ttp_input_dim_ {TTP_INPUT_DIM};
//   bool is_mle_ {false};
//   bool no_tcp_info_ {false};

//   /* preprocess the data */
//   void normalize_in_place(size_t i, std::vector<double> & input);

//   void reinit_sending_time() override;

//   bool use_puffer_ = false;
// };

// #endif /* PUFFER_EXP3_HH */
