// #include "puffer_raw.hh"
// #include "ws_client.hh"

// using namespace std;

// PufferRaw::PufferRaw(const WebSocketClient & client,
//                      const string & abr_name, const YAML::Node & abr_config)
//   : Puffer(client, abr_name, abr_config)
// {
//   if (abr_config["st_var_coeff"]) {
//     st_var_coeff_ = abr_config["st_var_coeff"].as<double>();
//   }
// }

// void PufferRaw::reinit_sending_time()
// {
//   static double unit_st[MAX_LOOKAHEAD_HORIZON + 1 + MAX_NUM_PAST_CHUNKS];
//   static double st_prob[MAX_DIS_SENDING_TIME + 1];

//   size_t num_past_chunks = past_chunks_.size();
//   auto it = past_chunks_.begin();

//   for (size_t i = 1; it != past_chunks_.end(); it++, i++) {
//     unit_st[i] = (double) it->trans_time / it->size / 1000;
//   }

//   for (size_t i = 1; i <= lookahead_horizon_; i++) {
//     double tmp = 0;
//     for (size_t j = 0; j < num_past_chunks; j++) {
//       tmp += unit_st[i + j];
//     }

//     if (num_past_chunks != 0) {
//       unit_st[i + num_past_chunks] = tmp / num_past_chunks;
//     } else {
//       /* set the sending time to be a default hight value */
//       unit_st[i + num_past_chunks] = HIGH_SENDING_TIME;
//     }

//     double st;
//     bool is_all_ban = true;

//     for (size_t j = 0; j < num_formats_; j++) {
//       if (curr_sizes_[i][j] > 0) {
//         st = curr_sizes_[i][j] * unit_st[i + num_past_chunks];
//       } else {
//         is_ban_[i][j] = true;
//         continue;
//       }

//       size_t dis_st = min(discretize_buffer(st), dis_sending_time_);
//       if (dis_st == dis_sending_time_) {
//         is_ban_[i][j] = true;
//         continue;
//       } else {
//         is_ban_[i][j] = false;
//         is_all_ban = false;
//       }

//       for (size_t k = 0; k <= dis_sending_time_; k++) {
//         st_prob[k] = 0;
//       }

//       double tmp = 1;
//       st_prob[dis_st] = 1;
//       for (size_t k = 1; dis_st + k <= dis_sending_time_ and dis_st >= k; k++) {
//         st_prob[dis_st + k] = st_prob[dis_st + k - 1] * st_var_coeff_;
//         st_prob[dis_st - k] = st_prob[dis_st - k + 1] * st_var_coeff_;

//         if (st_prob[dis_st + k] < st_prob_eps_) {
//           st_prob[dis_st + k] = 0;
//         }

//         if (st_prob[dis_st - k] < st_prob_eps_) {
//           st_prob[dis_st - k] = 0;
//         }

//         tmp += st_prob[dis_st + k] + st_prob[dis_st - k];
//       }

//       for (size_t k = 0; k <= dis_sending_time_; k++) {
//         sending_time_prob_[i][j][k] = st_prob[k] / tmp;
//       }
//     }

//     if (is_all_ban) {
//       deal_all_ban(i);
//     }
//   }
// }
