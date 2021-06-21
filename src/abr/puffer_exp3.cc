#include "puffer_exp3.hh"
#include "ws_client.hh"


using namespace std;

PufferExp3::PufferExp3(const WebSocketClient & client,
                     const string & abr_name, const YAML::Node & abr_config)
  : Exp3Policy(client, abr_name, abr_config)
{
  
  /* load neural networks */
  if (abr_config["model_dir"]) {
    fs::path model_dir = abr_config["model_dir"].as<string>();

    for (size_t i = 0; i < max_lookahead_horizon_; i++) {
      // load PyTorch models
      string model_path = model_dir / ("cpp-" + to_string(i) + ".pt");
      ttp_modules_[i] = torch::jit::load(model_path.c_str());
      if (not ttp_modules_[i]) {
        throw runtime_error("Model " + model_path + " does not exist");
      }

      // load normalization weights
      ifstream ifs(model_dir / ("cpp-meta-" + to_string(i) + ".json"));
      json j = json::parse(ifs);

      obs_mean_[i] = j.at("obs_mean").get<vector<double>>();
      obs_std_[i] = j.at("obs_std").get<vector<double>>();
    }

    if (abr_name == "puffer_ttp_mle") {
      is_mle_= true;
    }

    if (abr_name == "puffer_ttp_no_tcp_info") {
      ttp_input_dim_ = 17;
      no_tcp_info_ = true;
    }
  } else {
    throw runtime_error("Puffer requires specifying model_dir in abr_config");
  }
  std::cout << "Finish cons. " << max_lookahead_horizon_ << std::endl;

}

void PufferExp3::normalize_in_place(size_t i, vector<double> & input)
{
  assert(input.size() == obs_mean_[i].size());
  assert(input.size() == obs_std_[i].size());

  for (size_t j = 0; j < input.size(); j++) {
    input[j] -= obs_mean_[i][j];

    if (obs_std_[i][j] != 0) {
      input[j] /= obs_std_[i][j];
    }
  }
}

void PufferExp3::reinit_sending_time()
{
  /* prepare the raw inputs for ttp */
  const auto & curr_tcp_info = client_.tcp_info().value();
  vector<double> raw_input;

  size_t num_past_chunks = past_chunks_.size();

  if (num_past_chunks == 0) {
    for (size_t i = 0; i < max_num_past_chunks_; i++) {
      if (not no_tcp_info_) {
        raw_input.insert(raw_input.end(), {
          (double) curr_tcp_info.delivery_rate / PKT_BYTES,
          (double) curr_tcp_info.cwnd,
          (double) curr_tcp_info.in_flight,
          (double) curr_tcp_info.min_rtt / MILLION,
          (double) curr_tcp_info.rtt / MILLION,
        });
      }
      raw_input.insert(raw_input.end(), {0, 0});
    }
  } else {
    auto it = past_chunks_.begin();
    for (size_t i = 0; i < max_num_past_chunks_; i++) {
      if (not no_tcp_info_) {
        raw_input.insert(raw_input.end(), {
          (double) it->delivery_rate / PKT_BYTES,
          (double) it->cwnd,
          (double) it->in_flight,
          (double) it->min_rtt / MILLION,
          (double) it->rtt / MILLION,
        });
      }

      raw_input.insert(raw_input.end(), {
        (double) it->size / PKT_BYTES,
        (double) it->trans_time / THOUSAND,
      });

      if (i + num_past_chunks >= max_num_past_chunks_) {
        it++;
      }
    }
  }

  if (not no_tcp_info_) {
    raw_input.insert(raw_input.end(), {
      (double) curr_tcp_info.delivery_rate / PKT_BYTES,
      (double) curr_tcp_info.cwnd,
      (double) curr_tcp_info.in_flight,
      (double) curr_tcp_info.min_rtt / MILLION,
      (double) curr_tcp_info.rtt / MILLION,
    });
  }
  raw_input.insert(raw_input.end(), {0});

  assert(raw_input.size() == ttp_input_dim_);

  for (size_t i = 1; i <= lookahead_horizon_; i++) {
    /* prepare the inputs for each ahead timestamp and format */
    static double inputs[MAX_NUM_FORMATS * TTP_INPUT_DIM];

    for (size_t j = 0; j < num_formats_; j++) {
      raw_input[ttp_input_dim_ - 1] = (double) curr_sizes_[i][j] / PKT_BYTES;
      vector<double> norm_input {raw_input};

      normalize_in_place(i - 1, norm_input);
      for (size_t k = 0; k < ttp_input_dim_; k++) {
        inputs[j * ttp_input_dim_ + k] = norm_input[k];
      }
    }

    /* feed in the input batch and get the output batch */
    vector<torch::jit::IValue> torch_inputs;

    torch_inputs.push_back(torch::from_blob(inputs,
                           {(int) num_formats_, (int)ttp_input_dim_},
                           torch::kF64));

    at::Tensor output = ttp_modules_[i-1]->forward(torch_inputs).toTensor();

    // convert to vector
    inputs_.insert(inputs_.begin() + (i-1) * num_formats_ * ttp_input_dim_, inputs, inputs + num_formats_*ttp_input_dim_);

    // for (size_t j = 0; j < num_formats_; j++) {
    //   for (size_t k = 0; k < 22; k++) {
    //     hidden2_.push_back(output[j][k].item<double>());
    //   }
    // }
  }
}
