#include <map>
#include <vector>
#include <string>
#include <set>
#include <torch/torch.h>
#include <random>
#include "torch/script.h"
#include <tuple>
#include <iterator> // For std::forward_iterator_tag
#include <cstddef>  // For std::ptrdiff_t

#include <string>
#include <iostream>
#include <filesystem>

#include "yaml.hh"

struct Settings
{
private:
  std::vector<std::string> DELETED_COLS = {"file_index", "chunk_index", 
          "max_pacing_rate", "reordering", "advmss", "pmtu", "fackets", "snd_mss", "probes"};
  std::vector<std::string> QUALITIES = {"426x240", "640x360", "854x480", "1280x720", "1920x1080"};
  std::vector<std::string> QUALITY_COLS = {"file_index", "chunk_index", "ssim", 
                                              "video_buffer", "cum_rebuffer", "media_chunk_size", "trans_time"};

  std::vector<std::string> CC_COLS = {"file_index", "chunk_index", "sndbuf_limited", "rwnd_limited", "busy_time",
            "delivery_rate", "data_segs_out", "data_segs_in", "min_rtt", "notsent_bytes",
            "segs_in", "segs_out", "bytes_received", "bytes_acked", "max_pacing_rate",
            "total_retrans", "rcv_space", "rcv_rtt", "reordering", "advmss",
            "snd_cwnd", "snd_ssthresh", "rttvar", "rtt", "rcv_ssthresh", "pmtu",
            "last_ack_recv", "last_data_recv", "last_data_sent", "fackets",
            "retrans", "lost", "sacked", "unacked", "rcv_mss", "snd_mss", "ato",
            "rto", "backoff", "probes", "ca_state", "timestamp"};
public:
  Settings(std::string& yaml_path)
  {
    abr = YAML::LoadFile(yaml_path + "abr.yml")['abr'].as<std::string>();
    auto cc_yaml = YAML::LoadFile(yaml_path + "cc.yml");
    history_size = cc_yaml['history_size'].as<int>();
    random_sample = cc_yaml['sample_size'].as<int>();

    weights_path = cc_yaml['python_weights_path'].as<std::string>();
    weights_cpp_path = cc_yaml['cpp_weights_path'].as<std::string>();
    predict_score = cc_yaml['predict_score'].as<bool>();

    for(const auto& cc: cc_yaml['ccs'])
    {
      ccs.push_back(cc.as<std::string>());
    }

    buffer_length_coef = cc_yaml['scoring_mu'].as<double>();
    quality_change_qoef = cc_yaml['scoring_lambda'].as<double>();
    scoring_type = cc_yaml['scoring_type'].as<std::string>();

    sample_size = (int) (CC_COLS.size() + ccs.size() - DELETED_COLS.size());

    input_size = history_size * random_sample * sample_size;
    prediction_size = predict_score ? 1 : 2 * ((int) QUALITY_COLS.size() - 2);
  };
  std::vector<std::string> ccs = {};
  std::string abr, weights_path, weights_cpp_path, scoring_type;
  double buffer_length_coef, quality_change_qoef, version = 1.0;
  bool predict_score, resample = true;
  const int epochs = 20;
  const int batch_size = 16;
  int random_sample, sample_size, prediction_size, history_size, input_size;
};

class Model : torch::nn::Module
{
private:
  std::vector<int64_t> network_sizes = {500, 200, 100, 80, 50, 40, 30, 20};
  std::vector<torch::nn::AnyModule> network;
  std::default_random_engine generator_;
  torch::optim::Adam *optimizer_{nullptr};
  int64_t input_size;
  int64_t output_size;
  double lr = 1e-4;
  double beta1 = 0.5;
  double beta2 = 0.999;
  double weights_decay = 1e-4;
  const int epochs = 20;

  torch::nn::Sequential model;
public:
  torch::Device device = torch::kCPU;

  Model(Settings& settings): input_size(settings.input_size), output_size(settings.prediction_size)
  {
    if(torch::cuda::is_available())
    {
      device = torch::kCUDA;
    }

    model = torch::nn::Sequential(
      torch::nn::Linear(input_size, network_sizes[0]),
      torch::nn::Functional(torch::relu),

      torch::nn::Linear(network_sizes[0], network_sizes[1]),
      torch::nn::Functional(torch::relu),

      torch::nn::Linear(network_sizes[1], network_sizes[2]),
      torch::nn::Functional(torch::relu),

      torch::nn::Linear(network_sizes[2], network_sizes[3]),
      torch::nn::Functional(torch::relu),

      torch::nn::Linear(network_sizes[3], network_sizes[4]),
      torch::nn::Functional(torch::relu), 

      torch::nn::Linear(network_sizes[4], network_sizes[5]),
      torch::nn::Functional(torch::relu),

      torch::nn::Linear(network_sizes[5], network_sizes[6]),
      torch::nn::Functional(torch::relu), 

      torch::nn::Linear(network_sizes[6], network_sizes[7]),
      torch::nn::Functional(torch::relu), 

      torch::nn::Linear(network_sizes[network_sizes.size() - 1], output_size)
    );

    model->to(device);
    optimizer_ = new torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(lr).beta1(beta1).beta2(beta2).weight_decay(weights_decay));
  }

  torch::Tensor forward(torch::Tensor x)
  {
    return model->forward(x);
  }
};


/*
 * adds random size random indexes to vector.
 * each index is up to size max
*/
void get_random_indexes(Settings& settings, std::vector<int>& indexes, int min, int max)
{
  if(settings.resample)
  {
    for(size_t i = 0; i < settings.random_sample; i++)
    {
      indexes.push_back((rand() % (max - min)) + min);
    }
    std::sort(std::begin(indexes), std::end(indexes)); 
  }
  else
  {

  }
}


class Iterator 
{
public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>;
    using pointer           = value_type*;  // or also value_type*
    using reference         = value_type&;  // or also value_type&

    Iterator(uint64_t offset, uint64_t total, std::vector<uint64_t>& permutation, 
            std::vector<std::vector<double>>& answers_vec,
            std::vector<uint64_t>& chunk_start_index,
            std::vector<uint64_t>& chunk_end_index, 
            std::vector<std::vector<double>>& chunks_vec, Settings& settings_): 
              answer_offset(offset), total_iters(total),
              perm(permutation), answers(answers_vec),
              chunk_start(chunk_start_index),
              chunk_end(chunk_end_index),
              chunks(chunks_vec), settings(settings_)
            {};

    std::vector<double> generate_history(uint64_t offset)
    {
      std::vector<double> history;
      //add the history itself
      for(int history = settings.history_size - 1; history >= 0; history--)
      {
        std::vector<int> indexes(0);
        get_random_indexes(settings, indexes, chunk_start[offset], chunk_end[offset]);
        for(auto& index: indexes)
        {
          history.insert(std::end(history), std::begin(chunks[index]), std::end(chunks[index]));
        }  
      }
      //add next cc data
      auto& next_chunk = chunk_start[offset + 1];
      for(uint64_t i = next_chunk.size() - settings.ccs.size(); i < next_chunk.size(); i++)
      {
        history.push_back(next_chunk[i]);
      }
      return history;
    }

    std::vector<double> generate_answer(std::vector<double>& answer, std::vector<double>& next_answer)
    {
      std::vector<double> nn_answer;
      nn_answer.insert(std::end(nn_answer), std::begin(answer) + 2, std::end(answer));
      nn_answer.insert(std::end(nn_answer), std::begin(next_answer) + 2, std::end(next_answer));
      return nn_answer;
    }

    reference operator*()
    {
      std::vector<std::vector<double>> batch_chunks;
      std::vector<std::vector<double>> batch_answers;
      uint64_t curr_offset = 0;
      while(batch_answers.size() != batch_size)
      {
        if(curr_offset + answer_offset + 1 > answers.size())
        {
          break;
        }
        uint64_t offset = perm[curr_offset + answer_offset];
        curr_offset += 1;
        if(offset + 1 >= answers.size())
        {
          continue;
        }
        std::vector<double>& current_answer = answers[offset];
        std::vector<double>& next_answer = answers[offset + 1];
        bool condition2 = current_answer[CHUNK_INDEX] >= settings.history_size;
        bool condition3 = current_answer[FILE_INDEX] == next_answer[FILE_INDEX];
        if((current_answer[CHUNK_INDEX] < settings.history_size) or (current_answer[FILE_INDEX] != next_answer[FILE_INDEX]))
        {
          continue;
        }
        batch_chunks.push_back(generate_history(offset));
        batch_answers.push_back(generate_answer(answer, next_answer));
      }
      current_offset = curr_offset - 1;
      if(batch_answers.size() != settings.batch_size)
      {
        return {0, 0};
      }
      return {batch_chunks, batch_answers};
    };

    pointer operator->() {return &(*(*this));};

    // Prefix increment
    Iterator& operator++() 
    { 
      answer_offset += (current_batch.first - 1);
      current_batch = {0, std::vector<std::vector<double>>(0)};
      return *this; 
    }  

    // Postfix increment
    Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }

    friend bool operator== (const Iterator& a, const Iterator& b) { return a.answer_offset == b.answer_offset; };
    friend bool operator!= (const Iterator& a, const Iterator& b) { return a.answer_offset != b.answer_offset; }; 

    size_t size() const {return total_iters;};

private:
  static const int FILE_INDEX = 0, CHUNK_INDEX = 1;
  Settings& settings;
  const int batch_size = 16;
  uint64_t current_offset;
  uint64_t answer_offset;
  uint64_t total_iters;
  std::vector<uint64_t>& perm;
  std::vector<std::vector<double>>& answers;
  std::vector<uint64_t>& chunk_start;
  std::vector<uint64_t>& chunk_end;
  std::vector<std::vector<double>>& chunks;
};


class Loader
{
private:
  bool filter_entry(std::string& entry)
  {
    return true;
    // static std::string helper_string = "abr_" + settings.abr + "_";
    // return entry.find_last_of(".txt") == 0 and entry.find(helper_string) != std::string::npos;
  }

  void update_file(std::ifstream& file_stream, int file_index, int skip=1)
  {
    int counter = 0, chunk_index = 0;
    for(std::string& line : file_stream)
    {
      bool new_run = line.find_first_of("new_run,") == 0;
      if (line == "" or new_run or line.find_first_of("audio,") == 0)
      {
        chunk_index = new_run ? 0: chunk_index;
        continue;
      }
      bool is_video = line.find_first_of("video,");
      if(is_video)
      {
        //handle video
      }else if(counter % skip == 0)
      {
        //handle cc
      }
      counter = (counter + 1) % skip;
    }
  }

public:
  Loader(std::string input_dir, Settings& settings_): settings(settings_)
  {
    std::vector<std::string> files;
    for (const auto & entry : std::filesystem::directory_iterator(path))
    {
      if(filter_entry(entry))
      {
        files.push_back(entry);
      }
    }

    

  }
  ~Loader() = default;

  void restart()
  {;
    perm = std::vector<uint64_t>(0);
    for(size_t i = 0; i < answers.size(); i++)
    {
      perm.push_back(i);
    }
    std::random_shuffle(perm.begin(), perm.end());
  };

  Iterator begin() { return Iterator(0, total_iters, perm, answers, chunk_start, chunk_end, chunks, settings); }
  Iterator end()   { return Iterator(answers.size() - 1, total_iters, perm, answers, chunk_start, chunk_end, chunks, settings); }

private:
  Settings& settings;
  uint64_t total_iters;
  std::vector<uint64_t> perm;
  std::vector<std::vector<double>> answers;
  std::vector<uint64_t> chunk_start;
  std::vector<uint64_t> chunk_end;
  std::vector<std::vector<double>> chunks;
};



int main()
{
  return 0;
}