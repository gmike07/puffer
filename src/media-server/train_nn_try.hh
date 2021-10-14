#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <bits/stdc++.h>
#include <boost/algorithm/string.hpp>
#include <tuple>
#include "yaml.hh"
#include "tqdm.hpp"

class Settings
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
  Settings(std::string& yaml_path);
  std::vector<std::string> ccs = {};
  YAML::Node cc_yaml;
  std::string abr, weights_path, weights_cpp_path, scoring_type;
  double buffer_length_coef = 100.0, quality_change_qoef = 1.0, version = 1.0;
  bool predict_score, resample = true;
  int random_sample=0, sample_size = 0, prediction_size = 0, history_size = 0, input_size = 0;
  const int epochs = 20;
  const int batch_size = 16;
  std::string training_files_path = "";
};


class Loader
{
private:
    inline bool check_ends_with(const std::string& s1, const std::string& s2) const
    {
        return (s1.size() < s2.size()) ? check_ends_with(s2, s1) : s1.substr(s1.size() - s2.size()) == s2;
    }

    inline bool check_starts_with(const std::string& s1, const std::string& s2) const
    {
        return (s1.size() < s2.size()) ? check_starts_with(s2, s1) : s1.substr(s2.size()) == s2;
    }

    inline bool filter_file(const std::string& filename) const
    {
        static std::string helper_string = "abr_" + settings.abr + "_";
        //return check_ends_with(filename, ".yml");
        return check_ends_with(filename, ".txt") and filename.find(helper_string) != std::string::npos; 
    }

    void update_data_file(boost::filesystem::path& filepath, int file_index, int skip=1);

public:
    Loader(const Settings& settings_): settings(settings_)
    {
        std::vector<boost::filesystem::path> files;
        std::cout << "preparing data..." << std::endl;
        for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(settings.training_files_path), {}))
        {
            if(filter_file(entry.path().filename().string()))
            {
                files.push_back(entry.path());
            }
        }
        for(auto i: tq::trange(files.size()))
        {
            update_data_file(files[i], i);
        }
    }
    void restart();
private:
    const Settings& settings;
    std::vector<std::vector<double>> qoe_answers = {};
    std::vector<std::vector<double>> chunk_data = {};
    std::vector<std::tuple<size_t, size_t, size_t, size_t>> chunk_start_end = {};
};