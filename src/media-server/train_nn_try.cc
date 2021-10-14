#include "train_nn_try.hh"

namespace po = boost::program_options;

inline std::vector<double> normalize_vector(const std::vector<uint64_t>& vec)
{
    static constexpr double million = 1000000;
    static constexpr double pkt_bytes = 1500;
    return
    {
        vec[0] / (10 * million), vec[1] / (10 * million), 
        vec[2] / (1000 * million), vec[3] / (10 * million),
        vec[4] / (16 * pkt_bytes), vec[5] / (16 * pkt_bytes), 
        vec[6] / (10 * million), vec[7] / (1024 *  pkt_bytes),
        vec[8] / (16 * pkt_bytes), vec[9] / (16 * pkt_bytes), 
        vec[10] / (1024 *  pkt_bytes), vec[11] / (16 * 1024 *  pkt_bytes), 
        vec[12] / 100.0,vec[13] / (16 * pkt_bytes), 
        vec[14] / (10 * million), vec[15] / (16 * pkt_bytes), 
        vec[16] / (1000 * 1024 * pkt_bytes), vec[17] / (10 * million),
        vec[18] / (10 * million), vec[19] / (1000 * 16 * pkt_bytes), 
        vec[20] / (16 * pkt_bytes), vec[21] / (16 * pkt_bytes),
        vec[22] / (16 * pkt_bytes), vec[23] / 1024.0, 
        vec[24] / 1024.0, vec[25] / 1024.0, 
        vec[26] / 1024.0, vec[27] / (16 * pkt_bytes), 
        vec[28] / (10 * million), vec[29] / (10 * million), 
        vec[30] / 1.0, vec[31] / 4.0,
        vec[32] / (10 * million)
    };
}
std::string get_directory(std::string path)
{
  for(int i = path.size() - 1; i >= 0; i--)
  {
    if(path[i] == '/')
    {
      return path.substr(0, i);
    }
  }
  return "";
}

Settings::Settings(std::string& yaml_path): cc_yaml(YAML::LoadFile(yaml_path + "cc.yml")),
  abr(YAML::LoadFile(yaml_path + "abr.yml")["abr"].as<std::string>()), 
  weights_path(cc_yaml["python_weights_path"].as<std::string>()),
  weights_cpp_path(cc_yaml["cpp_weights_path"].as<std::string>()),
  scoring_type(cc_yaml["scoring_type"].as<std::string>()),
  predict_score(cc_yaml["predict_score"].as<bool>()), resample(true),
  random_sample(cc_yaml["sample_size"].as<int>()),
  history_size(cc_yaml["history_size"].as<int>())
  {

    for(const auto& cc: cc_yaml["ccs"])
    {
      ccs.push_back(cc.as<std::string>());
    }

    buffer_length_coef = cc_yaml["scoring_mu"].as<double>();
    quality_change_qoef = cc_yaml["scoring_lambda"].as<double>();
    scoring_type = cc_yaml["scoring_type"].as<std::string>();

    sample_size = (int) (CC_COLS.size() + ccs.size() - DELETED_COLS.size());

    input_size = history_size * random_sample * sample_size;
    prediction_size = predict_score ? 1 : 2 * ((int) QUALITY_COLS.size() - 2);
    training_files_path = get_directory(cc_yaml["cc_monitoring_path"].as<std::string>());
  }


void Loader::update_data_file(boost::filesystem::path& filepath, int file_index, int skip)
{
    //std::cout << filepath << std::endl;
    boost::filesystem::ifstream file_reader;
    file_reader.open(filepath);
    if(not file_reader.is_open())
    {
        return;
    }
    size_t chunk_index = 0, counter = 0, prev_offset = 0;

    for(std::string line; std::getline(file_reader, line);)
    {   
        if((line == "") or (line == "new run,") or check_starts_with(line, "audio,"))
        {
            chunk_index = (line == "new run,") ? 0 : chunk_index;
            continue;
        }
        std::vector<std::string> result;
        boost::split(result, line, boost::is_any_of(","));
        if(check_starts_with(line, "video,"))
        {
            //result = {video, ssim, video_buffer, cum_rebuffer, media_size, trans_time}
            qoe_answers.push_back({(double) file_index, (double) chunk_index, 
                                    std::stod(result[1]), std::stod(result[2]), std::stod(result[3]), 
                                    std::stod(result[4]) / 100000.0, std::stod(result[5]) / 1000.0});
            // {file_index, chunk_index, start_chunk, end_chunk}
            chunk_start_end.push_back({file_index, chunk_index, prev_offset, chunk_data.size()});
            prev_offset = chunk_data.size();
            chunk_index++;
        }else
        {
            if(counter % skip == 0)
            {
                std::vector<uint64_t> cc_sample_temp;
                for(size_t i = 1; i < result.size(); i++)
                {
                    cc_sample_temp.push_back((uint64_t)std::stoull(result[i]));
                }
                auto vec = normalize_vector(cc_sample_temp);
                std::string& cc = result[0];
                for(size_t i = 0; i < settings.ccs.size(); i++)
                {
                    vec.push_back((settings.ccs[i] == cc) ? 1.0 : 0.0);
                }
                chunk_data.push_back(vec);
            }
            counter += 1;
        }
    }
    file_reader.close();
}


template <typename T>
T get_flagged_result(po::variables_map& vm, const std::string& key, T default_value)
{
    return vm.count(key) ? vm[key].as<T>() : default_value;
}

int main(int argc, char* argv[])
{
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    ("input_dir", po::value<std::string>(), "the path to the yaml dir")
    ("abr", po::value<std::string>(), "the abr to train if not the default");
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help")) 
    {
        std::cout << desc << std::endl;
        return 1;
    }

    std::string input_dir = get_flagged_result(vm, "input_dir", "/home/mike/puffer/helper_scripts/");
    std::string abr = get_flagged_result(vm, "abr", "");

    Settings settings(input_dir);
    if(abr != "")
    {
      settings.abr = abr;
    }
    Loader loader(settings);
    return 0;
}