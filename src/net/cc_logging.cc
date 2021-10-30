/* -*-mode:c++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
#include <map>
#include <set>
#include <functional>
#include <deque>
#include <iostream>
#include <thread>
#include <chrono>

#include <stdexcept>
#include <crypto++/sha.h>
#include <crypto++/hex.h>
#include <crypto++/base64.h>
#include <string>
#include <fstream>
#include <torch/torch.h>
#include "torch/script.h"
#include <memory>
#include <tuple>
#include <algorithm>
#include <random>
#include <iterator>
#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/Infos.hpp>

#include "http_response.hh"
#include "exception.hh"
#include "socket.hh"
#include "nb_secure_socket.hh"
#include "poller.hh"
#include "address.hh"
#include "http_request_parser.hh"
#include "ws_message_parser.hh"
#include "cc_logging.hh"

#include "json.hpp"

using json = nlohmann::json;

/* nanoseconds per millisecond */
static const uint64_t MILLION = 1000000;

/* nanoseconds per second */
static const uint64_t BILLION = 1000 * MILLION;

inline ChunkInfo normalize_chunk(std::vector<double> chunk)
{
  return {true, 60.0 * chunk[0], chunk[1], chunk[2], (unsigned int) (chunk[3] * 100 * 100000.0),
          (uint64_t) (chunk[4] * 1000), ""};
}

double score_nn_prediction(TCPSocket& socket, const torch::Tensor& preds)
{
  if(socket.predict_score)
  {
    return preds.item<double>();
  }
  ChunkInfo prev_chunk = normalize_chunk({preds[0].item<double>(), preds[1].item<double>(),
                                          preds[2].item<double>(), preds[3].item<double>(),
                                          preds[4].item<double>()});
  ChunkInfo curr_chunk = normalize_chunk({preds[5].item<double>(), preds[6].item<double>(),
                                          preds[7].item<double>(), preds[8].item<double>(),
                                          preds[9].item<double>()});

  return socket.score_chunks(prev_chunk, curr_chunk);
}


/*
 * returns the current time in uint64_t type
*/
inline uint64_t get_timestamp_ms()
{
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);

  const uint64_t nanos = ts.tv_sec * BILLION + ts.tv_nsec;
  return nanos / MILLION;
}

/*
 * changes the curremt cc to the given string in the socket
*/
inline void change_cc(TCPSocket& socket, std::string cc)
{
  socket.set_congestion_control(cc);
  std::cerr << "cc: " << cc << std::endl;
}

/*
 * changes the curremt cc to the given string of the corresponding index in the supported ccs
*/
inline void change_cc(TCPSocket& socket, int index)
{
  static std::vector<std::string>& cc_supported = socket.get_supported_cc();
  change_cc(socket, cc_supported[index]);
}

/*
 * changes the curremt cc to a random cc in supported ccs
*/
inline void random_cc(TCPSocket& socket)
{
  if(socket.random_cc)
  {
    change_cc(socket, rand() % socket.get_supported_cc().size());
  }
}

/*
 * adds random size random indexes to vector.
 * each index is up to size max
*/
void get_random_indexes(std::vector<int>& indexes, size_t size, int max)
{
  for(size_t i = 0; i < size; i++)
  {
    indexes.push_back(rand() % max);
  }
  std::sort(std::begin(indexes), std::end(indexes)); 
}

/*
 * adds sample_size elements from elements to sampled_elements
*/
template<typename T>
void sample_random_elements(std::vector<T>& elements, std::vector<T>& sampled_elements, size_t sample_size)
{
  std::vector<int> indexes(0);
  sampled_elements = std::vector<T>(0);
  get_random_indexes(indexes, sample_size, elements.size());
  for(size_t i = 0; i < indexes.size(); i++)
  {
    sampled_elements.push_back(elements[indexes[i]]);
  }
}

/*
* a class to store the thread's data
*/
struct LoggingChunk
{
  const int MILLISECONDS_TO_SLEEP = 1;
  const int SKIP_NN = 300;
  bool logging_file_created, scoring_file_created, model_created, server_created, first_run, abr_time;
  std::ofstream logging_file, scoring_file;
  std::shared_ptr<torch::jit::script::Module> model;
  uint64_t monitoring_start_time, start_time_nn;
  int counter = 0;

  public:
  LoggingChunk(TCPSocket& socket): 
    SKIP_NN(socket.nn_roundup),
    logging_file_created(socket.logging_path != ""), 
    scoring_file_created(socket.scoring_path != ""),
    model_created(socket.model_path != ""),
    server_created(socket.server_path != ""),
    first_run(true), abr_time(socket.abr_time),
    logging_file(std::ofstream(socket.logging_path, std::ios::out | std::ios::app)),
    scoring_file(std::ofstream(socket.scoring_path, std::ios::out | std::ios::app)),
    model(model_created ? torch::jit::load(socket.model_path) : nullptr),
    monitoring_start_time(get_timestamp_ms()), start_time_nn(monitoring_start_time),
    counter(0)
  {
  }
  ~LoggingChunk(){}
};

/*
* a class to handle storing the history for the nn model
*/
class ChunkHistory
{
public:
  ChunkHistory(TCPSocket& socket): 
    history_size(socket.history_size),
    sample_size(socket.sample_size),  curr_chunk(0),  history_chunks(0)
  {
  }

  ~ChunkHistory(){}

  /*
  * adds a cc sample to the current chunk
  */
  inline void update_chunk(std::vector<double> sample_cc)
  {
    curr_chunk.push_back(sample_cc);
  }

  /*
  * adds the current chunk to the history and updates the size of the history if needed
  */
  void push_chunk()
  {
    history_chunks.push_back(curr_chunk);
    curr_chunk = std::vector<std::vector<double>>(0);
    if(history_chunks.size() > (size_t) history_size)
    {
      history_chunks.pop_front();
    }
  }

  void push_statistic(TCPSocket& socket)
  {
    qoe_statistics.push_back(socket.get_qoe_vector());
    if(qoe_statistics.size() > (size_t) history_size)
    {
      qoe_statistics.pop_front();
    }
  }

  inline std::vector<double> normalize_statistics(const ChunkInfo& info)
  {
    return {TCPSocket::ssim_db_cc(info.ssim) / 60.0, info.video_buffer / 20.0, info.cum_rebuffer, 
        info.media_chunk_size /  (100.0 * 100000.0), info.trans_time / 1000.0};
  }

  /*
  * adds to vec a random sample of each chunk of the history
  */
  void get_sample_history(std::vector<double>& vec)
  {
    for(size_t i = 0; i < history_chunks.size(); i++)
    {
      auto& chunk = history_chunks[i];
      std::vector<std::vector<double>> sample_chunk(0);
      sample_random_elements(chunk, sample_chunk, sample_size);
      for(auto& curr_state : sample_chunk)
      {
        vec.insert(std::end(vec), std::begin(curr_state), std::end(curr_state));
      }
      auto vec_statistics = qoe_statistics[i];
      vec.insert(std::end(vec), std::begin(vec_statistics), std::end(vec_statistics));
      // auto& last_sample = chunk[chunk.size() - 1];
      // vec.push_back(last_sample[last_sample.size() - 1]); //the transmition of the own chunk
    }
  }

  /*
  * returns the history current size
  */
  inline size_t size()
  {
    return history_chunks.size();
  }

  inline std::string get_info() const
  {
    return "history size:" + std::to_string(history_size) + 
          ",\tsample size" + std::to_string((*history_chunks.begin())[0].size()) +
          ",\tnum of samples" + std::to_string(sample_size) +
          ",\tqoe length" + std::to_string((*qoe_statistics.begin()).size());
  }

private:
  const int history_size;
  const int sample_size;
  std::vector<std::vector<double>> curr_chunk;
  std::deque<std::vector<std::vector<double>>> history_chunks;
  std::deque<std::vector<double>> qoe_statistics;
};


/*
 * returns the predicted cc sample normalized
*/
inline std::vector<double> convert_tcp_info_normalized_vec(TCPSocket& socket, uint64_t start_time)
{
  std::vector<double> vec = socket.get_tcp_full_normalized_vector(get_timestamp_ms() - start_time);
  auto& cc_vec = socket.get_supported_cc();
  auto cc = socket.get_congestion_control();
  for(size_t i = 0; i < cc_vec.size(); i++)
  {
    vec.push_back(cc_vec[(int)i] == cc ? 1.0 : 0.0);
  }
  return vec;
}

/*
 * returns the predicted score for the given input
*/
double calc_score_nn(TCPSocket& socket, std::shared_ptr<torch::jit::script::Module>& model, 
                      double* state, size_t length)
{

  std::vector<torch::jit::IValue> torch_inputs;

  torch_inputs.push_back(torch::from_blob(state, {(signed long)length}, torch::kDouble).unsqueeze(0));
  
  torch::Tensor preds = model->forward(torch_inputs).toTensor().squeeze().detach();
  if(socket.predict_score)
  {
    return preds[0].item<double>() - preds[1].item<double>() - preds[2].item<double>();
  }
  ChunkInfo prev_chunk = normalize_chunk({preds[0].item<double>(), preds[1].item<double>(),
                                          preds[2].item<double>(), preds[3].item<double>(),
                                          preds[4].item<double>()});
  ChunkInfo curr_chunk = normalize_chunk({preds[5].item<double>(), preds[6].item<double>(),
                                          preds[7].item<double>(), preds[8].item<double>(),
                                          preds[9].item<double>()});

  return socket.score_chunks(prev_chunk, curr_chunk);
}

/*
 * returns an array pointer and its size to a psuedo state that can ve given the the nn
*/
std::pair<double*, size_t> create_state_from_history(TCPSocket& socket, ChunkHistory& chunk_history)
{
  size_t number_ccs = socket.get_supported_cc().size();
  std::vector<double> inputs(0);
  chunk_history.get_sample_history(inputs);
  for(size_t j = 0; j < number_ccs; j++)
  {
    inputs.push_back(0.0);
  }
  double* state = new double[inputs.size()];
  for(size_t i = 0; i < inputs.size(); i++)
  {
    state[i] = inputs[i];
  }
  return {state, inputs.size()};
}

std::pair<double*, size_t> create_nn_input(TCPSocket& socket, ChunkHistory& chunk_history)
{
  size_t number_ccs = socket.get_supported_cc().size();
  std::vector<double> inputs(0);
  chunk_history.get_sample_history(inputs);
  for(size_t j = 0; j < number_ccs; j++)
  {
    inputs.push_back(0.0);
  }

  size_t input_size = inputs.size();
  double* nn_input = new double[number_ccs * input_size];
  for(size_t i = 0; i < number_ccs; i++)
  {
    std::copy(std::begin(inputs), std::end(inputs), nn_input + (i - 1) * input_size);
    nn_input[(i - 1) * input_size + (input_size - number_ccs + i)] = 1.0;
  }
  return {nn_input, number_ccs * input_size};
}


void change_nn_cc(TCPSocket& socket, LoggingChunk& logging_chunk, ChunkHistory& chunk_history)
{
  size_t number_ccs = socket.get_supported_cc().size();
  std::pair<double*, size_t> pair = create_nn_input(socket, chunk_history);
  std::vector<torch::jit::IValue> torch_inputs;
  torch_inputs.push_back(torch::from_blob(pair.first, {(signed long) number_ccs, (signed long) (pair.second / number_ccs)}, torch::kDouble));
  torch::Tensor preds = logging_chunk.model->forward(torch_inputs).toTensor().detach();
  int best_cc = -1;
  double best_score = 0;
  for(size_t cc = 0; cc < number_ccs; cc++)
  {
    double score = score_nn_prediction(socket, preds[cc]);
    if((best_cc == -1) or (best_score < score))
    {
      best_cc = cc;
      best_score = score;
    }
  }
  delete[] pair.first;
  change_cc(socket, best_cc);
}

inline int congestion_control_index(TCPSocket& socket, std::string& cc)
{
  auto& ccs = socket.get_supported_cc();
  for(int i = 0; i < ccs.size(); i++)
  {
    if(ccs[i] == cc)
    {
      return i;
    }
  }
  return -1;
}

/*
 * handles the thread to store statistics of the cc
*/
inline void handle_monitoring(TCPSocket& socket, LoggingChunk& logging_chunk)
{
  std::string data = "";
  std::string cc = socket.get_congestion_control();
  if(socket.is_new_chunk_logging)
  {
    logging_chunk.monitoring_start_time = get_timestamp_ms();
    data = socket.generate_chunk_statistics() + "\n";
    random_cc(socket); //works only if random_cc is true
    socket.is_new_chunk_logging = false;
  }
  // auto vec = socket.get_tcp_full_vector();
  // vec.push_back(get_timestamp_ms() - logging_chunk.monitoring_start_time);
  auto vec = socket.get_tcp_full_normalized_vector(get_timestamp_ms() - logging_chunk.monitoring_start_time);
  int cc_index = congestion_control_index(socket, cc);
  int ccs_size = socket.get_supported_cc().size();

  for(int i = 0; i < ccs_size; i++)
  {
    vec.push_back((cc_index == i) ? 1 : 0);
  }

  for(const auto& val: vec)
  {
    data += std::to_string(val);
    data += ",";
  }

  logging_chunk.logging_file << data.substr(0, data.size() - 1) << std::endl;
}

/*
 * handles the thread to store statistics of the abr
*/
inline void handle_scoring(TCPSocket& socket, LoggingChunk& logging_chunk, bool is_nn)
{
  if((not socket.is_new_chunk_scoring))
  {
    return;
  }
  if((socket.scoring_type != "ssim" and socket.scoring_type != "bit_rate") or 
      (not socket.prev_chunk.is_video) or 
      (not socket.curr_chunk.is_video))
  {
    socket.is_new_chunk_scoring = false;
    return;
  }
  double score = socket.score_chunks();
  if(not socket.random_cc)
  {
    std::string s = socket.get_congestion_control() + " " + std::to_string(score);
    if(is_nn)
    {
      s = "nn " + s;
    }
    std::cout << s << std::endl;
  }
  logging_chunk.scoring_file << score << std::endl;
  socket.is_new_chunk_scoring = false;
}

bool update_history(TCPSocket& socket, LoggingChunk& logging_chunk, ChunkHistory& chunk_history)
{
  chunk_history.update_chunk(convert_tcp_info_normalized_vec(socket, logging_chunk.start_time_nn));
  logging_chunk.counter = (logging_chunk.counter + 1) % logging_chunk.SKIP_NN;
  bool change_cc_1 = (logging_chunk.abr_time and socket.is_new_chunk_model);
  socket.is_new_chunk_model = false;
  bool change_cc_2 = ((not logging_chunk.abr_time) and (logging_chunk.counter % logging_chunk.SKIP_NN == 0));
  if((not change_cc_1) and (not change_cc_2))
  {
    return false;
  }
  //should change cc
  chunk_history.push_chunk();
  chunk_history.push_statistic(socket);
  if(chunk_history.size() != ((size_t) socket.history_size))
  {
    return false;
  }
  return true;
}

/*
 * handles the thread to store statistics of the socket for the model and change the cc via the model
*/
void handle_nn_model(TCPSocket& socket, LoggingChunk& logging_chunk, ChunkHistory& chunk_history)
{ 
  if(not update_history(socket, logging_chunk, chunk_history))
  {
    return;
  }

  std::pair<double*, size_t> inputs = create_state_from_history(socket, chunk_history);
  double* state = inputs.first;
  int inputs_size = (int) (inputs.second), best_cc = -1;
  size_t number_ccs = socket.get_supported_cc().size();
  double score = 0;
  for(size_t i = 0; i < number_ccs; i++)
  {
    state[inputs_size - number_ccs + i] = 1.0;
    if(i != 0)
    {
      state[inputs_size - number_ccs + (i - 1)] = 0.0;
    }
    double current_score = calc_score_nn(socket, logging_chunk.model, state, inputs_size);
    if((best_cc == -1) or (score < current_score))
    {
      best_cc = (int)i;
      score = current_score;
    }
  }
  delete[] state;
  change_cc(socket, best_cc);
}


void switch_cc_server(TCPSocket& socket, std::vector<double>& state)
{
  json json_state(state);
  
  json data;
  data["state"] = json_state;
  data["server_id"] = socket.server_id;
  data["qoe"] = socket.score_chunks();

  // send request
  std::list<std::string> header;
  header.push_back("Content-Type: application/json");

  curlpp::Cleanup clean;
  curlpp::Easy request;
  std::ostringstream response;
  request.setOpt(new curlpp::options::Url(socket.server_path));
  request.setOpt(new curlpp::options::HttpHeader(header));
  request.setOpt(new curlpp::options::PostFields(data.dump()));
  request.setOpt(new curlpp::options::PostFieldSize(data.dump().size()));
  request.setOpt(new curlpp::options::WriteStream(&response));

  try {
    request.perform(); // 200 = ok and not enough data to switch, 409 = ok and sent json with {"cc": cc_to switch to}
    long status = curlpp::infos::ResponseCode::get(request);
    if(status == 400)
    {
      std::cout << "error" << std::endl;
    }
    if(status != 200)
    {
      change_cc(socket, (int)(status - 406));
    }
  }
  catch (std::exception& e) {
    std::cout << "exception " << e.what() << std::endl;
  }
}

void handle_server_model(TCPSocket& socket, LoggingChunk& logging_chunk, ChunkHistory& chunk_history)
{
  if(not update_history(socket, logging_chunk, chunk_history))
  {
    return;
  }
  std::vector<double> state(0);
  chunk_history.get_sample_history(state);
  switch_cc_server(socket, state);
}


/*
 * handles the thread to store statistics of the socket and change the cc
*/
void logging_cc_func(TCPSocket* socket)
{
  if(socket == nullptr)
  {
    std::cout << "oh shit, socket doesn't exists" << std::endl;
    return;
  }
  while(!socket->created_socket) {}
  TCPSocket& sock = *socket;
  LoggingChunk logging_chunk(sock);
  ChunkHistory chunk_history(sock);
  random_cc(sock); //works only if random_cc is true
  if(logging_chunk.logging_file_created)
  {
    logging_chunk.logging_file << "new run," << std::endl;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(logging_chunk.MILLISECONDS_TO_SLEEP));
  try{
    while(true)
    {
      if(logging_chunk.logging_file_created)
      {
        handle_monitoring(sock, logging_chunk);
      }
      if(logging_chunk.scoring_file_created)
      {
        handle_scoring(sock, logging_chunk, logging_chunk.model_created or logging_chunk.server_created);
      }
      if(logging_chunk.model_created)
      {
        handle_nn_model(sock, logging_chunk, chunk_history);
      }
      if(logging_chunk.server_created)
      {
        handle_server_model(sock, logging_chunk, chunk_history);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(logging_chunk.MILLISECONDS_TO_SLEEP));
    }
  }catch (const std::exception& e)
  {
      std::cerr << e.what() << "hmmm" << std::endl;
  }
}