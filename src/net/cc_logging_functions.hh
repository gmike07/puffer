/* -*-mode:c++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
#ifndef CC_LOGGING_FUNCTIONS_HH
#define CC_LOGGING_FUNCTIONS_HH
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
#include "cc_socket_helper.hh"

using json = nlohmann::json;

/* nanoseconds per millisecond */
static const uint64_t MILLION = 1000000;

/* nanoseconds per second */
static const uint64_t BILLION = 1000 * MILLION;

/*
 * returns the current time in uint64_t type
*/
static uint64_t get_timestamp_ms()
{
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);

  const uint64_t nanos = ts.tv_sec * BILLION + ts.tv_nsec;
  return nanos / MILLION;
}

/*
 * changes the curremt cc to the given string in the socket
*/
static void change_cc(SocketHelper& socket_helper, std::string cc)
{
  socket_helper.set_congestion_control(cc);
  std::cerr << "cc: " << cc << std::endl;
}

/*
 * changes the curremt cc to the given string of the corresponding index in the supported ccs
*/
static void change_cc(SocketHelper& socket_helper, int index)
{ 
  static std::vector<std::string>& cc_supported = socket_helper.get_supported_cc();
  change_cc(socket_helper, cc_supported[index]);
}

/*
 * changes the curremt cc to a random cc in supported ccs
*/
static void random_cc(SocketHelper& socket_helper, bool replace_cc)
{
  if(replace_cc)
  {
    change_cc(socket_helper, rand() % socket_helper.get_supported_cc().size());
  }
}


/*
 * returns the predicted cc sample normalized
*/
static std::vector<double> convert_tcp_info_normalized_vec(SocketHelper& socket_helper, uint64_t start_time)
{
  std::vector<double> vec = socket_helper.get_tcp_full_normalized_vector(get_timestamp_ms() - start_time);
  auto& cc_vec = socket_helper.get_supported_cc();
  auto cc = socket_helper.get_congestion_control();
  for(size_t i = 0; i < cc_vec.size(); i++)
  {
    vec.push_back(cc_vec[(int)i] == cc ? 1.0 : 0.0);
  }
  return vec;
}

/*
 * adds random size random indexes to vector.
 * each index is up to size max
*/
static void get_random_indexes(std::vector<int>& indexes, size_t size, int max)
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
static void sample_random_elements(std::vector<T>& elements, std::vector<T>& sampled_elements, size_t sample_size)
{
  std::vector<int> indexes(0);
  sampled_elements = std::vector<T>(0);
  get_random_indexes(indexes, sample_size, elements.size());
  for(size_t i = 0; i < indexes.size(); i++)
  {
    sampled_elements.push_back(elements[indexes[i]]);
  }
}

#endif /* CC_LOGGING_FUNCTIONS_HH */
