/* -*-mode:c++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
#include "cc_logging_objects.hh"

using json = nlohmann::json;

void ScoreHandler::operator()()
{
  if(not socket.is_new_chunk_scoring)
    {
      return;
    }
    if(not socket.is_valid_score_type() or (not socket.prev_chunk.is_video) or (not socket.curr_chunk.is_video))
    {
      socket.is_new_chunk_scoring = false;
      return;
    }
    double score = socket.score_chunks();
    std::string s = socket.get_congestion_control() + " " + std::to_string(score);
    if(not pref.empty())
    {
      s = pref + " " + s;
    }
    std::cout << s << std::endl;
    scoring_file << score << std::endl;
    socket.is_new_chunk_scoring = false;
}


int ServerSender::send(std::vector<double> state)
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
      return -1;
    }
    if(status != 200)
    {
      return status - base_good_code;
    }
  }
  catch (std::exception& e) {
    std::cout << "exception " << e.what() << std::endl;
  }
  return -1;
}

void ServerSender::send_state_and_replace_cc(std::vector<double> state)
{
  int cc_index = send(state);
  if(cc_index != -1)
  {
    change_cc(socket, cc_index);
  }
}

void StateServerHandler::operator()()
{
  history.update_chunk(convert_tcp_info_normalized_vec(socket, start_time));
  counter = (counter + 1) % nn_roundup;
  bool change_cc_1 = (abr_time and socket.is_new_chunk_model);
  socket.is_new_chunk_model = false;
  bool change_cc_2 = ((not abr_time) and (counter % nn_roundup == 0));
  if((not change_cc_1) and (not change_cc_2))
  {
    return;
  }
  //should change cc
  history.push_chunk();
  history.push_statistic(socket);
  start_time = get_timestamp_ms();
  if(history.size() != ((size_t) socket.history_size))
  {
    return;
  }
  std::vector<double> state(0);
  history.get_sample_history(state);
  std::thread([this, state](){sender.send_state_and_replace_cc(state);}).detach();
}


// void StatelessServerHandler::operator()()
// {
//   counter = (counter + 1) % nn_roundup;
//   bool change_cc_1 = (abr_time and socket.is_new_chunk_model);
//   socket.is_new_chunk_model = false;
//   bool change_cc_2 = ((not abr_time) and (counter % nn_roundup == 0));
//   if((not change_cc_1) and (not change_cc_2))
//   {
//     return;
//   }
//   //should change cc
//   std::vector<double> state(0);
//   std::thread([this, state](){sender.send_state_and_replace_cc(state);}).detach();
// }