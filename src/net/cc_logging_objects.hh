/* -*-mode:c++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
#ifndef CC_LOGGING_OBJECTS_HH
#define CC_LOGGING_OBJECTS_HH
#include "cc_logging_functions.hh"
#include "socket.hh"
#include "cc_socket_helper.hh"

typedef std::vector<double> qoe_sample;
typedef std::vector<std::vector<double>> chunk_sample;


class DefaultHandler
{
public:
  DefaultHandler(SocketHelper& socket_helper_): socket_helper(socket_helper_) {}
  virtual void operator()()=0;
  virtual ~DefaultHandler()=default;
protected:
  SocketHelper& socket_helper;
};

class ScoreHandler: public virtual DefaultHandler
{
public:
  ScoreHandler(SocketHelper& socket_helper_, std::string prefix=""): DefaultHandler(socket_helper_), 
            scoring_file(std::ofstream(socket_helper_.scoring_path, std::ios::out | std::ios::app)), 
            pref(prefix){};
  virtual void operator()();
  virtual ~ScoreHandler()=default;
private:
  std::ofstream scoring_file;
  std::string pref;
};


class ServerSender
{
public:
  ServerSender(SocketHelper& socket_helper_): socket_helper(socket_helper_) {};
  int send(std::vector<double> state, bool stateless=false);
  static std::pair<int, std::string> send_and_receive(const std::string& host, const json& js);
  static std::pair<int, std::string> send_and_receive_str(const std::string& host, const std::string& str, int server_id);
  void send_state_and_replace_cc(std::vector<double> state, bool stateless=false);
private:
  SocketHelper& socket_helper;
};


class AbstractHistory
{
protected:
  SocketHelper& socket_helper;
public:
  AbstractHistory(SocketHelper& socket_helper_): socket_helper(socket_helper_){}
  virtual ~AbstractHistory()=default;

  virtual void update_chunk(std::uint64_t start_time) = 0;

  virtual void push_statistic() = 0;

  /*
  * adds to vec a random sample of each chunk of the history
  */
  virtual void get_state(std::vector<double>& vec) = 0;

  
  virtual size_t size() = 0;

  /*
  * adds the current chunk to the history and updates the size of the history if needed
  */
  virtual void push_chunk() = 0;

  virtual std::string get_info() const{return "";}

};



template <typename ChunkObject, typename QoEObject>
class AbstractHistoryHelper: public virtual AbstractHistory
{
protected:
  const int history_size;
  ChunkObject curr_chunk;
  std::deque<ChunkObject> history_chunks;
  std::deque<QoEObject> qoe_statistics;

  void push_statistic_helper(const QoEObject& curr_statistics)
  {
    qoe_statistics.push_back(curr_statistics);
    if(qoe_statistics.size() > (size_t) history_size)
    {
      qoe_statistics.pop_front();
    }
  }
public:
  AbstractHistoryHelper(SocketHelper& socket_helper_):  AbstractHistory(socket_helper_), history_size(socket_helper_.history_size), curr_chunk(0),  history_chunks(0), qoe_statistics(0){}
  virtual ~AbstractHistoryHelper()=default;

  size_t size() {return history_chunks.size();}

  /*
  * adds the current chunk to the history and updates the size of the history if needed
  */
  void push_chunk()
  {
    history_chunks.push_back(curr_chunk);
    curr_chunk = ChunkObject(0);
    if(history_chunks.size() > (size_t) history_size)
    {
      history_chunks.pop_front();
    }
  }

  std::string get_info() const
  {
    return "history size:" + std::to_string(history_size) + 
          // ",\tsample size" + std::to_string((*history_chunks.begin())[0].size()) +
          ",\tqoe length" + std::to_string((*qoe_statistics.begin()).size());
  }
};


class ChunkHistory: public virtual AbstractHistoryHelper<chunk_sample, qoe_sample>
{
public:
  ChunkHistory(SocketHelper& socket_helper_): AbstractHistory(socket_helper_), AbstractHistoryHelper(socket_helper_), sample_size(socket_helper_.sample_size){}
  virtual ~ChunkHistory()=default;

  /*
  * adds a cc sample to the current chunk
  */
  void update_chunk(std::uint64_t start_time) {curr_chunk.push_back(convert_tcp_info_normalized_vec(socket_helper, start_time));}

  void push_statistic(){push_statistic_helper(socket_helper.get_qoe_vector());}

  /*
  * adds to vec a random sample of each chunk of the history
  */
  void get_state(std::vector<double>& vec)
  {
    for(size_t i = 0; i < history_chunks.size(); i++)
    {
      auto& chunk = history_chunks[i];
      chunk_sample sample_chunk(0);
      sample_random_elements(chunk, sample_chunk, sample_size);
      for(auto& curr_state : sample_chunk)
      {
        vec.insert(std::end(vec), std::begin(curr_state), std::end(curr_state));
      }
      auto vec_statistics = qoe_statistics[i];
      vec.insert(std::end(vec), std::begin(vec_statistics), std::end(vec_statistics));
    }
  }
private:
  const int sample_size;
};


class StateServerHandler: public virtual DefaultHandler
{
public:
  StateServerHandler(SocketHelper& socket_helper_, const std::shared_ptr<AbstractHistory>& history_p_r): 
                                DefaultHandler(socket_helper_), sender(socket_helper_), history_p(history_p_r),
                                start_time(get_timestamp_ms()), counter(0), 
                                nn_roundup(socket_helper_.nn_roundup), abr_time(socket_helper_.abr_time){};
  virtual void operator()();
  virtual ~StateServerHandler()=default;
private:
  ServerSender sender;
  std::shared_ptr<AbstractHistory> history_p;
  uint64_t start_time;
  uint64_t counter;
  uint64_t nn_roundup;
  bool abr_time;
};

#endif /* CC_LOGGING_OBJECTS_HH */
