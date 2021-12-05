/* -*-mode:c++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
#ifndef CC_LOGGING_OBJECTS_HH
#define CC_LOGGING_OBJECTS_HH
#include "cc_logging_functions.hh"


typedef std::vector<double> qoe_sample;
typedef std::vector<std::vector<double>> chunk_sample;


class DefaultHandler
{
public:
  DefaultHandler(TCPSocket& sock): socket(sock) {}
  virtual void operator()()=0;
  ~DefaultHandler();
protected:
  TCPSocket& socket;
};

class ScoreHandler: public virtual DefaultHandler
{
public:
  ScoreHandler(TCPSocket& sock, std::string prefix=""): DefaultHandler(sock), 
            scoring_file(std::ofstream(sock.scoring_path, std::ios::out | std::ios::app)), pref(prefix){};
  void operator()();
  ~ScoreHandler(){};
private:
  std::ofstream scoring_file;
  std::string pref;
};

class MonitoringHandler: public virtual DefaultHandler
{
public:
  MonitoringHandler(TCPSocket& sock): 
    DefaultHandler(sock), logging_file(std::ofstream(sock.logging_path, std::ios::out | std::ios::app)), start_time(get_timestamp_ms())
    {
      logging_file << "new run," << std::endl;
    };
  void operator()();
  ~MonitoringHandler(){};
private:
  std::ofstream logging_file;
  uint64_t start_time;
};


class ServerSender
{
public:
  ServerSender(TCPSocket& sock, int start_good_code=406): socket(sock), base_good_code(start_good_code) {};
  ~ServerSender(){};
  int send(std::vector<double> state);
  void send_state_and_replace_cc(std::vector<double> state);
private:
  TCPSocket& socket;
  int base_good_code;
};


class ChunkHistory
{
public:
  ChunkHistory(TCPSocket& socket): 
    history_size(socket.history_size), sample_size(socket.sample_size),  curr_chunk(0),  history_chunks(0){}
  
  ~ChunkHistory(){}
  /*
  * returns the history current size
  */
  inline size_t size()
  {
    return history_chunks.size();
  }

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
    curr_chunk = chunk_sample(0);
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

  /*
  * adds to vec a random sample of each chunk of the history
  */
  void get_sample_history(std::vector<double>& vec)
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
  chunk_sample curr_chunk;
  std::deque<chunk_sample> history_chunks;
  std::deque<qoe_sample> qoe_statistics;
};


class StateServerHandler: public virtual DefaultHandler
{
public:
  StateServerHandler(TCPSocket& sock): 
                                DefaultHandler(sock), sender(sock), history(sock),
                                start_time(get_timestamp_ms()), counter(0), 
                                nn_roundup(sock.nn_roundup), abr_time(sock.abr_time){};
  void operator()();
  ~StateServerHandler(){};
private:
  ServerSender sender;
  ChunkHistory history;
  uint64_t start_time;
  uint64_t counter;
  uint64_t nn_roundup;
  bool abr_time;
};

class StatelessServerHandler: public virtual DefaultHandler
{
public:
  StatelessServerHandler(TCPSocket& sock): DefaultHandler(sock), sender(sock), counter(0), 
                                nn_roundup(sock.nn_roundup), abr_time(sock.abr_time){};
  void operator()();
  ~StatelessServerHandler(){};
private:
  ServerSender sender;
  uint64_t counter;
  uint64_t nn_roundup;
  bool abr_time;
};

#endif /* CC_LOGGING_OBJECTS_HH */
