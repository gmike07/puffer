#ifndef SENDER_HH
#define SENDER_HH

#include <vector>
#include <string>

class Sender 
{
public:
  Sender(){};
  void send_datapoint(std::vector<double> datapoint, 
                      size_t buf_size, 
                      size_t last_format);
};

#endif /* SENDER_HH */
