#ifndef SENDER_HH
#define SENDER_HH

#include <vector>
#include <string>
#include "json.hpp"

using json = nlohmann::json;

class Sender 
{
public:
  Sender(){};
  std::string post(json data,
                   std::string endpoint);
};

#endif /* SENDER_HH */
