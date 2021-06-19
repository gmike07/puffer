#include <sender.hh>

#include <curl/curl.h>
#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/Infos.hpp>

#include <exception>
#include "json.hpp"

using json = nlohmann::json;

void Sender::send_datapoint(std::vector<double> datapoint, 
                            size_t buf_size, 
                            size_t last_format)
{
  json data;
  data["datapoint"] = datapoint;
  data["buffer_size"] = buf_size;
  data["last_format"] = last_format;

  std::list<std::string> header;
  header.push_back("Content-Type: application/json");

  curlpp::Easy request;
  request.setOpt(new curlpp::options::Url("http://localhost:8888/"));
  request.setOpt(new curlpp::options::HttpHeader(header));
  request.setOpt(new curlpp::options::PostFields(data.dump()));
  request.setOpt(new curlpp::options::PostFieldSize(data.dump().size()));

  try {
    request.perform();
  }
  catch (std::exception& e) {
    std::cout << "exception " << e.what() << std::endl;
  }
}