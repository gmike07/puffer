#include <sender.hh>

#include <curl/curl.h>
#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/Infos.hpp>

#include <exception>

long Sender::post(json data,
                  std::string endpoint)
{
  std::list<std::string> header;
  header.push_back("Content-Type: application/json");

  // std::ostringstream response;

  curlpp::Easy request;

  request.setOpt(new curlpp::options::Url("http://localhost:8888/" + endpoint));
  request.setOpt(new curlpp::options::HttpHeader(header));
  request.setOpt(new curlpp::options::PostFields(data.dump()));
  request.setOpt(new curlpp::options::PostFieldSize(data.dump().size()));
  // request.setOpt(new cURLpp::Options::WriteStream(&response));

  try {
    request.perform();

    long status = curlpp::infos::ResponseCode::get(request);

    return status;
    // if (status == 200) {
      // std::cout << "response: " << response.str() << std::endl;
    // }
  }
  catch (std::exception& e) {
    std::cout << "exception " << e.what() << std::endl;
  }
}