#include "cc_logging_objects.hh"
#include "cc_logging.hh"
const int MILLISECONDS_TO_SLEEP = 1;


void create_handlers(TCPSocket& socket, std::vector<std::unique_ptr<DefaultHandler>>& handlers)
{
  if(socket.scoring_path != "")
  {
    handlers.push_back(std::unique_ptr<DefaultHandler>(new ScoreHandler(socket, (socket.server_path != "") ? "nn": "")));
  }
  if(socket.server_path != "")
  {
    handlers.push_back(std::unique_ptr<DefaultHandler>(new StateServerHandler(socket)));
  }
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
  std::vector<std::unique_ptr<DefaultHandler>> handlers(0);
  create_handlers(sock, handlers);
  std::this_thread::sleep_for(std::chrono::milliseconds(MILLISECONDS_TO_SLEEP));
  try{
    while(true)
    {
      for(auto& handler: handlers)
      {
        handler->operator()();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(MILLISECONDS_TO_SLEEP));
    }
  }catch (const std::exception& e)
  {
      std::cerr << e.what() << "hmmm" << std::endl;
  }
}