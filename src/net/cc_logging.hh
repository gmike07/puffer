/* -*-mode:c++; tab-width: 2; indent-tabs-mode: nil; c-basic-offset: 2 -*- */

#ifndef CC_LOGGING_HH
#define CC_LOGGING_HH
#include "socket.hh"

/*
 * handles the thread to store statistics of the socket and change the cc
*/
void logging_cc_func(TCPSocket* socket);


#endif /* CC_LOGGING_HH */