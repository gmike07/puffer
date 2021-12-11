/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef SOCKET_HH
#define SOCKET_HH

#include <functional>
#include <linux/tcp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <sstream>
#include <algorithm>
#include <memory>
#include "address.hh"
#include "file_descriptor.hh"
#include <deque>
#include "abr_algo.hh"

class SocketHelper;

/* class for network sockets (UDP, TCP, etc.) */
class Socket : public FileDescriptor
{
private:
    /* get the local or peer address the socket is connected to */
    Address get_address( const std::string & name_of_function,
                         const std::function<int(int, sockaddr *, socklen_t *)> & function ) const;

protected:
    /* default constructor */
    Socket( const int domain, const int type );

    /* construct from file descriptor */
    Socket( FileDescriptor && s_fd, const int domain, const int type );

    /* get and set socket option */
    template <typename option_type>
    socklen_t getsockopt( const int level, const int option, option_type & option_value ) const;

    template <typename option_type>
    void setsockopt( const int level, const int option, const option_type & option_value );

public:
    /* bind socket to a specified local address (usually to listen/accept) */
    void bind( const Address & address );

    /* connect socket to a specified peer address */
    void connect( const Address & address );

    /* accessors */
    Address local_address( void ) const;
    Address peer_address( void ) const;

    /* allow local address to be reused sooner, at the cost of some robustness */
    void set_reuseaddr( void );
    void set_reuseport( void );
};

/* UDP socket */
class UDPSocket : public Socket
{
public:
    UDPSocket() : Socket( AF_INET, SOCK_DGRAM ) {}

    /* receive datagram and where it came from */
    std::pair<Address, std::string> recvfrom( void );

    /* send datagram to specified address */
    void sendto( const Address & peer, const std::string & payload );

    /* send datagram to connected address */
    void send( const std::string & payload );

    /* turn on timestamps on receipt */
    void set_timestamps( void );
};

/* tcp_info of our interest; keep the units used in the kernel */
struct TCPInfo
{
  uint32_t cwnd;      /* congestion window (packets) */
  uint32_t in_flight; /* packets "in flight" */
  uint32_t min_rtt;   /* minimum RTT in microsecond */
  uint32_t rtt;       /* RTT in microsecond */
  uint64_t delivery_rate;  /* bytes per second */
};

struct ChunkInfo
{
    bool is_video;
    double ssim;
    double video_buffer;
    double cum_rebuffer;
    unsigned int media_chunk_size;
    uint64_t trans_time;
    std::string resolution;
};


/* TCP socket */
class TCPSocket : public Socket
{
private:
    static constexpr double UNIT_BUF_LENGTH = 0.5;
    static constexpr double MAX_SSIM = 60;
    static constexpr double MIN_SSIM = 0;
    static constexpr double million = 1000000;
    static constexpr double pkt_bytes = 1500;
    std::vector<std::string> scoring_types = {"ssim"}; //"bit_rate"
    std::string current_cc = "";
    std::vector<std::string> supported_ccs{};

    std::string get_congestion_control_tcp() const;
protected:
    /* constructor used by accept() and SecureSocket() */
    TCPSocket( FileDescriptor && fd ) : Socket( std::move( fd ), AF_INET, SOCK_STREAM ){
        current_cc = get_congestion_control_tcp();
    }

public:
    //finished initializing all variables
    bool created_socket = false;
    std::shared_ptr<SocketHelper> socket_helper_p = nullptr;

    TCPSocket() : Socket( AF_INET, SOCK_STREAM ) {
        current_cc = get_congestion_control_tcp();
    }

    /* mark the socket as listening for incoming connections */
    void listen( const int backlog = 16 );

    /* accept a new incoming connection */
    TCPSocket accept( void );

    /* original destination of a DNAT connection */
    Address original_dest( void ) const;

    /* are there pending errors on a nonblocking socket? */
    void verify_no_errors() const;

    /* set the current congestion control algorithm */
    void set_congestion_control( const std::string & cc );

    /* get the current congestion control algorithm */
    std::string get_congestion_control() const;

    TCPInfo get_tcp_info() const;

    tcp_info get_tcp_full_info() const;
};

#endif /* SOCKET_HH */
