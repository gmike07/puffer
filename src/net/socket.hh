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

#include "address.hh"
#include "file_descriptor.hh"

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
    static const double million = 1000000;
    static const double pkt_bytes = 1500;
    bool is_pcc = false;
    std::vector<std::string> supported_ccs{"bbr", "vegas", "cubic"};
    std::string current_cc = "";

    std::string get_congestion_control_tcp() const;
    std::string socket_double_to_string(const double input, const int precision) const
    {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(precision) << input;
        return stream.str();
    }

protected:
    /* constructor used by accept() and SecureSocket() */
    TCPSocket( FileDescriptor && fd ) : Socket( std::move( fd ), AF_INET, SOCK_STREAM ) {
        is_pcc = false;
        //UDT::startup();
        //pcc_fd = UDT::socket(AF_INET, SOCK_STREAM, AI_PASSIVE);
        current_cc = get_congestion_control_tcp();
    }

public:
    //finished initializing all variables
    bool created_socket = false;
    //should randomize cc
    bool random_cc = true;

    //new chunk booleans
    bool is_new_chunk_logging = false;
    bool is_new_chunk_scoring = false;
    bool is_new_chunk_model = false;

    //paths
    std::string logging_path = "";
    std::string scoring_path = "";
    std::string model_path = "";

    //data for model
    int history_size = 40;
    int sample_size = 7;
    bool abr_time = false;
    int nn_roundup = 1000;

    //scoring data
    double scoring_mu = 1.0;
    double scoring_lambda = 1.0;
    std::string scoring_type = "ssim";
    ChunkInfo prev_chunk = {false, 0, 0, 0, 0, 0, ""};
    ChunkInfo curr_chunk = {false, 0, 0, 0, 0, 0, ""};

    TCPSocket() : Socket( AF_INET, SOCK_STREAM ) {
        is_pcc = false;
        // UDT::startup();
        //pcc_fd = UDT::socket(AF_INET, SOCK_STREAM, AI_PASSIVE);
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

    void add_chunk(ChunkInfo& info);

    double score_chunks(const ChunkInfo& prev_chunk, const ChunkInfo& curr_chunk) const;
    double score_chunks() const;

    const std::vector<std::string>& get_supported_cc() const
    {
        return supported_ccs;
    }

    inline std::string generate_chunk_statistics() const
    {
        if(not curr_chunk.is_video)
        {
            return "audio,";
        }
        return "video," + std::to_string(curr_chunk.ssim) + "," +
                socket_double_to_string(curr_chunk.video_buffer, 8) + "," + 
                socket_double_to_string(curr_chunk.cum_rebuffer, 8) + "," +
                std::to_string(curr_chunk.media_chunk_size) + "," +
                std::to_string(curr_chunk.trans_time) + "," + 
                curr_chunk.resolution;
    }

    inline std::vector<uint64_t> get_tcp_full_vector() const
    {
        tcp_info info = get_tcp_full_info();
        return 
            {info.tcpi_sndbuf_limited, info.tcpi_rwnd_limited, info.tcpi_busy_time, info.tcpi_delivery_rate,
            info.tcpi_data_segs_out, info.tcpi_data_segs_in, info.tcpi_min_rtt, info.tcpi_notsent_bytes,
            info.tcpi_segs_in, info.tcpi_segs_out, info.tcpi_bytes_received, info.tcpi_bytes_acked, 
            info.tcpi_max_pacing_rate, info.tcpi_total_retrans, info.tcpi_rcv_space, info.tcpi_rcv_rtt,
            info.tcpi_reordering, info.tcpi_advmss, info.tcpi_snd_cwnd, info.tcpi_snd_ssthresh, info.tcpi_rttvar,
            info.tcpi_rtt, info.tcpi_rcv_ssthresh, info.tcpi_pmtu, info.tcpi_last_ack_recv, info.tcpi_last_data_recv,
            info.tcpi_last_data_sent, info.tcpi_fackets, info.tcpi_retrans, info.tcpi_lost, info.tcpi_sacked, 
            info.tcpi_unacked, info.tcpi_rcv_mss, info.tcpi_snd_mss, info.tcpi_ato, info.tcpi_rto, 
            info.tcpi_backoff, info.tcpi_probes, info.tcpi_ca_state};
    }

    inline std::vector<double> get_tcp_full_normalized_vector(uint64_t delta_time) const
    {
        tcp_info info = get_tcp_full_info();
        return 
        {
            info.tcpi_sndbuf_limited / (10 * million), info.tcpi_rwnd_limited / (10 * million), 
            info.tcpi_busy_time / (1000 * million), info.tcpi_delivery_rate / (10 * million),
            info.tcpi_data_segs_out / (16 * pkt_bytes), info.tcpi_data_segs_in / (16 * pkt_bytes), 
            info.tcpi_min_rtt / (10 * million), info.tcpi_notsent_bytes / (1024 *  pkt_bytes),
            info.tcpi_segs_in / (16 * pkt_bytes), info.tcpi_segs_out / (16 * pkt_bytes), 
            info.tcpi_bytes_received / (1024 *  pkt_bytes), info.tcpi_bytes_acked / (16 * 1024 *  pkt_bytes), 
            info.tcpi_total_retrans / 100.0, info.tcpi_rcv_space / (16 * pkt_bytes), 
            info.tcpi_rcv_rtt / (10 * million), info.tcpi_snd_cwnd / (16 * pkt_bytes), 
            info.tcpi_snd_ssthresh / (1000 * 1024 * pkt_bytes), info.tcpi_rttvar / (10 * million),
            info.tcpi_rtt / (10 * million), info.tcpi_rcv_ssthresh / (1000 * 16 * pkt_bytes), 
            info.tcpi_last_ack_recv / (16 * pkt_bytes), info.tcpi_last_data_recv / (16 * pkt_bytes),
            info.tcpi_last_data_sent / (16 * pkt_bytes), info.tcpi_retrans / 1024.0, 
            info.tcpi_lost / 1024.0, info.tcpi_sacked / 1024.0, 
            info.tcpi_unacked / 1024.0, info.tcpi_rcv_mss / (16 * pkt_bytes), 
            info.tcpi_ato / (10 * million), info.tcpi_rto / (10 * million), 
            info.tcpi_backoff / 1.0, info.tcpi_ca_state / 4.0,
            delta_time / (10 * million)
        };
    }

};

#endif /* SOCKET_HH */
