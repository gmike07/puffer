/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#ifndef DELAY_QUEUE_HH
#define DELAY_QUEUE_HH

#include <queue>
#include <cstdint>
#include <string>
#include <thread>

#include "file_descriptor.hh"

#define DEFAULT_MS 5000
#define DEFAULT_SEED 10000000

static std::thread delay_thread;

class DelayQueue
{
private:
    uint64_t delay_ms_;
    std::vector<uint64_t> delays_;
    uint64_t delay_ms_switch;
    uint64_t seed_;
    bool seed_given_;
    std::queue< std::pair<uint64_t, std::string> > packet_queue_;

    /* release timestamp, contents */

public:
    DelayQueue(const std::vector<uint64_t> delays, uint64_t switch_ms=DEFAULT_MS, uint64_t seed=DEFAULT_SEED, bool seed_given=false) : 
            delay_ms_(delays[0]), delays_(delays), delay_ms_switch(switch_ms), seed_(seed), seed_given_(seed_given), packet_queue_() {
                delay_thread = std::thread( [this] { handle_delay_thread(); } );
                delay_thread.detach();
            }

    void read_packet( const std::string & contents );

    void write_packets( FileDescriptor & fd );

    unsigned int wait_time( void ) const;

    bool pending_output( void ) const { return wait_time() <= 0; }

    static bool finished( void ) { return false; }

    void handle_delay_thread();
};

#endif /* DELAY_QUEUE_HH */
