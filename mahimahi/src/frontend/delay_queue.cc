/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */
#include <iostream>
#include <limits>
#include <random>
#include "delay_queue.hh"
#include "timestamp.hh"

using namespace std;

void DelayQueue::read_packet( const string & contents )
{
    packet_queue_.emplace( timestamp() + delay_ms_, contents );
}

void DelayQueue::write_packets( FileDescriptor & fd )
{
    while ( (!packet_queue_.empty())
            && (packet_queue_.front().first <= timestamp()) ) {
        fd.write( packet_queue_.front().second );
        packet_queue_.pop();
    }
}

unsigned int DelayQueue::wait_time( void ) const
{
    if ( packet_queue_.empty() ) {
        return numeric_limits<uint16_t>::max();
    }

    const auto now = timestamp();

    if ( packet_queue_.front().first <= now ) {
        return 0;
    } else {
        return packet_queue_.front().first - now;
    }
}


void DelayQueue::handle_delay_thread()
{
    std::mt19937 gen;
    if(seed_given_)
    {
        gen = std::mt19937(seed_); // seed the generator
        
    }
    else
    {
        std::random_device rd; // obtain a random number from hardware
        gen = std::mt19937(rd());
    }
    std::uniform_int_distribution<> distr(0, delays_.size()-1); // define the range
    while(true)
    {
        delay_ms_ = delays_[distr(gen)]; // generate numbers
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_switch));
    }
}