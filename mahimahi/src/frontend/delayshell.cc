/* -*-mode:c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

#include <vector>
#include <string>

#include "delay_queue.hh"
#include "util.hh"
#include "ezio.hh"
#include "packetshell.cc"
#include <sstream>

using namespace std;

const char delimiter_delays = ',';
const string delimiter_ms = "-";
const string delimiter_seed = "#";
const uint64_t default_seed = 0;
const uint64_t default_ms = 5000;

vector<uint64_t> parse_input(string argument, uint64_t& ms, uint64_t& seed, bool& found_seed)
{
    //format: delay1,...,delayn#seed-ms
    ms = default_ms;
    seed = default_seed;
    size_t pos = 0;
    if((pos=argument.find(delimiter_ms)) != string::npos)
    {
        ms = myatoi(argument.substr(pos + delimiter_ms.length()));
        argument = argument.substr(0, pos);
    }
    if((pos=argument.find(delimiter_seed)) != string::npos)
    {
        seed = myatoi(argument.substr(pos + delimiter_seed.length()));
        found_seed = true;
        argument = argument.substr(0, pos);
    }
    vector<uint64_t> delays;
    stringstream argument_stream(argument);
    string delay_string;
    while(getline(argument_stream, delay_string, delimiter_delays))
    {
        delays.push_back(myatoi(delay_string));
    }
    return delays;
}


int main( int argc, char *argv[] )
{
    try {
        /* clear environment while running as root */
        char ** const user_environment = environ;
        environ = nullptr;

        check_requirements( argc, argv );

        if ( argc < 2 ) {
            throw runtime_error( "Usage: " + string( argv[ 0 ] ) + " delay-milliseconds [command...]" );
        }
        vector< string > command;

        if ( argc == 2 ) {
            command.push_back( shell_path() );
        } else {
            for ( int i = 2; i < argc; i++ ) {
                command.push_back( argv[ i ] );
            }
        }

        bool seed_given = false;
        uint64_t seed, ms;
        vector<uint64_t> delays = parse_input(argv[1], ms, seed, seed_given);
        PacketShell<DelayQueue> delay_shell_app( "delay", user_environment );

        delay_shell_app.start_uplink( "[delay " + to_string( delays[0] ) + " ms] ",
                                      command,
                                      delays, ms, seed, seed_given);
        delay_shell_app.start_downlink( delays, ms, seed, seed_given);
        return delay_shell_app.wait_for_exit();
    } catch ( const exception & e ) {
        print_exception( e );
        return EXIT_FAILURE;
    }
}
