cd helper_scripts
cwd=$(pwd)
cd $cwd
./download_libraries.sh # setup libraries
cd $cwd
./download_movies.sh # download film to stream
cd $cwd
./compile_mahimahi.sh # compile a altered version of mahimahi with changing delays
cd $cwd
./compile_puffer.sh # compile the code
cd $cwd
./setup_dbs.sh #setup the databases to store the emulation
cd $cwd
./migrate.sh # define symbolic links
cd $cwd
./create_trace.sh # create traces to run on