cd ..
./autogen.sh
sudo ./configure
sudo make -j CXXFLAGS='-DNONSECURE -g' CFLAGS='-g'
cd helper_scripts
