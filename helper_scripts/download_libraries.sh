cd ..
git branch -a
git checkout amazon_congestion_control
git submodule update --recursive --init

sudo apt-get update -qq
cwd=$(pwd)
sudo apt-get install -y xmlto libboost-all-dev aptitude
aptitude search boost
sudo apt-get install -y -q gcc-7 g++-7 libmpeg2-4-dev libpq-dev \
                          libssl-dev libcrypto++-dev libyaml-cpp-dev \
                          libboost-dev liba52-dev opus-tools libopus-dev \
                          libsndfile-dev libavformat-dev libavutil-dev ffmpeg \
                          git automake libtool python python3 cmake wget chromium-browser unzip
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 99
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 99
# sudo apt install -y mahimahi
git clone https://github.com/jtv/libpqxx.git
cd libpqxx && git checkout 7.1.0 && sudo ./configure --enable-documentation=no && sudo make -j3 install
sudo useradd --create-home --shell /bin/bash user
sudo cp -R . /home/user/puffer
sudo chown ubuntu -R /home/user/puffer # machine home name
cd ..

cd third_party/
wget https://github.com/StanfordSNR/pytorch/releases/download/v1.0.0-puffer/libtorch.tar.gz
tar -xvf libtorch.tar.gz
# export PATH="$cwd:$PATH"
# source ~/.bashrc
cd ..

cd helper_scripts
