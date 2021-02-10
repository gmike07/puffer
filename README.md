# Puffer [![Build Status](https://travis-ci.org/StanfordSNR/puffer.svg?branch=master)](https://travis-ci.org/StanfordSNR/puffer)

# Prerequisites
1. validate cc: bbr and cubic are available with `sysctl net.ipv4.tcp_available_congestion_control`
2. If not, add bbr:
    - edit the file: `sudo vi /etc/sysctl.conf`
    - add the lines:
        ```
        net.core.default_qdisc=fq
        net.ipv4.tcp_congestion_control=bbr
        ```
    - restart: `sudo sysctl --system`'
4. fix ports in `src/portal/puffer/static/puffer/js/puffer.js` the argument that determine `ws_host_port`
5. edit in `./src/settings.yml` the variables:
    - `media_dir` - from where videos are served
    - `log_dir` - log dir
    - `experiments` - experiments to run
6. you can view videos http://localhost:8080/player/{?wsport=9361}

# How it works?
ports begin from ws_port + i where i=1,... as the number of experiments

# How to run experiments?
1. run `./src/media-server/run_servers ./src/settings.yml` (running the servers).
2. run `python3 ./src/portal/manage.py runserver 0:8080` (to run the python server that serves html files).
3. run `python3 ./src/media-server/offline_test.py` (config the number of chromiums to run, as the numebr of experiments).
