# Puffer [![Build Status](https://travis-ci.org/StanfordSNR/puffer.svg?branch=master)](https://travis-ci.org/StanfordSNR/puffer)

# First thing to set up
- Follow the wiki from the github repo of puffer  
- Fix build issuses with putting line opus-encoder.cc:265 in comment  
- install libcurl4-openssl-dev

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
4. fix ports in `src/portal/puffer/static/puffer/js/puffer.js` the argument that determine `ws_host_port` (see Ports section)
5. edit in `./src/settings.yml` the variables:
    - `media_dir` - from where videos are served
    - `log_dir` - log dir
    - `experiments` - experiments to run
6. you can view videos http://localhost:8080/player/{?wsport=9361}

#  Ports
ports begin from ws_port + i where i=1,... as the number of experiments

# How to run experiments?
0. run `sudo influxd`.
1. run `./src/media-server/run_servers ./src/settings_offline.yml` (running the servers).
2. run `python3 ./src/portal/manage.py runserver 0:8080` (to run the python server that serves html files).
3. run `python3 ./src/media-server/offline_test.py` (config the number of chromiums to run, as the numebr of experiments).

# Data
Data is collected from the streaming and written to influxdb.  
- Useful commands:  
    - start db: `sudo systemctl start influxdb`  
    - `show databases`  
    - `use puffer`  
    - `show measurements`  
    - `select * from measurements` 
    - delete all measurements `drop series from /.*/` 

# Debug
- Remember to compile with the flag `-g`: `sudo make -j CXXFLAGS='-DNONSECURE -g' CFLAGS='-g'` 
- export env variable: `source ~/.bashrc`  
- Debugging the server: `./src/media-server/ws_media_server src/settings.yml 1 3`  

# Puffer Reinforce
Puffer reinforce uses the first 2 layer of the ttp and reiforce to determine the policy to use.  
In order to use ttp we must omit it last layer:  
- Download a trained model from puffer site, and place it for example in `ttp/models/bbr-20210112-1`  
- In each `ccp-{}.pt` (may depend on the horizon):  
    - edit the file `model.json` and delete the last layer  
    - edit the file `code/ccp-{}.py` and delete the last layer  

In order to train a new model: `python3.7 src/scripts/ttp.py src/settings_offline.yml --save-model ttp/models/fcc/`
(install torch==1.0.0, matplotlib==3.0.0)

# Postgress
psotgress quick guide:
* connect: 
	- `psql "host=127.0.0.1 port=5432 dbname=puffer user=puffer password=$PUFFER_PORTAL_DB_KEY"`
	- `psql "host=127.0.0.1 port=5432 dbname=puffer user=puffer password=123456"`
* exit: `\q`
* use db: `use puffer`
* show tables: `\dt`
* execute commands: `SELECT * FROM puffer_experiment;` (note for the semicolon)
* delete: `DELETE FROM puffer_experiment;`


influx paths: 
* data: `/var/lib/influxdb`
* config: `/etc/influxdb/influxdb.conf`

## Generate Plots (without the pipeline of puffer-statistics)
`python3 src/scripts/plot_ssim_rebuffer.py src/settings.yml -o ofir.png`

## FCC dataset
Go to `pensieve` repo and run:
* `cd traces`
* `python load_webget_data.py`
* `python convert_mahimahi_format.py`