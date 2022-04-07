# Puffer [![Build Status](https://travis-ci.org/StanfordSNR/puffer.svg?branch=master)](https://travis-ci.org/StanfordSNR/puffer)

# First things to set up
- Follow the wiki from the github repo of puffer  
- Fix build issuses with putting line opus-encoder.cc:265 in comment  
- install libcurl4-openssl-dev, libcurlpp-dev
- install Pillow before installing tf-agents,tensorflow

## Prerequisites
1. validate cc: bbr and cubic are available with `sysctl net.ipv4.tcp_available_congestion_control`
2. If not, add bbr:
    - edit the file: `sudo vi /etc/sysctl.conf`
    - add the lines:
        ```
        net.core.default_qdisc=fq
        net.ipv4.tcp_congestion_control=vegas
	net.ipv4.tcp_congestion_control=bbr
        ```
    - restart: `sudo sysctl --system`'
4. fix ports in `src/portal/puffer/static/puffer/js/puffer.js` the argument that determine `ws_host_port`
5. edit in `./src/settings.yml` the variables:
    - `media_dir` - from where videos are served
    - `log_dir` - log dir
    - `experiments` - experiments to run
6. you can view videos http://localhost:8080/player/{?wsport=9361}

## FCC dataset
Generate traces with:
* `cd traces`
* `python load_webget_data.py`
* `python convert_mahimahi_format.py`

# How to run experiments?
0. run `sudo influxd`.
1. run `./src/media-server/run_servers ./src/settings_offline.yml` (running the servers).
2. run `python3 ./src/portal/manage.py runserver 0:8080` (to run the python server that serves html files).
3. run `python3 ./src/media-server/offline_test.py` (config the number of chromiums to run, as the numebr of experiments).
* number of servers needs to be equal to number of clients.

## Debug
- Remember to compile with the flag `-g`: `sudo make -j CXXFLAGS='-DNONSECURE -g' CFLAGS='-g'` 
- export env variable: `source ~/.bashrc`  
- Debugging the server: `./src/media-server/ws_media_server src/settings.yml 1 3`  

# InfluxDB
Data is collected from the streaming and written to influxdb.  
- influx paths: 
    - data: `/var/lib/influxdb`
    - config: `/etc/influxdb/influxdb.conf`
- Useful commands:  
    - start db: `sudo systemctl start influxdb`  
    - `show databases`  
    - `use puffer`  
    - `show measurements`  
    - `select * from measurements` 
    - set UTC: `precision rfc3339`
    - delete all measurements `drop series from /.*/` 
    - delete by id: `delete from video_acked where expt_id='36'`
    - `delete from active_streams, client_buffer, client_sysinfo, video_acked, video_sent where expt_id='180'`

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

## Generate Plots (without the pipeline of puffer-statistics)
`python3.7 src/scripts/plot_ssim_rebuffer.py src/settings_offline.yml -o weights/plot.png`

# Training models
* TTP (install torch==1.0.0, matplotlib==3.0.0): `python3.7 src/scripts/ttp.py src/settings_offline.yml --save-model weights/ttp/original/ --tune`
* TTP check inference: `python3.7 src/scripts/ttp.py src/settings_offline.yml --inference --load-model ./weights/ttp/original/ --from 2021-07-25T12:28:28.381Z --to 2021-07-25T15:19:25.682Z`
* Data Collect Server: `python3.7 src/scripts/data_collect_server.py -f`
* Clustering: `python3.7 src/scripts/clustering.py -f`
* Exp3Server: `python3.7 src/scripts/exp3_server.py`
