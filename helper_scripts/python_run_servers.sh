export PUFFER_PORTAL_SECRET_KEY='1'
. ~/.bashrc
export INFLUXDB_PASSWORD='1'
. ~/.bashrc
export PUFFER_PORTAL_DB_KEY='1'
. ~/.bashrc
LD_LIBRARY_PATH=/home/mike/Desktop/puffer_scripts/puffer/src/pcc/core
export LD_LIBRARY_PATH
sudo sysctl -p
sysctl net.ipv4.tcp_allowed_congestion_control
./src/media-server/run_servers ./src/settings_offline.yml
