
sudo apt install python3-pip
sudo apt install python3
pip3 install django psycopg2-binary influxdb pyyaml matplotlib flask
pip3 install django[argon2]
sudo apt install postgresql
# sudo -u postgres psql
# DO ONCE
# CREATE DATABASE puffer;
# CREATE USER puffer WITH PASSWORD '<postgres password>';
# GRANT ALL PRIVILEGES ON DATABASE puffer TO puffer;
# \q

wget -qO- https://repos.influxdata.com/influxdb.key | sudo apt-key add -
source /etc/lsb-release
echo "deb https://repos.influxdata.com/${DISTRIB_ID,,} ${DISTRIB_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/influxdb.list

# install and start the InfluxDB service
sudo apt-get update && sudo apt-get install influxdb
sudo systemctl unmask influxdb.service
sudo systemctl start influxdb

# influx
# CREATE USER puffer WITH PASSWORD '<influxdb password>' WITH ALL PRIVILEGES
#TODO: Next, modify /etc/influxdb/influxdb.conf to enable authentication by setting the auth-enabled option to true in the [http] section


#sudo systemctl restart influxdb
#influx
#auth
#username: puffer
#password: <influxdb password>
#CREATE DATABASE puffer
#SHOW DATABASES
