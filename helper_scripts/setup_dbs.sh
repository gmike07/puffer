
sudo apt install python3-pip -y
sudo apt install python3 -y
pip3 install django psycopg2-binary influxdb pyyaml matplotlib flask, tqdm
pip3 install django[argon2]
sudo apt install postgresql -y
# sudo -u postgres psql
# DO ONCE
# CREATE DATABASE puffer1;
# CREATE USER puffer1 WITH PASSWORD '<postgres password>';
# GRANT ALL PRIVILEGES ON DATABASE puffer1 TO puffer1;
# \q

echo '======================================'
echo enter the following commands:
echo 1. sudo -u postgres psql
echo 2. CREATE DATABASE puffer1;
echo 3. CREATE USER puffer1 WITH PASSWORD \'<postgres password>\';
echo 4. GRANT ALL PRIVILEGES ON DATABASE puffer1 TO puffer1;
echo 5. \\q
echo '======================================'

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
echo '======================================'
echo enter the following commands:
echo 1. sudo systemctl restart influxdb
echo 2. influx
echo 3. auth
echo 4. username: puffer1
echo 5. "password: <influxdb password>"
echo 6. "CREATE DATABASE puffer1"
echo 7. "SHOW DATABASES"
echo 8. "exit"
echo '======================================'
