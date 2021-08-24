export PUFFER_PORTAL_SECRET_KEY='1'
. ~/.bashrc
export INFLUXDB_PASSWORD='1'
. ~/.bashrc
export PUFFER_PORTAL_DB_KEY='1'
. ~/.bashrc
LD_LIBRARY_PATH=/home/mike/Desktop/puffer_scripts/puffer/src/pcc/core
export LD_LIBRARY_PATH
pwd
# ./src/portal/manage.py migrate
# ln -s ../../../../../third_party/dist-for-puffer/ src/portal/puffer/static/puffer/dist
./src/portal/manage.py runserver 0:8080

