export PUFFER_PORTAL_SECRET_KEY='1'
. ~/.bashrc
export INFLUXDB_PASSWORD='1'
. ~/.bashrc
export PUFFER_PORTAL_DB_KEY='1'
. ~/.bashrc
cd ..
./src/portal/manage.py migrate
ln -s ../../../../../third_party/dist-for-puffer/ src/portal/puffer/static/puffer/dist
./src/portal/manage.py runserver 0:8080
cd helper_scripts
