cd ..
wget http://data.fcc.gov/download/measuring-broadband-america/2017/data-raw-2016-jan.tar.gz --no-check-certificate
# wget http://data.fcc.gov/download/measuring-broadband-america/2017/data-raw-2016-aug.tar.gz --no-check-certificate
# wget http://data.fcc.gov/download/measuring-broadband-america/2017/data-raw-2016-apr.tar.gz --no-check-certificate
# wget http://users.ugent.be/~jvdrhoof/dataset-4g/logs/logs_all.zip
# wget -r --no-parent --reject "index.html*" https://datasets.simula.no/hsdpa-tcp-logs/
tar -zxvf data-raw-2016-jan.tar.gz 201601/curr_webget_2016_01.csv
# tar -zxvf data-raw-2016-aug.tar.gz 201608/curr_webget_2016_08.csv
# tar -zxvf data-raw-2016-apr.tar.gz 201604/curr_webget_2016_04.csv
# unzip logs_all.zip -d belgium
rm data-raw-2016-jan.tar.gz
# rm data-raw-2016-aug.tar.gz
# rm data-raw-2016-apr.tar.gz
cd traces
python load_webget_data.py --file ../201601/curr_webget_2016_01.csv
# python load_webget_data.py --file ../201608/curr_webget_2016_08.csv
# python load_webget_data.py --file ../201604/curr_webget_2016_04.csv
cd ..
rm -rf 201601/
# rm -rf 201608/ 
# rm -rf 201604/
cd traces
python3 convert_mahimahi_format.py
python3 split_train_test.py
# rm -rf cooked
# rm -rf mahimahi
