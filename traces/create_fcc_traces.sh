year=2016
next_year=$((year + 1))

declare -A months
months["jan"]="01"
months["feb"]="02"
months["mar"]="03"
months["apr"]="04"
months["may"]="05"
months["jun"]="06"
months["jul"]="07"
months["aug"]="08"
months["sep"]="09"
months["oct"]="10"
months["nov"]="11"
months["dec"]="12"
if [ -z "$1" ]; then 
    month=jan
else 
    month=$1
fi
month_num="${months[$month]}"

rm -rf cooked mahimahi final_traces
echo cd ..
if [ -z "$2" ]; then
    echo "skipped downloading"
    unzip cooked-$month.zip
    mv cooked-$month cooked
else
    wget http://data.fcc.gov/download/measuring-broadband-america/$next_year/data-raw-$year-$month.tar.gz --no-check-certificate
    tar -zxvf data-raw-$year-$month.tar.gz $year$month_num/curr_webget_"$year"_$month_num.csv
    rm data-raw-$year-$month.tar.gz
    cd traces
    python load_webget_data.py --file ../$year$month_num/curr_webget_"$year"_$month_num.csv
    cd ..
    rm -rf $year$month_num/
fi
cd traces
python3 convert_mahimahi_format.py
python3 split_train_test.py