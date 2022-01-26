wget http://users.ugent.be/~jvdrhoof/dataset-4g/logs/logs_all.zip
unzip logs_all.zip -d belgium
rm logs_all.zip
python3 convert_mahimahi_format_belgium.py
mkdir final_traces
mv mahimahi final_traces/test