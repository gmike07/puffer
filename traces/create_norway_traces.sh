wget -r --no-parent --reject "index.html*" https://datasets.simula.no/hsdpa-tcp-logs/
python convert_mahimahi_format_norway.py
python3 cut_mahimahi_norway.py
rm -rf mahimahi
mkdir final_traces
mv mahimahi_chunks final_traces/test
