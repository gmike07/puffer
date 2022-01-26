from calendar import c
import os
import numpy as np
import shutil

DATA_PATH = './datasets.simula.no/hsdpa-tcp-logs/'
MID_PATH = './helper_mahimahi/'
OUTPUT_PATH = './mahimahi/'
BYTES_PER_PKT = 1500.0
MILLISEC_IN_SEC = 1000.0
BITS_IN_BYTE = 8.0


def recreated_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)


def create_mid_files():
    files = os.listdir(DATA_PATH)
    for dir in files:
        if dir.endswith('.gz'):
            continue
        for f in os.listdir(DATA_PATH + dir):
            file_path = DATA_PATH + dir + "/" + f
            output_path = MID_PATH + f
            if not f.endswith('.log'):
                continue
            with open(file_path, 'rb') as f1, open(output_path, 'wb') as mf:
                mf.write(f1.read())



def main():
    recreated_dir(MID_PATH)
    recreated_dir(OUTPUT_PATH)
    create_mid_files()

    files = os.listdir(MID_PATH)

    for f in files:
        file_path = MID_PATH +  f
        output_path = OUTPUT_PATH + f

        print(file_path)

        with open(file_path, 'rb') as f, open(output_path, 'w') as mf:
            time_ms = []
            bytes_recv = []
            recv_time = []
            for line in f:
                parse = line.split()
                if len(time_ms) > 0 and float(parse[0]) < time_ms[-1]:  # trace error, time not monotonically increasing
                    break
                if parse[2] == 'NOFIX':
                    continue
                time_ms.append(float(parse[0]))
                bytes_recv.append(float(parse[1]))
                recv_time.append(float(parse[2]))

            time_ms = np.array(time_ms)
            bytes_recv = np.array(bytes_recv)
            recv_time = np.array(recv_time)
            throughput_all = bytes_recv / recv_time

            millisec_time = 0
            mf.write(str(millisec_time) + '\n')

            for i in xrange(len(throughput_all)):

                throughput = throughput_all[i]
                
                pkt_per_millisec = throughput / BYTES_PER_PKT 

                millisec_count = 0
                pkt_count = 0

                while True:
                    millisec_count += 1
                    millisec_time += 1
                    to_send = (millisec_count * pkt_per_millisec) - pkt_count
                    to_send = np.floor(to_send)

                    for _ in xrange(int(to_send)):
                        mf.write(str(millisec_time) + '\n')

                    pkt_count += to_send

                    if millisec_count >= recv_time[i]:
                        break
	

if __name__ == '__main__':
	main()