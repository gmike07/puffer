import os
import shutil
import numpy as np
import argparse


FILE_SIZE = 2000
BYTES_PER_PKT = 1500.0
MILLISEC_IN_SEC = 1000.0
EXP_LEN = 5000.0  # millisecond
MAX_TRACE_TIME = (10*60*1000)


def convert_file(trace_file, output_file):
    with open(trace_file, 'r') as f, open(output_file, 'w') as mf:
        millisec_time = 0
        mf.write(str(millisec_time) + '\n')
        for line in f:
            throughput = float(line.split()[0])
            pkt_per_millisec = throughput / BYTES_PER_PKT / MILLISEC_IN_SEC

            millisec_count = 0
            pkt_count = 0
            while True:
                millisec_count += 1
                millisec_time += 1
                to_send = (millisec_count * pkt_per_millisec) - pkt_count
                to_send = np.floor(to_send)

                for i in range(int(to_send)):
                    if millisec_time > MAX_TRACE_TIME:
                        return

                    mf.write(str(millisec_time) + '\n')

                pkt_count += to_send

                if millisec_count >= EXP_LEN:
                    break


def main():
    parser = argparse.ArgumentParser(description="Traces converter")
    parser.add_argument(
        "--cooked",
        default='./cooked/',
        help='cooked traces dir'
    )
    parser.add_argument(
        "--output",
        default='./mahimahi/',
        help='mahimahi trace dir'
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=500,
        help='num of traces'
    )

    args = parser.parse_args()

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.mkdir(args.output)

    files = sorted(os.listdir(args.cooked))

    i = 0
    for trace_file in files:
        if i >= args.count:
            break

        if os.stat(args.cooked + trace_file).st_size >= FILE_SIZE or True:
            input = args.cooked + trace_file
            output = args.output + trace_file
            convert_file(input, output)

            i += 1
            print(str(i) + '/' + str(min(args.count, len(files))))


if __name__ == '__main__':
    main()