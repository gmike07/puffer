from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import argparse
import sys
import scipy
import scipy.stats
import random

import matplotlib
matplotlib.use('Agg')

formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))


OUTPUT_PATH = './cooked/'
NUM_LINES = 1000000  # np.inf
TIME_ORIGIN = datetime.datetime.utcfromtimestamp(0)

bw_measurements = {}
BYTES_IN_MB = 1024**2 / 8.0


def get_avg__min_throughput(throughputs):
    """
    Every line in a cooked trace is throughput for 5 seconds.
    Generating mahimahi trace file dine with reading each cooked line,
        avg the number of 1500 bytes packet.
    """
    t = np.array(throughputs)
    t /= BYTES_IN_MB
    return t.mean(), t.min()


def plot_tput_distr(tputs):
    fig, ax = plt.subplots()
    ax.set_xlabel('Throughput (Mbps)')
    ax.set_ylabel('CDF')

    # tot = len(tputs)
    # tputs = [tput for tput in tputs]
    # tputs_rate = [i / tot for i in range(tot)]


    tputs = np.array(tputs)
    # norm_cdf = scipy.stats.norm.cdf(tputs)

    count, bins = np.histogram(tputs, bins=100)
    pdf = count / float(sum(count))

    cdf = np.cumsum(pdf)

    ax.semilogx(bins[1:], cdf, label='FCC traces')
    ax.set_ylim(0,1)
    ax.set_xlim(0.1)
    ax.legend()

    ax.xaxis.set_major_formatter(formatter)
    output = './plot_throughput.png'
    fig.savefig(output, bbox_inches='tight')
    sys.stderr.write('Saved plot to {}\n'.format(output))


def main(traces_file, min_average_throughput, max_average_throughput):
    traces = []

    line_counter = 0
    with open(traces_file, 'r') as f:
        for line in f:
            parse = line.split(',')
            uid = parse[0]
            dtime = (datetime.datetime.strptime(parse[1], '%Y-%m-%d %H:%M:%S')
                     - TIME_ORIGIN).total_seconds()
            target = parse[2]
            address = parse[3]
            throughput = float(parse[6])  # bytes per second

            k = (uid, target)

            if k in bw_measurements:
                bw_measurements[k].append(throughput)
            else:
                bw_measurements[k] = [throughput]

            line_counter += 1
            if line_counter >= NUM_LINES:
                break
    
    c = 0
    bw_keys = list(bw_measurements.keys())
    random.shuffle(bw_keys)
    for k in bw_keys:
        if c > 500:
            break

        throughputs = bw_measurements[k]
        t = np.array(throughputs)
        t /= BYTES_IN_MB
        avg_throughput, min_throughput = t.mean(), t.min()

        if not (avg_throughput < max_average_throughput and 
            avg_throughput > min_average_throughput and min_throughput > 0.2):
            print('skipped {}, {}'.format(avg_throughput, min_throughput))
            continue

        if len(throughputs)*5 < 10*60:
            print('skipped short file {}'.format(len(throughputs)*5))

        c += 1
        traces += t.tolist()

        out_file = 'trace_' + '_'.join(k)
        out_file = out_file.replace(':', '-')
        out_file = out_file.replace('/', '-')

        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)

        out_file = OUTPUT_PATH + out_file
        with open(out_file, 'w') as f:
            for i in bw_measurements[k]:
                f.write(str(i) + '\n')

    plot_tput_distr(traces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Traces converter")
    parser.add_argument(
        "--file",
        help='traces file'
    )
    parser.add_argument(
        "-maxat",
        "--max_average_throughput",
        default=3.0,
        type=float
    )
    parser.add_argument(
        "-minat",
        "--min_average_throughput",
        default=0.2,
        type=float
    )

    args = parser.parse_args()

    main(args.file, args.min_average_throughput, args.max_average_throughput)