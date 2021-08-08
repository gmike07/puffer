import os
import argparse
import numpy as np
import functools
from shutil import copy2


def concat_files(paths, num_of_traces, output_dir, trace_len):
    for i in range(num_of_traces):
        offset = 0
        str_data = ''
        while offset < trace_len*1000:
            with open(paths[i], 'r') as p:
                data = p.read()

            converted_data = list(
                map(lambda a: int(a), data.split('\n')[:-1]))
            offset_data = offset + np.array(converted_data)

            str_data += functools.reduce(
                lambda a, b: str(a)+'\n'+str(b), offset_data)
            str_data += '\n'

            offset = offset_data[-1]
            i += 1

        output_file = output_dir + str(i) + ".com"
        with open(output_file, 'w') as f:
            f.write(str_data)


def main(traces_dir, output_dir):
    files = [traces_dir + file for file in os.listdir(traces_dir)]
    np.random.shuffle(files)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_dir = output_dir + "train"
    test_dir = output_dir + "test"

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for f in files[:int(0.8*len(files))]:
        copy2(f, train_dir)

    for f in files[int(0.8*len(files)):]:
        copy2(f, test_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Traces converter")
    parser.add_argument(
        "--traces",
        default='./mahimahi/',
        help='mahimahi traces dir'
    )
    parser.add_argument(
        "--output",
        default='./final_traces/',
        help='output trace dir'
    )
    args = parser.parse_args()

    main(args.traces, args.output)
