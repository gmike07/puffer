import os
import argparse
import numpy as np
import functools


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


def main(traces_dir, num_of_traces, output_dir, trace_len):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    files = [traces_dir + file for file in os.listdir(traces_dir)]
    num_of_traces_to_select = 10 * num_of_traces

    if len(files) < num_of_traces_to_select:
        raise Exception(
            f"Expected at least {num_of_traces_to_select} traces in trace dir")

    random_files = np.random.choice(files, num_of_traces_to_select)
    concat_files(random_files, num_of_traces, output_dir, trace_len)


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
    parser.add_argument(
        "-n",
        "--num-of-traces",
        default=700,
        type=int,
        help='number of traces to generate'
    )
    parser.add_argument(
        "-l",
        "--trace-len",
        default=600,
        type=int
    )
    args = parser.parse_args()

    main(args.traces, args.num_of_traces, args.output, args.trace_len)
