import os
import argparse
import numpy as np
import functools
from shutil import copy2


def main(traces_dir, output_dir, seed):
    files = list(sorted(traces_dir + file for file in os.listdir(traces_dir)))

    if seed is not None:
        np.random.seed(int(seed))
    
    np.random.shuffle(files)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_dir = output_dir + "train"
    test_dir = output_dir + "test"

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    split_index = int(len(files) * 0.8)
    for f in files[:split_index]:
        copy2(f, train_dir)

    for f in files[split_index:]:
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
    parser.add_argument(
        "--seed",
        default=None,
        help='seed value'
    )
    args = parser.parse_args()

    main(args.traces, args.output, args.seed)
