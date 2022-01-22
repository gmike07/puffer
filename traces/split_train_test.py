import os
import argparse
import numpy as np
import functools
from shutil import copy2


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

    for f in files[:-16 * 8]:
        copy2(f, train_dir)

    for f in files[-16 * 8:]:
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
