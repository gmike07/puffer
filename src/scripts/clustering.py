# /usr/bin/env python3

import numpy as np
import os
from sklearn.cluster import KMeans
import pickle
import argparse

DELTA = 0.7


def check_dir(saving_dir, force):
    if len(os.listdir(saving_dir)) != 0 and not force:
        raise Exception("File exists")


def read_file(filename):
    datapoints = []
    with open(filename, 'rb') as file:
        while file.tell() < os.path.getsize(filename):
            datapoint = np.load(file)
            datapoints.append(datapoint)

    return np.array(datapoints)


def normalize(input):
    mean = np.mean(input, axis=0)
    std = np.std(input, axis=0)
    std[std == 0] = 1
    normalized_input = (input - mean[np.newaxis, :]) / std[np.newaxis, :]
    return normalized_input, mean, std


def cluster(datapoints_file, buffer_format_file, saving_dir):
    raw_inputs = read_file(datapoints_file)
    raw_inputs, raw_inputs_mean, raw_inputs_std = normalize(raw_inputs)

    mpc = read_file(buffer_format_file)
    mpc, mpc_mean, mpc_std = normalize(mpc)

    assert raw_inputs.shape[0] == mpc.shape[0]

    X = np.hstack([(1-DELTA)*raw_inputs, DELTA*mpc])

    kmeans = KMeans()
    kmeans.fit(X)

    with open(saving_dir + "clusters.pkl", 'wb') as f:
        pickle.dump(kmeans, f)

    np.savetxt(saving_dir + "mean.txt", np.hstack([raw_inputs_mean, mpc_mean]))
    np.savetxt(saving_dir + "std.txt", np.hstack([raw_inputs_std, mpc_std]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run k-means")
    parser.add_argument(
        "--inputs-file",
        default="./data_points/raw_input.npy"
    )
    parser.add_argument(
        "--buffer-format-file",
        default="./data_points/raw_mpc.npy"
    )
    parser.add_argument(
        "--saving-dir",
        default='./weights/kmeans/'
    )
    parser.add_argument(
        "-f",
        "--force",
        default=False,
        action='store_true'
    )
    args = parser.parse_args()

    check_dir(args.saving_dir, args.force)
    cluster(args.inputs_file, args.buffer_format_file, args.saving_dir)
