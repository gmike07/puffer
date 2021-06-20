# /usr/bin/env python3

import numpy as np
import os
from sklearn.cluster import KMeans
import pickle
import argparse

DELTA = 0.7


def check_dir(filepath, force):
    dir = os.path.dirname(filepath)
    if not os.path.isdir(dir):
        os.mkdir(dir)
    if os.path.exists(filepath) and not force:
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


def cluster(datapoints_file, buffer_format_file, saving_clusters_file, saving_normalized):
    raw_inputs = read_file(datapoints_file)
    raw_inputs, raw_inputs_mean, raw_inputs_std = normalize(raw_inputs)

    mpc = read_file(buffer_format_file)
    mpc, mpc_mean, mpc_std = normalize(mpc)

    assert raw_inputs.shape[0] == mpc.shape[0]

    X = np.hstack([(1-DELTA)*raw_inputs, DELTA*mpc])

    kmeans = KMeans()
    kmeans.fit(X)

    with open(saving_clusters_file, 'wb') as f:
        pickle.dump(kmeans, f)

    with open(saving_normalized, 'wb') as f:
        np.save(saving_normalized, [np.hstack([raw_inputs_mean, mpc_mean]),
                                    np.hstack([raw_inputs_std, mpc_std])])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run k-means")
    parser.add_argument(
        "--raw_inputs-file"
    )
    parser.add_argument(
        "--buffer_format_file"
    )
    parser.add_argument(
        "--saving-clusters",
        default='./weights/kmeans/clusters.pkl'
    )
    parser.add_argument(
        "--saving-normalized",
        default='./weights/kmeans/normalized.npy'
    )
    parser.add_argument(
        "-f",
        "--force",
        default=False,
        action='store_true'
    )
    args = parser.parse_args()

    check_dir('./weights/kmeans/kmeans.pkl', True)
    cluster('./data_points/raw_input.npy',
            './data_points/raw_mpc.npy',
            './weights/kmeans/clusters.pkl',
            './weights/kmeans/normalized.npy')

    # check_dir(args.saving_clusters, args.force)
    # check_dir(args.saving_normalized, args.force)
    # cluster(args.raw_inputs_file, args.buffer_format_file, args.saving_clusters, args.saving_normalized)
