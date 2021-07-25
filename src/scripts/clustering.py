# /usr/bin/env python3

import numpy as np
import os
from sklearn.cluster import KMeans
import pickle
import argparse
import yaml


def check_dir(saving_dir, force):
    if not os.path.isdir(saving_dir):
        os.mkdir(saving_dir)

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


def cluster(datapoints_file, buffer_format_file, saving_dir, delta, clusters):
    raw_inputs = read_file(datapoints_file)
    raw_inputs, raw_inputs_mean, raw_inputs_std = normalize(raw_inputs)

    mpc = read_file(buffer_format_file)
    mpc, mpc_mean, mpc_std = normalize(mpc)

    assert raw_inputs.shape[0] == mpc.shape[0]

    X = np.hstack([(1-delta)*raw_inputs, delta*mpc])

    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(X)

    with open(saving_dir + "clusters.pkl", 'wb') as f:
        pickle.dump(kmeans, f)

    np.savetxt(saving_dir + "mean.txt", np.hstack([raw_inputs_mean, mpc_mean]))
    np.savetxt(saving_dir + "std.txt", np.hstack([raw_inputs_std, mpc_std]))

    return kmeans


def prepare_X(datapoints_file, buffer_format_file, delta):
    raw_inputs = read_file(datapoints_file)
    raw_inputs, _, _ = normalize(raw_inputs)

    mpc = read_file(buffer_format_file)
    mpc, _, _ = normalize(mpc)

    assert raw_inputs.shape[0] == mpc.shape[0]

    X = np.hstack([(1-delta)*raw_inputs, delta*mpc])

    return X


def calc_min_centers_dist(cluster_centers):
    b = cluster_centers.reshape(
        cluster_centers.shape[0], 1, cluster_centers.shape[1])
    clusters_dist = np.sqrt(
        np.einsum('ijk, ijk->ij', cluster_centers-b, cluster_centers-b))
    np.fill_diagonal(clusters_dist, np.amax(clusters_dist))
    return np.amin(clusters_dist)


def binary_search_optimal_n_clusters(X, min_n_clusters=1, max_n_clusters=16, min_clusters_dist=5):
    n_clusters = min_n_clusters + (max_n_clusters - min_n_clusters) // 2

    if min_n_clusters == n_clusters or max_n_clusters == n_clusters:
        return n_clusters

    print(min_n_clusters, n_clusters, max_n_clusters)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    min_dist = calc_min_centers_dist(kmeans.cluster_centers_)
    print(f'min_dist {min_dist}')

    if min_dist < min_clusters_dist:
        return binary_search_optimal_n_clusters(X, min_n_clusters, n_clusters)

    return binary_search_optimal_n_clusters(X, n_clusters, max_n_clusters)


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
        "--yaml-settings",
        default='./src/settings_offline.yml'
    )
    parser.add_argument(
        "-f",
        "--force",
        default=False,
        action='store_true'
    )
    args = parser.parse_args()

    with open(args.yaml_settings, 'r') as fh:
        yaml_settings = yaml.safe_load(fh)

    delta = float(yaml_settings["experiments"][0]
                  ['fingerprint']['abr_config']['delta'])
    clusters = int(yaml_settings["experiments"][0]
                   ['fingerprint']['abr_config']['clusters'])

    check_dir(args.saving_dir, args.force)

    # X = prepare_X(args.inputs_file, args.buffer_format_file, delta)
    # n_clusters = binary_search_optimal_n_clusters(X)
    # print(f'best n_clusters: {n_clusters}')

    kmeans = cluster(args.inputs_file, args.buffer_format_file,
                     args.saving_dir, delta, clusters)
    min_dist = calc_min_centers_dist(kmeans.cluster_centers_)
    print(f'min centers dist {min_dist}')
