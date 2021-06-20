#!/usr/bin/env python3.7

import argparse
import json
import pickle
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
from scripts.clustering import DELTA, normalize

from scripts.exp3.exp3 import Exp3KMeans


def load_kmeans(kmeans_path):
    clusters_path = kmeans_path + "clusters.pkl"
    normalization_weights_path = kmeans_path + "normalized.npy"

    with open(clusters_path, 'rb') as f:
        kmeans = pickle.load(f)

    with open(normalization_weights_path, 'rb') as f:
        normalization_weights = np.load(f)

    return kmeans, normalization_weights[0], normalization_weights[1]


class Exp3Server:
    def __init__(self, kmeans_dir, num_of_arms):
        kmeans, self.mean, self.std = load_kmeans(kmeans_dir)
        self.exp3 = Exp3KMeans(num_of_arms=num_of_arms, kmeans=kmeans)

    def _prepare_input(self, raw_inputs, buffer_size, last_format):
        mpc = np.array([buffer_size, last_format])
        X = np.hstack([(1-DELTA)*raw_inputs, DELTA*mpc])
        normalized_X = (X - self.mean[np.newaxis, :]) / self.std[np.newaxis, :]
        return normalized_X

    def update_exp3(self):
        pass

    def __get_handler_class(outer_self):
        class HandlerClass(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()

            def do_POST(self):
                try:
                    content_len = int(self.headers.get('Content-Length'))
                    data = self.rfile.read(content_len)
                    parsed_data = json.loads(data)

                    datapoint = outer_self._prepare_input(parsed_data["datapoint"], parsed_data["buffer_size"], parsed_data["last_format"])

                    if self.path == "/get-bitrate":
                        bitrate = outer_self.exp3.predict(datapoint)
                        self.send_response(200, bitrate)
                    elif self.path == "/update":
                        outer_self.exp3.update(datapoint, parsed_data["last_arm"], parsed_data["reward"])
                        self.send_response(200, "ok")
                    self.end_headers()
                except:
                    self.send_response(400, "error occurred")
                    self.end_headers()

        return HandlerClass

    def run_server(self, addr, port):
        server_address = (addr, port)
        handler = self.__get_handler_class()
        httpd = HTTPServer(server_address, handler)

        print(f"Starting httpd server on {addr}:{port}")
        httpd.serve_forever()


def check_dir(filepath, force):
    dir = os.path.dirname(filepath)
    if not os.path.isdir(dir):
        os.mkdir(dir)
    if os.path.exists(filepath) and not force:
        raise Exception("File exists")
    else:
        with open(filepath, 'w') as file:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp3 server")
    parser.add_argument(
        "--addr",
        default="localhost",
        help="Specify the IP address on which the server listens",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Specify the port on which the server listens",
    )
    parser.add_argument(
        "--kmeans-dir",
        default="./weights/kmeans/",
        help="kmeans weights parent dir",
    )
    parser.add_argument(
        "--ttp-weights-file",
        default="./data_points/ttp_hidden2.npy",
        help="Specify the saving file path for ttp",
    )
    parser.add_argument(
        "--num-of-arms",
        default=10,
        type=int
    )
    parser.add_argument(
        "-f",
        "--force",
        default=False,
        action='store_true'
    )
    args = parser.parse_args()

    check_dir(args.raw_weights_file, args.force)
    check_dir(args.ttp_weights_file, args.force)

    exp3Server = Exp3Server(kmeans_dir=args.kmeans_dir,
                            num_of_arms=args.num_of_arms)
    exp3Server.run_server(args.addr, args.port)
