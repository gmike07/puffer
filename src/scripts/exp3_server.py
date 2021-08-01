#!/usr/bin/env python3.7

import argparse
import json
import pickle
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import os

from numpy.lib.npyio import save
import yaml

from exp3.exp3 import Exp3KMeans


def load_kmeans(kmeans_path):
    clusters_path = kmeans_path + "clusters.pkl"
    mean = np.loadtxt(kmeans_path + "mean.txt")
    std = np.loadtxt(kmeans_path + "std.txt")

    with open(clusters_path, 'rb') as f:
        kmeans = pickle.load(f)

    return kmeans, mean, std


class Exp3Server:
    CHECKPOINT = 500

    def __init__(self, kmeans_dir, num_of_arms, save_path, plots_dir, delta, lr):
        kmeans, self.mean, self.std = load_kmeans(kmeans_dir)
        self.exp3 = Exp3KMeans(num_of_arms=num_of_arms,
                               kmeans=kmeans, save_path=save_path,
                               plots_dir=plots_dir,
                               lr=lr,
                               checkpoint=Exp3Server.CHECKPOINT)
        self.delta = delta
        self.exp3.load()

    def _prepare_input(self, raw_inputs, buffer_size, last_format):
        mpc = np.array([buffer_size, last_format])
        raw_inputs = np.array(raw_inputs, dtype=np.double)
        X = np.hstack([(1-self.delta)*raw_inputs, self.delta*mpc])
        normalized_X = (X - self.mean[np.newaxis, :]) / self.std[np.newaxis, :]
        return normalized_X

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

                    datapoint = outer_self._prepare_input(
                        parsed_data["datapoint"], parsed_data["buffer_size"], parsed_data["last_format"])

                    if self.path == "/get-bitrate":
                        bitrate = outer_self.exp3.predict(datapoint)
                        message = bytes(str(bitrate), 'utf-8')
                        self.send_response(200)
                        self.send_header("Content-Length", len(message))
                        self.end_headers()
                        self.wfile.write(message)
                    elif self.path == "/update":
                        # print(f'curr ver: {outer_self.exp3._version}, got: {parsed_data["version"]}')
                        print(f'reward={parsed_data["reward"]}, arm={parsed_data["arm"]}, context={parsed_data["context_idx"]}')
                        context_idx = parsed_data["context_idx"]

                        updated = outer_self.exp3.update(
                            datapoint,
                            context_idx,
                            parsed_data["arm"],
                            parsed_data["reward"],
                            parsed_data["version"])

                        if not updated:
                            self.send_response(406)
                            self.end_headers()
                            return

                        self.send_response(200)
                        self.end_headers()
                    else:
                        self.send_response(400)
                        self.end_headers()
                except Exception as e:
                    print('exception', e)
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
        "--save-path",
        default='./weights/exp3'
    )
    parser.add_argument(
        "--plots-dir",
        default='./weights/plots'
    )
    parser.add_argument(
        "--yaml-settings",
        default='./src/settings_offline.yml'
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

    with open(args.yaml_settings, 'r') as fh:
        yaml_settings = yaml.safe_load(fh)
    
    delta = float(yaml_settings["experiments"][0]['fingerprint']['abr_config']['delta'])
    lr = float(yaml_settings["experiments"][0]['fingerprint']['abr_config']['learning_rate'])

    exp3Server = Exp3Server(kmeans_dir=args.kmeans_dir,
                            num_of_arms=args.num_of_arms,
                            save_path=args.save_path,
                            plots_dir=args.plots_dir,
                            delta=delta,
                            lr=lr)
    exp3Server.run_server(args.addr, args.port)
