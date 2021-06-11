#!/usr/bin/env python3.7

import argparse
import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import os


def append_to_file(filename, array):
    with open(filename, 'a') as file:
        file.write(str(array)+",\n")


def get_handler_class(args):
    class HandlerClass(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.wfile.write(b"hi!")
            self.end_headers()

        def do_POST(self):
            try:
                content_len = int(self.headers.get('Content-Length'))
                data = self.rfile.read(content_len)
                parsed_data = json.loads(data)
                np.array(parsed_data["datapoint"])
                
                if self.path == "/raw-input":
                    append_to_file(args.raw_weights_file, parsed_data["datapoint"])
                elif self.path == "/ttp-hidden2":
                    append_to_file(args.ttp_weights_file, parsed_data["datapoint"])

                self.send_response(200, "ok")
                self.end_headers()
            except:
                self.send_response(400, "error occurred")
                self.end_headers()

    return HandlerClass


def run_server(args):
    server_address = (args.addr, args.port)
    handler = get_handler_class(args)
    httpd = HTTPServer(server_address, handler)

    print(f"Starting httpd server on {args.addr}:{args.port}")
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
    parser = argparse.ArgumentParser(description="Run a simple HTTP server")
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
        "--raw-weights-file",
        default="./data_points/raw_inputs.log",
        help="Specify the saving file path for raw inputs",
    )
    parser.add_argument(
        "--ttp-weights-file",
        default="./data_points/ttp_hidden2.log",
        help="Specify the saving file path for ttp",
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

    run_server(args)