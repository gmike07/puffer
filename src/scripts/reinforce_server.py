#!/usr/bin/env python3.7

import argparse
from email.message import Message
from http import server
import json
import torch
import torch.nn.functional as F
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import time
from threading import Thread
from queue import Queue
import os

DEVICE = torch.device('cpu')
MEASURES = Queue()
BATCH_SIZE = 3
MIN_MEASUREMENTS = 10

ROUNDS_TO_SAVE = 1
SLEEP_SEC = 5
CPP_BASE_DIR = './weights/policy/cpp/'
PYTHON_BASE_DIR = './weights/policy/python/'
LOGS_FILE = './weights/policy/reinforce_server_logs.txt'
VERSION = 1

class Model:
    DIM_IN = 20 * 64
    DIM_H1 = 64
    DIM_H2 = 64
    DIM_OUT = 10
    WEIGHT_DECAY = 1e-4
    LEARNING_RATE = 1e-4
    GAMMA = 1e-4

    def __init__(self):
        self.model = torch.nn.Sequential(
            torch.nn.Linear(Model.DIM_IN, Model.DIM_OUT)
        ).double().to(device=DEVICE)
        self.loss_fn = torch.nn.CrossEntropyLoss().to(device=DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=Model.LEARNING_RATE,
                                          weight_decay=Model.WEIGHT_DECAY)

    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=1)

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + Model.GAMMA**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
        #     discounted_rewards.std() + 1e-9)  # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

    def predict(self, state):
        state = torch.from_numpy(
            state).double().unsqueeze(0).to(device=DEVICE)
        probs = self.forward(state)
        highest_prob_action = np.random.choice(
            self.DIM_OUT, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

    def load(self, model_path):
        # check if file exists
        files = list(sorted(os.listdir(model_path), key=lambda x: int(x[x.index('_')+1:x.index('.')])))
        if len(files) == 0:
            return
        
        # load weights
        last_weights = model_path + "/" + files[-1]
        checkpoint = torch.load(last_weights)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # set version
        global VERSION
        VERSION = int(last_weights[last_weights.index('_') + 1 : last_weights.index('.')])
        VERSION += 1

    def save(self, model_path):
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, model_path)

    def save_cpp_model(self, model_path):
        example = torch.rand(1, Model.DIM_IN).double()
        traced_script_module = torch.jit.trace(self.model, example)
        traced_script_module.save(model_path)


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
            version, state, qoe = parsed_data['version'], parsed_data['state'], parsed_data['qoe']

            # print('got version: ', version, '; curr version: ', VERSION)

            if version < VERSION:
                print(str(version) + " expected " + str(VERSION))
                self.send_response(409, "update weights")
                self.end_headers()
                return

            state = np.array(state, np.double)

            global MEASURES
            MEASURES.put({"state": state, "qoe": qoe})
            self.send_response(200, "ok")
            self.end_headers()
        except:
            self.send_response(400, "error occurred")
            self.end_headers()


def run(server_class=HTTPServer, addr="localhost", port=8200):
    server_address = (addr, port)

    handler = HandlerClass
    httpd = server_class(server_address, handler)

    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()


def remove_files_but_last(path):
    files = list(sorted(os.listdir(path), key=lambda x: int(x[x.index('_')+1:x.index('.')])))
    for f in files[:-1]:
        os.remove(path + f)


def train_model():
    model = Model()
    model.load(PYTHON_BASE_DIR)
    total_measurements = []
    rounds_to_save = ROUNDS_TO_SAVE 
    gradients = 0

    while True:
        global MEASURES
        time.sleep(SLEEP_SEC)

        while not MEASURES.empty() and len(total_measurements) < MIN_MEASUREMENTS:
            total_measurements.append(MEASURES.get())
        
        print(len(total_measurements))
        rounds_to_save -= 1

        if len(total_measurements) < MIN_MEASUREMENTS:
            continue

        # select batch and update weights
        measures = np.array(total_measurements)
        indices = np.random.choice(np.arange(measures.size), BATCH_SIZE)
        measures_batch = measures[indices]
        measures = np.delete(measures, indices)

        total_measurements = list(measures)

        states = list(map(lambda s: s["state"], measures_batch))
        rewards = list(map(lambda s: s["qoe"], measures_batch))

        log_probs = []
        for state in states:
            _, log_prob = model.predict(state)
            log_probs.append(log_prob)

        model.update_policy(rewards, log_probs)
        gradients += 1

        # save weights
        if rounds_to_save <= 0:
            global VERSION  

            filename = 'weights_' + str(VERSION) + '.pt'
            
            model.save_cpp_model(CPP_BASE_DIR + filename)
            model.save(PYTHON_BASE_DIR + filename)

            remove_files_but_last(CPP_BASE_DIR)
            remove_files_but_last(PYTHON_BASE_DIR)

            total_measurements = []
            MEASURES = Queue()
            rounds_to_save = ROUNDS_TO_SAVE
            VERSION += 1

            with open(LOGS_FILE, 'w') as logs_file:
                logs_file.write(f"Version: {VERSION}. Num of calculated gradients: {gradients}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a simple HTTP server")
    parser.add_argument(
        "-l",
        "--listen",
        default="localhost",
        help="Specify the IP address on which the server listens",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8200,
        help="Specify the port on which the server listens",
    )
    args = parser.parse_args()

    server_thread = Thread(target=lambda: run(
        addr=args.listen, port=args.port))
    model_thread = Thread(target=lambda: train_model())

    model_thread.start()
    server_thread.start()

    model_thread.join()
    server_thread.join()
