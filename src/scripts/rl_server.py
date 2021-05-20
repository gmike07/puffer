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

DEVICE = torch.device('cpu')
MEASURES = []
BATCH_SIZE = 3
MIN_MEASUREMENTS = 10

ROUNDS_TO_SAVE = 1
SLEEP_SEC = 5
CPP_BASE_DIR = '/home/csuser/puffer/ttp/policy/'
VERSION = 1

class Model:
    DIM_IN = 20 * 64
    DIM_H1 = 64
    DIM_H2 = 64
    DIM_OUT = 10
    WEIGHT_DECAY = 1e-4
    LEARNING_RATE = 1e-4
    GAMMA = 1e-4
    # VERSION = 1

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
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def save(self, model_path):
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, model_path)

    def save_cpp_model(self, model_path):
        example = torch.rand(1, Model.DIM_IN).double()
        traced_script_module = torch.jit.trace(self.model, example)
        traced_script_module.save(model_path)


def wrap(measures):
    class HandlerClass(BaseHTTPRequestHandler):
        def do_GET(self):
            self.wfile.write(self._html("hi!"))

        def do_POST(self):
            content_len = int(self.headers.get('Content-Length'))
            data = self.rfile.readlines()
            print(data)
            parsed_data = json.loads(data[0])
            version, state, qoe = parsed_data['version'], parsed_data['state'], parsed_data['qoe']

            # print('got version: ', version, '; curr version: ', VERSION)

            if version < VERSION:
                self.send_response(200)
                return

            state = np.array(state, np.double)
            measures.put({"state": state, "qoe": qoe})

            self.send_response(200)

    return HandlerClass


def run(q, server_class=HTTPServer, addr="localhost", port=8000):
    server_address = (addr, port)

    handler = wrap(q)
    httpd = server_class(server_address, handler)

    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()


def train_model(q):
    model = Model()
    total_measurements = []
    rounds_to_save = ROUNDS_TO_SAVE 

    while True:
        time.sleep(SLEEP_SEC)

        while not q.empty() and len(total_measurements) < MIN_MEASUREMENTS:
            total_measurements.append(q.get())
        
        print(len(total_measurements))

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

        rounds_to_save -= 1

        # save weights
        if rounds_to_save <= 0:
            global VERSION  
            VERSION += 1

            rounds_to_save = ROUNDS_TO_SAVE
            model.save_cpp_model(CPP_BASE_DIR + 'weights_' + str(VERSION) + '.pt')
            # version += 1
            total_measurements = []
            q = Queue()


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
        default=8000,
        help="Specify the port on which the server listens",
    )
    args = parser.parse_args()
    # run(addr=args.listen, port=args.port)

    q = Queue()
    server_thread = Thread(target=lambda: run(
        q, addr=args.listen, port=args.port))
    model_thread = Thread(target=lambda: train_model(q))

    model_thread.start()
    server_thread.start()

    model_thread.join()
    server_thread.join()
