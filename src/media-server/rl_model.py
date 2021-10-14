#!/usr/bin/env python3.7
import argparse
from numpy.core.fromnumeric import squeeze
import yaml
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpickle as npl
import numpy as np
import pandas as pd

class RLModel(torch.nn.Module):
    def __init__(self, CONFIG):
        super(Model, self).__init__()
        self.output_size = len(CONFIG['ccs'])
        sizes = [CONFIG['input_size']] + CONFIG['network_sizes'] + [self.output_size]
        activation_layer = lambda: torch.nn.ReLU()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation_layer())
        self.model = torch.nn.Sequential(*layers).double().to(CONFIG['device'])
        self.loss_quality = torch.nn.CrossEntropyLoss().to(device=CONFIG['device'])
        self.loss_metrics = torch.nn.MSELoss().to(device=CONFIG['device'])
        # self.loss_metrics = torch.nn.L1Loss().to(device=CONFIG['device'])

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=CONFIG['lr'],
                                          weight_decay=CONFIG['weights_decay'])
        self.CONFIG = CONFIG

    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=1)


class ShallowRLModel(torch.nn.Module):
    def __init__(self, sl_model, input_size, CONFIG):
        super(Model, self).__init__()
        self.output_size = len(CONFIG['ccs'])
        self.input_size = input_size
        self.sl_model = sl_model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, self.output_size)
        ).double().to(CONFIG['device'])
        self.loss_quality = torch.nn.CrossEntropyLoss().to(device=CONFIG['device'])
        self.loss_metrics = torch.nn.MSELoss().to(device=CONFIG['device'])
        # self.loss_metrics = torch.nn.L1Loss().to(device=CONFIG['device'])

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=CONFIG['lr'],
                                          weight_decay=CONFIG['weights_decay'])
        self.CONFIG = CONFIG

    def forward(self, x):
        x = self.model(self.sl_model(x))
        return F.softmax(x, dim=1)



def save_cpp_model(model, model_path):
    example = torch.rand(1, model.CONFIG['input_size']).double()
    traced_script_module = torch.jit.trace(model.model, example, check_trace=False)
    traced_script_module.save(model_path)

def save_model(model, model_path):
    torch.save({'model_state_dict': model.model.state_dict()}, 
        f"{model.CONFIG['weights_path']}{model_path}")

def predict(model, state):
    state = torch.from_numpy(state).double().unsqueeze(0).to(device=model.CONFIG['device'])
    probs = model.forward(state)
    highest_prob_action = np.random.choice(model.output_size, p=np.squeeze(probs.detach().numpy()))
    log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
    return highest_prob_action, log_prob

def update_policy(model, rewards, log_probs):
    discounted_rewards = []
    gamma_discounts = model.CONFIG['gamma'] ** np.arange(len(rewards))
    discounts = gamma_discounts * rewards
    discounts = np.cumsum(discounts[::-1])[::-1] / gamma_discounts
    discounted_rewards = torch.tensor(discounted_rewards)
    policy_gradient = -log_probs * discounted_rewards
    model.optimizer.zero_grad()
    policy_gradient = policy_gradient.sum()
    policy_gradient.backward()
    model.optimizer.step()





DEVICE = torch.device('cpu')
MEASURES = Queue()
BATCH_SIZE = 3
MIN_MEASUREMENTS = 10
SLEEP_SEC = 5


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
    model = RLModel()
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
