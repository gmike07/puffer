from queue import Queue
from models.helper_functions import fill_default_key_conf, get_updated_config_model, get_config
from models.sl_model import SLModel
import torch
from models.ae_trainer import train_ae
import numpy as np


class SLTrainer:
    def __init__(self, num_clients, config, helper_model):
        self.num_clients = num_clients
        self.measurements = [Queue() for _ in range(num_clients)]
        self.clean_data = Queue()
        self.prediction_model = helper_model
        self.num_actions = len(get_config()['ccs'])
        self.qoe_vec_len = 11
        if not get_config()['test']:
            self.model = SLModel(get_updated_config_model('sl', config))
        else:
            self.model = None
        self.sleep_time = fill_default_key_conf(config, 'sleep_sec')
        self.rounds_to_sleep = fill_default_key_conf(config, 'rounds_to_save')
        self.logs_file = fill_default_key_conf(config, 'logs_file')
        self.training = not get_config()['test']
        print('created SLTrainer')

    def predict(self, state):
        return self.prediction_model.predict(state)

    def update(self, state):
        if not self.training:
            return
        self.measurements[state['server_id']].put(state)
        if self.measurements[state['server_id']].qsize() > 1:
            prev_state = self.measurements[state['server_id']].get()['state']
            curr_cc = state['state'][-self.num_actions-self.qoe_vec_len:-self.qoe_vec_len]
            curr_qoe = state['state'][-3:]
            input = torch.from_numpy(np.append(prev_state, curr_cc).reshape(1, -1))
            output = torch.from_numpy(curr_qoe.reshape(1, -1))
            self.clean_data.put((input, output))


    def clear(self):
        self.measurements = [Queue() for _ in range(self.num_clients)]

    def save(self):
        if self.model is not None:
            self.model.save()

    def load(self):
        self.prediction_model.load()
        print('loaded SLTrainer')
    
    def done(self):
        pass

    def update_helper_model(self, helper_model):
        self.prediction_model = helper_model
        self.load()


def train_sl(model, event):
    train_ae(model, event)