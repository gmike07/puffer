import numpy as np
from models.helper_functions import fill_default_key, fill_default_key_conf, get_config
from queue import Queue
import time

def fill_default(dct, key, val):
    if key in dct:
        return dct[key]
    return val


class Exp3:
    def __init__(self, num_clients, model_config):

        self.max_weight_value = int(fill_default_key(model_config, 'max_weight_exp3', 1e6))
        self.num_clients = num_clients
        self.last_actions = [None for _ in range(num_clients)]
        self.num_actions = len(get_config()['ccs'])
        self.weights = np.ones(self.num_actions)
        self.gamma = fill_default_key(model_config, 'exp3_explore_parameter', 0.1)
        self.probabilites = self.calc_probabilities()
        self.should_load = fill_default(model_config, 'should_load_exp3', True)
        self.should_clear_weights = fill_default_key(model_config, 'should_clear_weights', False)
        self.save_name = fill_default_key(model_config, 'exp3_save_name', f"exp3_scoring_{get_config()['buffer_length_coef']}")
        self.save_path = f"{fill_default_key_conf(model_config, 'exp3_model_path')}{self.save_name}.npy"

        self.to_update_probability = Queue()
        print('created exp3')

    def clear(self):
        self.last_actions = [None for _ in range(self.num_clients)]
        if self.should_clear_weights:
            self.to_update_probability = Queue()
            self.weights = np.ones(self.num_actions)
        else:
            self.save()

    def save(self):
        if not self.should_load:
            return
        np.save(self.save_path, self.weights)

    def load(self):
        if not self.should_load:
            return
        self.weights = np.load(self.save_path)
        self.probabilites = self.calc_probabilities()
        print('loaded exp3')

    def calc_probabilities(self):
        sum_weights = self.weights.sum()
        return (1 - self.gamma) * self.weights / sum_weights + self.gamma / self.num_actions

    def predict(self, state):
        action = np.random.choice(np.arange(self.num_actions), p=self.probabilites)
        self.last_actions[state['server_id']] = action
        return action

    def update(self, state):
        if get_config()['test']:
            return
        qoe = state["normalized qoe"]
        action_chosen = self.last_actions[state['server_id']]
        if action_chosen is None:
            return
        self.to_update_probability.put((action_chosen, qoe, self.probabilites[:]))
        # if get_config()['test']:
        #     return
        # qoe = state["normalized qoe"]
        # action_chosen = self.last_actions[state['server_id']]
        # if action_chosen is None:
        #     return
        # self.update_probability(action_chosen, qoe, self.probabilites)

    def update_probability(self, action_chosen, qoe, probabilites):
        if action_chosen is None:
            return
        scaled_reward = qoe / probabilites[action_chosen]
        self.weights[action_chosen] *= np.exp(self.gamma / self.num_actions * scaled_reward)
        if np.max(self.weights) > self.max_weight_value:
            self.weights /= self.max_weight_value
        self.probabilites = self.calc_probabilities()

    def update_weights(self):
        if self.to_update_probability.qsize() > 0:
            action_chosen, qoe, probabilites = self.to_update_probability.get()
            self.update_probability(action_chosen, qoe, probabilites)

    def done(self):
        self.save()


def train_exp3(model, event, f=None):
    while not event.is_set():
        if model.to_update_probability.qsize() == 0:
            time.sleep(1)
        model.update_weights()