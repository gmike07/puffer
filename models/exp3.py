import numpy as np
from models.helper_functions import fill_default_key, fill_default_key_conf, get_config


class Exp3:
    def __init__(self, num_clients, model_config):

        self.max_weight_value = int(fill_default_key(model_config, 'max_weight_exp3', 1e6))
        self.num_clients = num_clients
        self.last_actions = [None for _ in range(num_clients)]
        self.num_actions = len(get_config()['ccs'])
        self.weights = np.ones(self.num_actions)
        self.gamma = fill_default_key(model_config, 'exp3_explore_parameter', 0.1)
        self.probabilites = self.calc_probabilities()
        self.should_load = fill_default_key(model_config, 'should_load_exp3', True)
        self.should_clear_weights = fill_default_key(model_config, 'should_clear_weights', False)
        self.save_name = fill_default_key(model_config, 'exp3_save_name', "exp3")
        self.save_path = f"{fill_default_key_conf(model_config, 'exp3_model_path')}{self.save_name}.npy"
        print('created exp3')

    def clear(self):
        self.last_actions = [None for _ in range(self.num_clients)]
        if self.should_clear_weights:
            self.weights = np.ones(self.num_actions)
        else:
            self.save()
    
    def normalize_qoe(self, qoe, min_value=-1000, max_value=60):
        qoe = max(qoe, min_value) # qoe >= min_value
        qoe = min(qoe, max_value) # qoe in [min_value, max_value]
        return (qoe - min_value) / (max_value - min_value)

    def save(self):
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
        qoe = self.normalize_qoe(state['qoe'])
        action_chosen = self.last_actions[state['server_id']]
        if action_chosen is None:
            return
        scaled_reward = qoe / self.probabilites[action_chosen]
        self.weights[action_chosen] *= np.exp(self.gamma / self.num_actions * scaled_reward)
        if np.max(self.weights) > self.max_weight_value:
            self.weights /= self.max_weight_value
        self.probabilites = self.calc_probabilities()

    def done(self):
        self.save()