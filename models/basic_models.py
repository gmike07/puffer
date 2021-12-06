import numpy as np
from ..config_creator import get_config
from helper_functions import fill_default_key


class ConstantModel:
    def __init__(self, config):
        self.val = fill_default_key(config, 'cc_id', 0)
    
    def predict(self, state):
        return self.val

    def update(self, state):
        pass

    def clear(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def done(self):
        pass


class RandomModel:
    def __init__(self, config):
        self.actions = np.arange(len(get_config()['ccs']))
    
    def predict(self, state):
        return np.random.choice(self.actions)

    def update(self, state):
        pass

    def clear(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def done(self):
        pass


class IdModel:
    def __init__(self, config):
        self.actions = len(get_config()['ccs'])
    
    def predict(self, state):
        return state['server_id'] % self.len_actions

    def update(self, state):
        pass

    def clear(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def done(self):
        pass