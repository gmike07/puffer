import numpy as np
from models.helper_functions import fill_default_key, get_config


class ConstantModel:
    def __init__(self, config):
        self.val = fill_default_key(config, 'cc_id', 0)
        print(f'creatad constant model with value {self.val}')
    
    def predict(self, state):
        return self.val

    def update(self, state):
        pass

    def clear(self):
        pass

    def save(self):
        pass

    def load(self):
        print(f'loaded costant model with value {self.val}')

    def done(self):
        pass


class RandomModel:
    def __init__(self, config):
        self.actions = np.arange(len(get_config()['ccs']))
        print('created random model')
    
    def predict(self, state):
        return np.random.choice(self.actions)

    def update(self, state):
        pass

    def clear(self):
        pass

    def save(self):
        pass

    def load(self):
        print('loaded random model')

    def done(self):
        pass


class IdModel:
    def __init__(self, config):
        self.len_actions = len(get_config()['ccs'])
        print('created id model')
    
    def predict(self, state):
        return state['server_id'] % self.len_actions

    def update(self, state):
        pass

    def clear(self):
        pass

    def save(self):
        pass

    def load(self):
        print('loaded id model')

    def done(self):
        pass