from models.helper_functions import fill_default_key_conf, fill_default_key, get_config
from models.nn_models import NN_Model
import torch.nn.functional as F
from queue import Queue
import numpy as np
import torch

class DRL(NN_Model):
    def __init__(self, num_clients, model_config):
        super(DRL, self).__init__(model_config, get_config()['nn_input_size'] - len(get_config()['ccs']), len(get_config()['ccs']),
                                    sizes=fill_default_key(model_config, 'drl_layers', None))
        self.CONFIG = get_config()
        self.model_name = fill_default_key(model_config, 'drl_model_name', 
                        f"drl_weights_abr_{get_config()['abr']}_scoring_{get_config()['buffer_length_coef']}.pt")
        self.actions = np.arange(len(self.CONFIG['ccs']))
        self.num_clients = num_clients
        self.measurements = [Queue() for _ in range(num_clients)]
        self.gamma = fill_default_key_conf(model_config, 'drl_gamma')
        self.model_path = fill_default_key_conf(model_config, 'weights_path')
        self.input_size = get_config()['nn_input_size'] - len(get_config()['ccs'])
        self.config = model_config
        print('created DRL')

    def forward(self, x):
        x = self.model(x)
        return F.softmax(x, dim=1)