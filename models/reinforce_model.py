from numpy.core.defchararray import mod
from models.helper_functions import fill_default_key_conf, fill_default_key, get_updated_config_model, get_config
from models.nn_models import NN_Model
from models.context_model import ContextModel
import torch.nn.functional as F
from queue import Queue
import numpy as np
import torch


class REINFORCE(torch.nn.Module):
    def __init__(self, num_clients, model_config):
        super(REINFORCE, self).__init__()
        self.CONFIG = get_config()
        self.actions = np.arange(len(self.CONFIG['ccs']))
        self.num_clients = num_clients
        self.measurements = [Queue() for _ in range(num_clients)]
        self.gamma = fill_default_key_conf(model_config, 'reinforce_gamma')
        self.model_path = fill_default_key_conf(model_config, 'weights_path')
        self.input_size = get_config()['nn_input_size'] - len(get_config()['ccs'])
        self.context_model = ContextModel(model_config)
        self.context_model.load()
        self.context_model.eval()
        self.model_name = fill_default_key(model_config, 'reinforce_model_name', 
                    f"reinforce_{self.context_model.context_type}_weights_abr_{get_config()['abr']}_scoring_{get_config()['buffer_length_coef']}.pt")
        self.config = model_config
        self.model = NN_Model(model_config, self.context_model.output_size, len(get_config()['ccs']), sizes=[])
        self.optimizer = self.model.optimizer
        self.loss_quality = self.model.loss_quality
        self.loss_metrics = self.model.loss_metrics
        print('created REINFORCE model')

    def forward(self, x):
        x = self.model(torch.from_numpy(self.context_model.generate_context(x)).double())
        # x /= torch.sum(x, dim=1)
        return F.softmax(x, dim=1)