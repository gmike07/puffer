import torch
from models.helper_functions import create_actions, fill_default_key_conf, merge_state_actions
import numpy as np
from models.helper_functions import fill_default_key_conf, fill_default_key, get_updated_config_model, get_config
from models.dnn_model import DNN, REBUFFER_INDEX, QOE_SSIM_INDEX, QOE_CHANGE_INDEX

from models.ae_model import AutoEncoder

class ContextModel(torch.nn.Module):
    def __init__(self, model_config):
        super(ContextModel, self).__init__()
        self.model_config = model_config
        self.context_type = fill_default_key(model_config, 'context_type', 'DNN')
        #contextless case
        self.forward_lambda = self.to_torch
        self.generate_context_lambda = self.to_numpy
        self.output_size = get_config()['nn_input_size'] - len(get_config()['ccs'])
        if self.context_type == 'DNN' or self.context_type == 'dnn':
            self.base_model = DNN(get_updated_config_model('DNN', model_config))
            self.base_model.load()
            self.base_model.eval()
            self.context_layers = fill_default_key_conf(model_config, 'context_layers')
            self.context_layers = set(self.context_layers)
            self.actions = create_actions()
            self.forward_lambda = self.forward_dnn
            self.generate_context_lambda = self.generate_context_dnn
            self.output_size = sum(self.base_model.sizes[i + 1] for i in self.context_layers) * len(get_config()['ccs'])
        elif self.context_type == 'ae':
            self.base_model = AutoEncoder(get_updated_config_model('ae', model_config))
            self.base_model.load()
            self.base_model.eval()
            self.forward_lambda = self.forward_ae
            self.generate_context_lambda = self.generate_context_ae
            self.output_size = self.base_model.encoder_sizes[-1]
        elif self.context_type == 'custom':
            self.base_model = DNN(get_updated_config_model('DNN', model_config))
            self.base_model.load()
            self.base_model.eval()
            self.actions = create_actions()
            self.num_actions = len(self.actions)
            self.forward_lambda = self.forward_custom
            self.generate_context_lambda = self.generate_context_custom
            self.output_size = 1
        self.is_custom_context = self.context_type == 'custom'

        print(f'created context model with context type {self.context_type}')

    def to_numpy(self, x):
        if isinstance(x, np.ndarray):
            return x
        return x.detach().cpu().numpy()

    def to_torch(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x

    def forward_ae(self, x):
        return self.base_model(self.to_torch(x))
    
    def generate_context_ae(self, x):
        return self.base_model.get_context(self.to_torch(x)).detach().cpu().numpy()
    
    def generate_context_dnn(self, x):
        x = self.to_numpy(x)
        context_action = []
        for action in self.actions:
            context_action.append(self(merge_state_actions(x, action)))
        return np.hstack(context_action)

    def forward_dnn(self, x):
        x = self.to_torch(x)
        context_layers = []
        if -1 in self.context_layers:
            context_layers.append(x)
        
        for i, module in enumerate(self.base_model.model):
            x = module(x)
            if i % 2 == 0 and (i // 2) in self.context_layers:
                context_layers.append(x)
        context_layers = [x.detach().numpy() for x in context_layers]
        return np.hstack(context_layers)

    def forward(self, x):
        return self.forward_lambda(x)

    def generate_context(self, x):
        return self.generate_context_lambda(x)

    def load(self):
        if self.context_type == 'contextless':
            return
        self.base_model.load()
        self.context_layers = set(fill_default_key_conf(self.model_config, 'context_layers'))
        print(f'loaded context model with context type {self.context_type}')

    def save(self):
        if self.context_type == 'contextless':
            return
        self.base_model.save()

    def done(self):
        pass

    def generate_context_custom(self, x):
        x = self.to_numpy(x)
        context_action = []
        for action in self.actions:
            context_action.append(self.to_numpy(self.base_model(merge_state_actions(x, action))).reshape(-1))
        context_action = np.array(context_action)
        max_ssim = np.argmax(self.base_model.get_ssim(context_action))
        rebuffer = self.base_model.get_rebuffer(context_action)
        max_rebuffer = np.argmax(rebuffer)
        min_rebuffer = np.argmin(rebuffer)
        if context_action[max_rebuffer, REBUFFER_INDEX] <= 0: #no rebuffer:
            max_rebuffer = self.num_actions
            min_rebuffer = 0
        return int(max_rebuffer * self.num_actions * self.num_actions + min_rebuffer * self.num_actions + max_ssim)

    def forward_custom(self, x):
        x = self.to_torch(x)
        return self.to_torch(self.generate_context_custom(x))