from models.nn_models import NN_Model
import torch
from models.helper_functions import fill_default_key_conf, fill_default_key, create_actions, merge_state_actions, get_config
import numpy as np

# OUTPUT_DIMS = {'rebuffer': 1, 'ssim': 3, 'bit_rate': 3, }
QOE_SSIM_INDEX = 0
QOE_CHANGE_INDEX = 1
REBUFFER_INDEX = 2
DEFAULT_MAX_BIN = 20


def get_output_dim(model_config):
    scoring_type = fill_default_key_conf(model_config, 'scoring_function_type')
    if scoring_type in ['ssim', 'bit_rate']:
        return 3 # ssim, ssim_change, rebuffer
    if scoring_type == 'rebuffer':
        return 1 # rebuffer
    if scoring_type == 'bin_rebuffer':
        return fill_default_key_conf(model_config, 'dnn_max_bin') + 1 # bins
    if scoring_type in ['ssim_bin_rebuffer', 'bit_rate_bin_rebuffer']:
        return 2 + fill_default_key_conf(model_config, 'dnn_max_bin') + 1 #ssim, ssim_change, bins


def get_sizes(model_config):   
    scoring_type = fill_default_key_conf(model_config, 'scoring_function_type')
    if scoring_type in ['bin_rebuffer', 'ssim_bin_rebuffer', 'bit_rate_bin_rebuffer']:
        return fill_default_key_conf(model_config, 'dnn_bin_model_network_sizes')



class DNN(NN_Model):
    def entropy_loss(self, x, y):
        return self.entropy(x, torch.flatten(y).type(torch.LongTensor))

    def ssim_entropy_loss(self, x, y):
        return self.norm_loss(x[:, :REBUFFER_INDEX], y[:, :REBUFFER_INDEX] + self.entropy_loss(x[:,REBUFFER_INDEX:], y[:,REBUFFER_INDEX:]))

    def __init__(self, model_config):
       super(DNN, self).__init__(model_config, get_config()['nn_input_size'], get_output_dim(model_config), sizes=get_sizes(model_config))
       self.CONFIG = get_config()
       self.scoring_type = fill_default_key_conf(model_config, 'scoring_function_type')
       self.buffer_coef, self.change_coef = self.CONFIG['buffer_length_coef'], self.CONFIG['quality_change_qoef']
       self.output_size = get_output_dim(model_config)
       self.actions = create_actions()
       self.model_name = fill_default_key(model_config, 'dnn_model_name', f"dnn_weights_abr_{get_config()['abr']}_{self.scoring_type}_scoring_{get_config()['buffer_length_coef']}.pt")
       self.model_config = model_config

       self.bin_size = fill_default_key_conf(model_config, 'dnn_bin_size')
       self.max_bins = fill_default_key_conf(model_config, 'dnn_max_bin')
       if self.scoring_type == 'bin_rebuffer':
            self.model_name = fill_default_key(model_config, 'dnn_model_name', f"dnn_weights_abr_{get_config()['abr']}_{self.scoring_type}_{self.bin_size}_{self.max_bins}_scoring_{get_config()['buffer_length_coef']}.pt")
            self.entropy = torch.nn.CrossEntropyLoss().to(device=get_config()['device'])
            self.loss_metric = self.entropy_loss
    
       elif self.scoring_type in ['ssim_bin_rebuffer', 'bit_rate_bin_rebuffer']:
            self.model_name = fill_default_key(model_config, 'dnn_model_name', f"dnn_weights_abr_{get_config()['abr']}_{self.scoring_type}_{self.bin_size}_{self.max_bins}_scoring_{get_config()['buffer_length_coef']}.pt")
            self.entropy = torch.nn.CrossEntropyLoss().to(device=get_config()['device'])
            self.norm_loss = torch.nn.MSELoss().to(device=get_config()['device'])
            self.loss_metric = self.ssim_entropy_loss
       else:
            self.loss_metric = self.loss_metrics
       print(f'created DNN model with scoring {self.scoring_type}')

    def to_torch(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x
    
    def get_bin_rebuffer(self, x):
        with torch.no_grad():
            x = self.to_torch(x)
            y_predicted = torch.max(x, 1)[1].to(device=get_config()['device'])
            ret = y_predicted.double().numpy()
            for i in range(len(ret)):
                bin_id = ret[i]
                if bin_id == 0:  # the first bin is defined differently
                    ret[i] = 0.25 * self.bin_size
                else:
                    ret[i] = bin_id * self.bin_size
            return ret

    def calc_score(self, state, action):
        with torch.no_grad():
            x = self.model(merge_state_actions(state['state'], action))
            if self.scoring_type == 'rebuffer':
                return -x[:, 0]
            if self.scoring_type == 'bin_rebuffer':
                return self.to_torch(self.get_bin_rebuffer(x))
            if self.scoring_type in ['ssim', 'bit_rate']:
                return 30.0 * x[:, QOE_SSIM_INDEX] - 30.0 * self.change_coef * x[:, QOE_CHANGE_INDEX] - self.buffer_coef * x[:, REBUFFER_INDEX] / 15.0
            if self.scoring_type in ['ssim_bin_rebuffer', 'bit_rate_bin_rebuffer']:
                return 30.0 * x[:, QOE_SSIM_INDEX] - 30.0 * self.change_coef * x[:, QOE_CHANGE_INDEX] - self.buffer_coef * self.to_torch(self.get_bin_rebuffer(x[:, REBUFFER_INDEX:]))

    def predict(self, sent_state):
        best_action = -1
        best_score = 0
        for i, action in enumerate(self.actions):
            score = self.calc_score(sent_state, action)
            if best_action == -1 or best_score < score:
                best_action, best_score = i, score
        return best_action

    def load(self):
        dct = torch.load(fill_default_key_conf(self.model_config, 'weights_path') + self.model_name)
        self.model.load_state_dict(dct['model_state_dict'])
        print('loaded DNN model')

    def save(self, path=''):
        if get_config()['test']:
            return
        if path == '':
            path = self.model_name
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, f"{fill_default_key_conf(self.model_config, 'weights_path')}{path}")
    
    def done(self):
        self.save()


    def get_ssim(self, x):
        if self.scoring_type in ['rebuffer', 'bin_rebuffer']:
            return np.zeros(len(x))
        return x[:, QOE_SSIM_INDEX]
    
    def get_rebuffer(self, x):
        if self.scoring_type == 'rebuffer':
            return x[:, 0]
        if self.scoring_type == 'bin_rebuffer':
            return self.get_bin_rebuffer(x)
        if self.scoring_type in ['ssim_bin_rebuffer', 'bit_rate_bin_rebuffer']:
            return self.get_bin_rebuffer(x[:, REBUFFER_INDEX:])
        return x[:, REBUFFER_INDEX]


    # special discretization: [0, 0.5 * BIN_SIZE)
    # [0.5 * BIN_SIZE, 1.5 * BIN_SIZE), [1.5 * BIN_SIZE, 2.5 * BIN_SIZE), ...
    def discretize_output(self, raw_out):
        # z = np.array(raw_out)
        z = torch.floor((raw_out + 0.5 * self.bin_size) / self.bin_size )
        return torch.clamp(z, min=0, max=self.max_bins)
