from config_creator import get_config
import torch
import torch.nn.functional as F
import numpy as np

class NN_Model(torch.nn.Module):
    def __init__(self, input_size, output_size, activation_function=torch.nn.ReLU):
        super(NN_Model, self).__init__()
        CONFIG = get_config()
        sizes = [input_size] + CONFIG['model_network_sizes'] + [output_size]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation_function())
        self.model = torch.nn.Sequential(*layers).double().to(CONFIG['device'])
        self.loss_quality = torch.nn.CrossEntropyLoss().to(device=CONFIG['device'])
        self.loss_metrics = torch.nn.MSELoss().to(device=CONFIG['device'])
        # self.loss_metrics = torch.nn.L1Loss().to(device=CONFIG['device'])

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=CONFIG['lr'],
                                          weight_decay=CONFIG['weights_decay'])
    
    def forward(self, x):
        return self.model(x)

    def predict(self, sent_state):
        pass

    def update(self, sent_state):
        pass


OUTPUT_DIMS = {'qoe': 1, 'ssim': 3, 'bit_rate': 3, 'all': 5}

class SL_Model(NN_Model):
    def __init__(self, output_type='ssim'):
       super(SL_Model, self).__init__(get_config()['nn_input_size'], OUTPUT_DIMS[output_type])
       self.CONFIG = get_config()
       ccs = self.CONFIG['ccs']
       self.buffer_coef, self.change_coef = self.CONFIG['buffer_length_coef'], self.CONFIG['quality_change_qoef']
       self.scoring_type = self.CONFIG['scoring_function_type']
       self.output_size = OUTPUT_DIMS[output_type]
       self.actions = [np.cat([np.arange(len(ccs)) == i for _ in range(len(ccs))], axis=0) for i in range(len(ccs))]
    
    def calc_score(self, state, action):
        x = super.forward(np.cat([state, action], axis=1))
        if self.output_size == 1:
            return x
        if self.output_size == 3 or self.scoring_type == 'ssim':
            return x[:, 0] - self.change_coef * x[:, 1] - self.buffer_coef * x[:, 2]
        return x[:, 3] - self.change_coef * x[:, 4] - self.buffer_coef * x[:, 2]

    def predict(self, sent_state):
        best_action = -1
        best_score = 0
        for i, action in enumerate(self.actions):
            score = self.calc_score(sent_state, action)
            if best_action == -1 or best_score < score:
                best_action, best_score = i, score
        return best_action
