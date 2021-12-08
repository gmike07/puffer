import torch
from models.helper_functions import fill_default_key_conf, get_config


class NN_Model(torch.nn.Module):
    def __init__(self, config, input_size, output_size, sizes=None, activation_function=torch.nn.ReLU):
        super(NN_Model, self).__init__()
        CONFIG = get_config()
        if sizes is None:
            self.sizes = fill_default_key_conf(config, 'model_network_sizes')
        else:
            self.sizes = sizes
        self.sizes = [input_size] + self.sizes + [output_size]
        layers = []
        for i in range(len(self.sizes) - 1):
            layers.append(torch.nn.Linear(self.sizes[i], self.sizes[i + 1]))
            layers.append(activation_function())
        layers = layers[:-1]
        self.model = torch.nn.Sequential(*layers).double().to(CONFIG['device'])
        self.loss_quality = torch.nn.CrossEntropyLoss().to(device=CONFIG['device'])
        self.loss_metrics = torch.nn.MSELoss().to(device=CONFIG['device'])
        # self.loss_metrics = torch.nn.L1Loss().to(device=CONFIG['device'])
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=fill_default_key_conf(config, 'lr'),
                                          weight_decay=fill_default_key_conf(config, 'weights_decay'))
        print('created NN model')
    
    def forward(self, x):
        return self.model(x)

    def predict(self, state):
        pass

    def update(self, state):
        pass

    def clear(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def done(self):
        self.save()