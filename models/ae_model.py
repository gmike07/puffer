import torch
from models.helper_functions import fill_default_key_conf, fill_default_key, get_config
from itertools import chain

class AutoEncoder(torch.nn.Module):
    def __init__(self, model_config):
        super(AutoEncoder, self).__init__()
        CONFIG = get_config()
        sizes = fill_default_key_conf(model_config, 'model_network_sizes')
        input_size = CONFIG['nn_input_size'] - len(CONFIG['ccs'])
        self.encoder_sizes = [input_size] + sizes
        decoder_sizes = self.encoder_sizes[::-1]
        activation_function = torch.nn.ReLU
        layers = []
        for i in range(len(self.encoder_sizes) - 1):
            layers.append(torch.nn.Linear(self.encoder_sizes[i], self.encoder_sizes[i + 1]))
            layers.append(activation_function())
        layers = layers[:-1]
        self.encoder_model = torch.nn.Sequential(*layers).double().to(CONFIG['device'])

        layers = []
        for i in range(len(decoder_sizes) - 1):
            layers.append(torch.nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
            layers.append(activation_function())
        layers = layers[:-1]
        self.decoder_model = torch.nn.Sequential(*layers).double().to(CONFIG['device'])

        self.loss_metric = torch.nn.MSELoss().to(device=CONFIG['device'])
        # self.loss_metrics = torch.nn.L1Loss().to(device=CONFIG['device'])

        self.optimizer = torch.optim.Adam(chain(self.encoder_model.parameters(), self.decoder_model.parameters()),
                                            lr=fill_default_key_conf(model_config, 'lr'),
                                            weight_decay=fill_default_key_conf(model_config, 'weights_decay'))
        self.decoder = CONFIG['nn_input_size'] - len(CONFIG['ccs'])
        self.model_name = fill_default_key(model_config, 'ae_model_name', f"ae_weights_abr_{get_config()['abr']}.pt")
        self.model_config = model_config
        print('created ae')

    def forward(self, x):
        return self.decoder_model(self.encoder_model(x))

    def load(self):
        dct = torch.load(fill_default_key_conf(self.model_config, 'weights_path') + self.model_name)
        self.encoder_model.load_state_dict(dct['model_state_dict'])
        print('loaded ae')

    def save(self, path=''):
        if get_config()['test']:
            return
        if path == '':
            path = self.model_name
        torch.save({
            'model_state_dict': self.encoder_model.state_dict()
        }, f"{fill_default_key_conf(self.model_config, 'weights_path')}{path}")

    def get_context(self, x):
        return self.encoder_model(x)

    def done(self):
        pass
