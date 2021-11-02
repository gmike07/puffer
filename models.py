from config_creator import CONFIG, get_config
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.cluster import KMeans
import copy
from itertools import chain
from queue import Queue

def fill_default(variable, default):
    if variable:
        return variable
    else:
        return default

def get_updated_config_model(model_name, config):
    conf = copy.deepcopy(CONFIG['all_models_config'][model_name])
    conf.update(config)
    return conf


def create_actions():
    ccs = get_config()['ccs']
    batch_size = get_config()['batch_size']
    return [torch.from_numpy(np.vstack([np.arange(len(ccs)) == i for _ in range(batch_size)]).astype(np.int)) for i in range(len(ccs))]


def merge_state_actions(state, action):
    return torch.from_numpy(np.hstack([state, action]))


def load_cluster():
    CONFIG = get_config()
    kmeans = None
    with open(f"{CONFIG['saving_cluster_path']}clusters_{CONFIG['num_clusters']}_{CONFIG['context_layers']}.pkl", 'rb') as f:
        kmeans = pickle.load(f)
    return kmeans


class NN_Model(torch.nn.Module):
    def __init__(self, input_size, output_size, sizes=None, activation_function=torch.nn.ReLU):
        super(NN_Model, self).__init__()
        CONFIG = get_config()
        if sizes is None:
            self.sizes = CONFIG['model_network_sizes']
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
                                          lr=CONFIG['lr'],
                                          weight_decay=CONFIG['weights_decay'])
    
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


OUTPUT_DIMS = {'qoe': 1, 'ssim': 3, 'bit_rate': 3, 'all': 5}

class SL_Model(NN_Model):
    def __init__(self, model_config):
       super(SL_Model, self).__init__(get_config()['nn_input_size'], OUTPUT_DIMS[model_config['sl_output_type']])
       self.CONFIG = get_config()
       self.buffer_coef, self.change_coef = self.CONFIG['buffer_length_coef'], self.CONFIG['quality_change_qoef']
       self.scoring_type = self.CONFIG['scoring_function_type']
       self.output_size = OUTPUT_DIMS[model_config['sl_output_type']]
       self.actions = create_actions()
       self.model_name = fill_default(model_config['sl_model_name'], self.CONFIG['sl_model_name'])
    
    def calc_score(self, state, action):
        x = self.model(merge_state_actions(state['state'], action))
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

    def load(self):
        dct = torch.load(self.CONFIG['weights_path'] + self.model_name)
        self.model.load_state_dict(dct['model_state_dict'])

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, f"{get_config()['weights_path']}{path}")


class AutoEncoder(torch.nn.Module):
    def __init__(self, model_config):
        super(AutoEncoder, self).__init__()
        CONFIG = get_config()
        sizes = fill_default(model_config['ae_sizes'], CONFIG['ae_sizes'])
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

        self.loss_metrics = torch.nn.MSELoss().to(device=CONFIG['device'])
        # self.loss_metrics = torch.nn.L1Loss().to(device=CONFIG['device'])

        self.optimizer = torch.optim.Adam(chain(self.encoder_model.parameters(), self.decoder_model.parameters()),
                                            lr=CONFIG['lr'],
                                            weight_decay=CONFIG['weights_decay'])
        self.decoder = CONFIG['nn_input_size'] - len(CONFIG['ccs'])
        self.CONFIG = get_config()
        self.model_name = fill_default(model_config['ae_model_name'], self.CONFIG['ae_model_name'])

    def forward(self, x):
        return self.decoder_model(self.encoder_model(x))

    def load(self):
        dct = torch.load(self.CONFIG['ae_weights_path'] + self.model_name)
        self.model.load_state_dict(dct['model_state_dict'])

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, f"{get_config()['ae_weights_path']}{path}")

    def get_context(self, x):
        return self.encoder_model(x)


class ContextModel(torch.nn.Module):
    def __init__(self, base_model, model_config):
        super(ContextModel, self).__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.context_layers = fill_default(model_config['context_layers'], get_config()['context_layers'])
        self.context_layers = set(self.context_layers)
        self.actions = create_actions()

    def forward(self, x):
        context_layers = []
        if -1 in self.context_layers:
            context_layers.append(x)
        
        for i, module in enumerate(self.base_model.model):
            x = module(x)
            if i % 2 == 0 and (i // 2) in self.context_layers:
                context_layers.append(x)
        context_layers = [x.detach().numpy() for x in context_layers]
        return np.hstack(context_layers)

    def generate_context(self, x):
        context_action = []
        for action in self.actions:
            context_action.append(self(merge_state_actions(x, action)))
        return np.hstack(context_action)

    def load(self):
        self.base_model.load()
        self.context_layers = set(get_config()['context_layers'])

    def save(self):
        self.base_model.save()


class Exp3:
    def __init__(self, num_clients, model_config):

        self.max_weight_value = 1e6
        self.num_clients = num_clients
        self.last_actions = [None for _ in range(num_clients)]
        CONFIG = get_config()
        self.num_actions = len(CONFIG['ccs'])
        self.weights = np.ones(self.num_actions)
        self.gamma = CONFIG['exp3_explore_parameter']
        self.probabilites = self.calc_probabilities()
        self.is_training = CONFIG['training']
        CONFIG['batch_size'] = 1
        self.should_load = model_config['should_load_exp3']
        self.should_clear_weights = model_config['should_clear_weights']
        self.save_name = fill_default(model_config['exp3_save_name'], f"exp3")
        self.save_path = f"{CONFIG['exp3_model_path']}{self.save_name}.npy"

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

    def calc_probabilities(self):
        sum_weights = self.weights.sum()
        return (1 - self.gamma) * self.weights / sum_weights + self.gamma / self.num_actions

    def predict(self, state):
        action = np.random.choice(np.arange(self.num_actions), p=self.probabilites)
        self.last_actions[state['server_id']] = action
        return action

    def update(self, state):
        if not self.is_training:
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



class ClusterModel:
    def __init__(self, model_config):

        CONFIG = get_config()
        contextless = model_config['cluster_contextless']
        self.num_clusters = fill_default(model_config['num_clusters'], CONFIG['num_clusters'])
        context_layers = fill_default(model_config['context_layers'], CONFIG['context_layers'])

        if contextless:
            self.get_context = lambda x: x.reshape(-1) if type(x) == np.ndarray else x.cpu().detach().numpy().reshape(-1)
            self.cluster_name = fill_default(model_config['cluster_name'], f"clusters_{self.num_clusters}_{context_layers}")
        else:
            if fill_default(model_config['cluster_type'], CONFIG['cluster_type']) == 'ae':
                ae_config = get_updated_config_model('ae', model_config)
                self.context_model = AutoEncoder(ae_config)
                self.get_context = lambda x: self.context_model.get_context(x).detach().cpu().numpy().reshape(-1)

                model_name = self.context_model.model_name
            else:
                sl_config = get_updated_config_model('sl', model_config)
                self.context_model = ContextModel(SL_Model(sl_config), sl_config)
                self.get_context = lambda x: self.context_model.generate_context(x).reshape(-1)
                model_name = self.context_model.base_model.model_name
            
            self.cluster_name = fill_default(model_config['cluster_name'], f"clusters_{model_name}_{self.num_clusters}_{context_layers}")
            self.context_model.load()
            self.context_model.eval()
        self.cluster_path = CONFIG['saving_cluster_path']
        self.kmeans = KMeans(n_clusters=self.num_clusters)

    def load(self):
        with open(f"{self.cluster_path}{self.cluster_name}.pkl", 'rb') as f:
            self.kmeans = pickle.load(f)
    
    def save(self):
        with open(f"{self.cluster_path}{self.cluster_name}.pkl", 'wb') as f:
            pickle.dump(self.kmeans, f)

    def get_cluster_id(self, x):
        context = self.get_context(x)
        answer = self.kmeans.predict([context])
        return answer[0]

    def fit(self, X):
        self.kmeans.fit(X)



class Exp3Kmeans:
    def __init__(self, num_clients, model_config):
        self.kmeans = load_cluster()
        CONFIG = get_config()

        cluster_config = copy.deepcopy(CONFIG['all_models_config']['clusterModel'])
        cluster_config.update(model_config)
        CONFIG['batch_size'] = 1
        self.cluster_model = ClusterModel(get_updated_config_model('clusterModel', model_config))
        self.cluster_model.load()
        exp3_config = get_updated_config_model('exp3', model_config)

        save_name = fill_default(model_config['save_name'], f"exp3_{self.cluster_model.cluster_name[len('clusters_'):]}")
        self.exp3_contexts = [Exp3(num_clients, exp3_config) for _ in range(self.cluster_model.num_clusters)]
        for i in range(self.cluster_model.num_clusters):
            self.exp3_contexts[i].save_path = f"{get_config()['exp3_model_path']}{save_name}_{i}.npy"

    def predict(self, state):
        return self.exp3_contexts[self.cluster_model.get_cluster_id(state['state'])].predict(state)

    def update(self, state):
        self.exp3_contexts[self.cluster_model.get_cluster_id(state['state'])].update(state)

    def clear(self):
        for exp3 in self.exp3_contexts:
            exp3.clear()

    def save(self):
        for exp3 in self.exp3_contexts:
            exp3.save()

    def load(self):
        for exp3 in self.exp3_contexts:
            exp3.load()


class ConstantModel:
    def __init__(self, val=1):
        self.val = val
    
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


class RandomModel:
    def __init__(self):
        self.actions = np.arange(len(CONFIG['ccs']))
    
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

class IdModel:
    def __init__(self):
        self.actions = len(CONFIG['ccs'])
    
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


def create_model(num_clients, model_name):
        conf = get_config()['all_models_config'][model_name]
        if model_name.startswith('constant'):
            return ConstantModel(conf['cc_id'])
        
        if model_name in ['resettingExp3', 'exp3']:
            return Exp3(num_clients, conf)
        
        if model_name in ['contextlessExp3Kmeans', 'exp3Kmeans', 'exp3KmeansAutoEncoder']:
            return Exp3Kmeans(num_clients, conf)
        
        if model_name == 'sl':
            return SL_Model(conf)
        
        if model_name == 'random':
            return RandomModel()
        
        if model_name == 'idModel':
            return IdModel()

        if model_name == 'stackingModel':
            return StackingModelsServer(conf['models'])
        
        print(model_name)


class RL_Model(NN_Model):
    def __init__(self, num_clients, model_config):
        super(RL_Model, self).__init__(get_config()['nn_input_size'] - len(get_config()['ccs']), len(get_config()['ccs']))
        self.CONFIG = get_config()
        self.model_name = fill_default(model_config['rl_model_name'], self.CONFIG['rl_model_name'])
        self.actions = np.arange(len(self.CONFIG['ccs']))
        self.measurements = [Queue() for _ in range(num_clients)]

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)

    def predict(self, sent_state):
        return np.random.choice(self.actions, p=self(torch.from_numpy(sent_state['state']).detach().cpu().numpy().reshape(-1)))

    def load(self):
        dct = torch.load(self.CONFIG['rl_weights_path'] + self.model_name)
        self.model.load_state_dict(dct['model_state_dict'])

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, f"{get_config()['rl_weights_path']}{path}")
    
    def update(self, state):
        self.measurements[state['server_id']].put(state)


class SRL_Model(torch.nn.Module):
    def __init__(self, num_clients, model_config):
        super(SRL_Model, self).__init__()
        CONFIG = get_config()
        if CONFIG['context_type'] == 'contextless':
            self.get_context = lambda x: x
            input_size = CONFIG['nn_input_size'] - len(CONFIG['ccs'])
        elif CONFIG['context_type'] == 'sl':
            sl_config = get_updated_config_model('sl', model_config)
            self.context_model = ContextModel(SL_Model(sl_config), sl_config)
            self.get_context = lambda x: self.context_model(x)
            input_size = sum(self.context_model.base_model.sizes[i + 1] for i in self.context_model.context_layers) * len(CONFIG['ccs'])
        elif CONFIG['context_type'] == 'ae':
            ae_config = get_updated_config_model('ae', model_config)
            self.context_model = AutoEncoder(ae_config)
            self.get_context = lambda x: self.context_model.get_context(x)
            input_size = self.context_model.encoder_sizes[-1]
        sizes = [input_size, len(CONFIG['ccs'])]
        activation_function = torch.nn.ReLU
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation_function())
        layers = layers[:-1]
        self.model = torch.nn.Sequential(*layers).double().to(CONFIG['device'])

        self.CONFIG = get_config()
        self.model_name = fill_default(model_config['srl_model_name'], self.CONFIG['srl_model_name'])
        self.actions = np.arange(len(self.CONFIG['ccs']))
        self.measurements = [Queue() for _ in range(num_clients)]

        self.loss_metrics = torch.nn.MSELoss().to(device=CONFIG['device'])
        # self.loss_metrics = torch.nn.L1Loss().to(device=CONFIG['device'])

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=CONFIG['lr'],
                                          weight_decay=CONFIG['weights_decay'])

    def forward(self, x):
        return F.softmax(self.model(self.get_context(x)), dim=1)

    def predict(self, sent_state):
        return np.random.choice(self.actions, p=self(torch.from_numpy(sent_state['state']).detach().cpu().numpy().reshape(-1)))

    def load(self):
        dct = torch.load(self.CONFIG['srl_weights_path'] + self.model_name)
        self.model.load_state_dict(dct['model_state_dict'])

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, f"{get_config()['srl_weights_path']}{path}")
    
    def update(self, state):
        self.measurements[state['server_id']].put(state)


class StackingModelsServer:
    def __init__(self, models_data):
        models_data = fill_default(models_data, get_config()['test_models'])
        print(models_data)
        self.models = [create_model(len(models_data), model_data) for model_data in models_data]
    
    def predict(self, state):
        return self.models[state['server_id']].predict(state)

    def update(self, state):
        return self.models[state['server_id']].update(state)

    def clear(self):
        for model in self.models:
            model.clear()

    def save(self):
        if get_config()['test']:
            return
        for model in self.models:
            model.save()

    def load(self):
        for model in self.models:
            model.load()