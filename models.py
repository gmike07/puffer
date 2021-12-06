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
                self.get_context = lambda x: self.context_model.get_context(torch.from_numpy(x)).detach().cpu().numpy().reshape(-1) if isinstance(x, np.ndarray) else \
                                                self.context_model.get_context(x).detach().cpu().numpy().reshape(-1)

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
        self.cluster_counter = np.zeros(self.cluster_model.num_clusters)
        self.cluster_counter_path = f'{self.cluster_model.cluster_path}{self.cluster_model.cluster_name}_counter.npy'

    def predict(self, state):
        return self.exp3_contexts[self.cluster_model.get_cluster_id(state['state'])].predict(state)

    def update(self, state):
        id = self.cluster_model.get_cluster_id(state['state'])
        if get_config()['training']:
            self.cluster_counter[id] += 1
        self.exp3_contexts[id].update(state)

    def clear(self):
        for exp3 in self.exp3_contexts:
            exp3.clear()

    def save(self):
        for exp3 in self.exp3_contexts:
            exp3.save()
        self.cluster_counter = np.load(self.cluster_counter_path)

    def load(self):
        for exp3 in self.exp3_contexts:
            exp3.load()
        np.save(self.cluster_counter_path, self.cluster_counter)


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
        
        if model_name == 'rl':
            return RL_Trainer_Model(RL_Model(num_clients, conf))
        
        if model_name in ['srl', 'srlAE']:
            return RL_Trainer_Model(SRL_Model(num_clients, conf))
        
        print(model_name)


class RL_Trainer_Model(torch.nn.Module):
    def __init__(self, model):
        super(RL_Trainer_Model, self).__init__()


        self.model = model
        self.CONFIG = self.model.CONFIG
        self.model_name = self.model.model_name
        self.actions = self.model.actions
        self.num_clients = self.model.num_clients
        self.measurements = self.model.measurements
        self.gamma = self.model.gamma
        self.model_path = self.model.model_path
        self.optimizer = self.model.optimizer
        self.input_size = self.model.input_size
    
    def forward(self, x):
        return self.model.forward(x)

    def load(self):
        dct = torch.load(self.model_path + self.model_name)
        self.model.load_state_dict(dct['model_state_dict'])

    def predict(self, sent_state):
        probabilities = self.forward(torch.from_numpy(sent_state['state']))
        return np.random.choice(self.actions, p=probabilities.detach().cpu().numpy().reshape(-1))

    def save(self, path=''):
        if path == '':
            path = self.model_name
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, f"{self.model_path}{path}")
    
    def update(self, state):
        self.measurements[state['server_id']].put(state)

    def clear(self):
        self.measurements = [Queue() for _ in range(self.num_clients)]

    def clear_client_history(self, client_id):
        self.measurements[client_id] = Queue()

    def get_log_highest_probability(self, state):
        probs = self.forward(torch.from_numpy(state))
        highest_prob_action = np.random.choice(self.actions, p=probs.detach().cpu().numpy().reshape(-1))
        return torch.log(probs.squeeze(0)[highest_prob_action])

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()
    


class RL_Model(NN_Model):
    def __init__(self, num_clients, model_config):
        super(RL_Model, self).__init__(get_config()['nn_input_size'] - len(get_config()['ccs']), len(get_config()['ccs']))
        self.CONFIG = get_config()
        self.model_name = fill_default(model_config['rl_model_name'], self.CONFIG['rl_model_name'])
        self.actions = np.arange(len(self.CONFIG['ccs']))
        self.num_clients = num_clients
        self.measurements = [Queue() for _ in range(num_clients)]
        self.gamma = fill_default(model_config['rl_gamma'], CONFIG['rl_gamma'])
        self.model_path = get_config()['rl_weights_path']
        self.input_size = get_config()['nn_input_size'] - len(get_config()['ccs'])

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)
        

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
            self.get_context = lambda x: self.context_model.generate_context(x)
            input_size = sum(self.context_model.base_model.sizes[i + 1] for i in self.context_model.context_layers) * len(CONFIG['ccs'])
        elif CONFIG['context_type'] == 'ae':
            ae_config = get_updated_config_model('ae', model_config)
            self.context_model = AutoEncoder(ae_config)
            self.get_context = lambda x: self.context_model.get_context(x).detach().cpu().numpy()
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
        self.gamma = fill_default(model_config['srl_gamma'], CONFIG['srl_gamma'])
        self.model_path = get_config()['srl_weights_path']
        self.num_clients = num_clients
        self.input_size = input_size


    def forward(self, x):
        return F.softmax(self.model(torch.from_numpy(self.get_context(x))), dim=1)
    

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
            print('loading ', type(model))
            model.load()
