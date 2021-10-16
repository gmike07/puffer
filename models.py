from numpy.core.numeric import base_repr
from config_creator import CONFIG, get_config
import torch
import torch.nn.functional as F
import numpy as np
import pickle


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
    def __init__(self, output_type='ssim'):
       super(SL_Model, self).__init__(get_config()['nn_input_size'], OUTPUT_DIMS[output_type])
       self.CONFIG = get_config()
       self.buffer_coef, self.change_coef = self.CONFIG['buffer_length_coef'], self.CONFIG['quality_change_qoef']
       self.scoring_type = self.CONFIG['scoring_function_type']
       self.output_size = OUTPUT_DIMS[output_type]
       self.actions = create_actions()
    
    def calc_score(self, state, action):
        x = super.forward(merge_state_actions(state, action))
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
        dct = torch.load(self.CONFIG['weights_path'] + self.CONFIG['sl_model_name'])
        self.model.load_state_dict(dct['model_state_dict'])

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, f"{CONFIG['weights_path']}{path}")


class ContextModel(torch.nn.Module):
    def __init__(self, base_model):
        super(ContextModel, self).__init__()
        self.base_model = base_model
        self.base_model.eval()
        self.context_layers = set(get_config()['context_layers'])
        self.actions = create_actions()

    def forward(self, x):
        context_layers = []
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
    def __init__(self, num_clients, should_clear_weights=False, is_training=True):
        self.max_weight_value = 1e6
        self.num_clients = num_clients
        self.last_actions = [None for _ in range(num_clients)]
        self.num_actions = len(get_config()['ccs'])
        self.weights = np.ones(self.num_actions)
        self.gamma = get_config()['exp3_explore_parameter']
        self.should_clear_weights = should_clear_weights
        self.probabilites = self.calc_probabilities()
        self.is_training = is_training

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

    def save(self, path=''):
        CONFIG = get_config()
        if path == '':
            path = f"{CONFIG['exp3_model_path']}exp3_{CONFIG['num_clusters']}_{CONFIG['context_layers']}.npy"
        np.save(path, self.weights)

    def load(self, path=''):
        CONFIG = get_config()
        if path == '':
            path = f"{CONFIG['exp3_model_path']}exp3_{CONFIG['num_clusters']}_{CONFIG['context_layers']}.npy"
        self.weights = np.load(path)
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

    
class Exp3Kmeans:
    def __init__(self, num_clients, should_clear_weights=False, is_training=True):
        self.kmeans = load_cluster()
        self.exp3_contexts = [Exp3(num_clients, should_clear_weights, is_training) for _ in range(get_config()['num_clusters'])]
        self.context_model = ContextModel(SL_Model())
        self.context_model.load()

    def predict(self, state):
        context_id = self.kmeans.predict(self.generate_context(state))
        return self.exp3_contexts[context_id].predict(state)

    def update(self, state):
        context_id = self.kmeans.predict(self.generate_context(state))
        self.exp3_contexts[context_id].update(state)

    def clear(self):
        for exp3 in self.exp3_contexts:
            exp3.clear()

    def save(self):
        for i, exp3 in enumerate(self.exp3_contexts):
            exp3.save(f"{CONFIG['exp3_model_path']}exp3_{CONFIG['num_clusters']}_{CONFIG['context_layers']}_{i}.npy")

    def load(self):
        for i, exp3 in enumerate(self.exp3_contexts):
            exp3.load(f"{CONFIG['exp3_model_path']}exp3_{CONFIG['num_clusters']}_{CONFIG['context_layers']}_{i}.npy")

    def generate_context(self, state):
        return self.context_model.generate_context(torch.from_numpy(state['state'].reshape(1, -1))).reshape(-1)


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


