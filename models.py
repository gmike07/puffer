from numpy.core.numeric import base_repr
from config_creator import CONFIG, get_config
import torch
import torch.nn.functional as F
import numpy as np
import pickle


def fill_default(variable, default):
    if variable:
        return variable
    else:
        return default


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
    def __init__(self, output_type='ssim', model_name=''):
       super(SL_Model, self).__init__(get_config()['nn_input_size'], OUTPUT_DIMS[output_type])
       self.CONFIG = get_config()
       self.buffer_coef, self.change_coef = self.CONFIG['buffer_length_coef'], self.CONFIG['quality_change_qoef']
       self.scoring_type = self.CONFIG['scoring_function_type']
       self.output_size = OUTPUT_DIMS[output_type]
       self.actions = create_actions()
       self.model_name = fill_default(model_name, self.CONFIG['sl_model_name'])
    
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
    def __init__(self, num_clients, should_clear_weights=False, is_training=True, save_name='', should_load=False):
        self.max_weight_value = 1e6
        self.num_clients = num_clients
        self.last_actions = [None for _ in range(num_clients)]
        self.num_actions = len(get_config()['ccs'])
        self.weights = np.ones(self.num_actions)
        self.gamma = get_config()['exp3_explore_parameter']
        self.should_clear_weights = should_clear_weights
        self.probabilites = self.calc_probabilities()
        self.is_training = is_training
        get_config()['batch_size'] = 1
        self.should_load = should_load

        self.save_name = fill_default(save_name, f"exp3_{CONFIG['num_clusters']}_{CONFIG['context_layers']}")
        self.save_path = f"{CONFIG['exp3_model_path']}{save_name}.npy"

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

    
class Exp3Kmeans:
    def __init__(self, num_clients, should_clear_weights=False, is_training=True, save_name='', cluster_name='', out_type='ssim', sl_model_name=''):
        self.kmeans = load_cluster()
        save_name = fill_default(save_name, f"exp3_{CONFIG['num_clusters']}_{CONFIG['context_layers']}")
        self.exp3_contexts = [Exp3(num_clients, should_clear_weights, is_training, f"{save_name}_{i}") for i in range(get_config()['num_clusters'])]
        self.context_model = ContextModel(SL_Model(output_type=out_type, model_name=sl_model_name))
        self.context_model.load()

    def get_context_id(self, state):
        context = self.generate_context(state)
        answer = self.kmeans.predict([context])
        return answer[0]

    def predict(self, state):
        return self.exp3_contexts[self.get_context_id(state)].predict(state)

    def update(self, state):
        self.exp3_contexts[self.get_context_id(state)].update(state)

    def clear(self):
        for exp3 in self.exp3_contexts:
            exp3.clear()

    def save(self):
        for exp3 in self.exp3_contexts:
            exp3.save()

    def load(self):
        for exp3 in self.exp3_contexts:
            exp3.load()

    def generate_context(self, state):
        return self.context_model.generate_context(torch.from_numpy(state['state'])).reshape(-1)


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


def create_models(num_clients, model_data):
        if model_data['model_name'].startswith('constant'):
            return ConstantModel(model_data['cc_id'])
        if model_data['model_name'] == 'resettingExp3':
            return Exp3(num_clients, model_data['should_clear_weights'], get_config()['training'], model_data['save_name'])
        if model_data['model_name'] == 'exp3Kmeans':
            return Exp3Kmeans(num_clients, model_data['should_clear_weights'], get_config()['training'], model_data['save_name'], 
                    model_data['cluster_name'],
                    model_data['sl_output_type'], model_data['sl_model_name'])
        if model_data['model_name'] == 'sl':
            return SL_Model(output_type=model_data['output_type'], model_name=model_data['sl_model_name'])
        if model_data['model_name'] == 'random':
            return RandomModel()
        print(model_data)
        


class StackingModelsServer:
    def __init__(self, models_data):
        self.models = [create_models(len(models_data), model_data) for model_data in models_data]
    
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
        for i, model in enumerate(self.models):
            model.load()