from models.helper_functions import get_config, fill_default
from models.basic_models import ConstantModel, RandomModel, IdModel
from models.exp3 import Exp3
from models.dnn_trainer import DNNTrainer
from models.ae_trainer import AETrainer
from models.rl_trainer import RLTrainer
from models.cluster_trainer import ClusterTrainer
from models.rl_model import DRL
from models.reinforce_model import REINFORCE
from models.exp3_kmeans import Exp3Kmeans
from models.dnn_model import DNN
from models.helper_functions import fill_default_key_conf


class StackingModelsServer:
    def __init__(self, models_data):
        self.model_names = fill_default_key_conf(models_data, 'models')
        print(self.model_names)
        self.models = [create_model(len(self.model_names), model_data) for model_data in self.model_names]
        self.path = fill_default_key_conf(models_data, 'scoring_path')
        self.counter = 0
        self.open_files()
        print('created stacking model')
    
    def open_files(self):
        self.files = [open(f"{self.path}cc_score_{i}_abr_{get_config()['abr']}_{self.counter}_{self.model_names[i]}.txt",'w') for i in range(len(self.models))]
        self.counter += 1

    def predict(self, state):
        self.files[state['server_id']].write(f"{state['qoe']}\n")
        return self.models[state['server_id']].predict(state)

    def update(self, state):
        return self.models[state['server_id']].update(state)

    def clear(self):
        for model in self.models:
            model.clear()
        for file in self.files:
            file.close()
        self.open_files()

    def save(self):
        if get_config()['test']:
            return
        for model in self.models:
            model.save()

    def load(self):
        for model in self.models:
            print('loading ', type(model))
            model.load()
    
    def done(self):
        self.save()


def create_model(num_clients, model_name, helper_model=''):
        conf = get_config()['all_models_config'][model_name]
        if model_name.startswith('constant'):
            return ConstantModel(conf)
        
        if model_name == 'random':
            return RandomModel(conf)
        
        if model_name == 'idModel':
            return IdModel(conf)

        if model_name == 'dnn':
            return DNN(conf)

        if model_name in ['resettingExp3', 'exp3']:
            return Exp3(num_clients, conf)

        if model_name == 'DNNTrainer':
            return DNNTrainer(num_clients, conf, create_model(num_clients, helper_model))

        if model_name == 'AETrainer':
            return AETrainer(conf, create_model(num_clients, helper_model))

        if model_name == 'DRL':
            return RLTrainer(DRL(num_clients, conf))
        
        if model_name in ['REINFORCE', 'REINFORCE_AE']: # no reason to support srlContextless
            return RLTrainer(REINFORCE(num_clients, conf))
        
        if model_name in ['input_k_means', 'DNN_k_means', 'AE_k_means', 'Morpheus']:
            return Exp3Kmeans(num_clients, conf)
        
        if model_name == 'stackingModel':
            return StackingModelsServer(conf)
        
        if model_name in ['inputClusterTrainer', 'DNNClusterTrainer', 'AEClusterTrainer']:
            return ClusterTrainer(conf, create_model(num_clients, helper_model))
        
        print(model_name)