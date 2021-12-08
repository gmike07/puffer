from models.helper_functions import get_config, fill_default
from models.basic_models import ConstantModel, RandomModel, IdModel
from models.exp3 import Exp3
from models.sl_trainer import SLTrainer
from models.ae_trainer import AETrainer
from models.rl_trainer import RLTrainer
from models.cluster_trainer import ClusterTrainer
from models.rl_model import RLModel
from models.srl_model import SRLModel
from models.exp3_kmeans import Exp3Kmeans


#models: [constant*, random, idModel, exp3, resettingExp3, sl, ae, slTrainer + random \ idmodel, aeTrainer + random \ idmodel, 
#           rl, srl, srlAE, 'contextlessExp3Kmeans', 'exp3Kmeans', 'exp3KmeansAutoEncoder', 'contextlessClusterTrainer', 'SLClusterTrainer', 'AEClusterTrainer', 'stackingModel']


class StackingModelsServer:
    def __init__(self, models_data):
        models_data = fill_default(models_data, get_config()['test_models'])
        print(models_data)
        self.models = [create_model(len(models_data), model_data) for model_data in models_data]
        print('created stacking model')
    
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

        if model_name in ['resettingExp3', 'exp3']:
            return Exp3(num_clients, conf)

        if model_name == 'SLTrainer':
            return SLTrainer(num_clients, conf, create_model(num_clients, helper_model))

        if model_name == 'AETrainer':
            return AETrainer(conf, create_model(num_clients, helper_model))

        if model_name == 'rl':
            return RLTrainer(RLModel(num_clients, conf))
        
        if model_name in ['srl', 'srlAE']: # no reason to support srlContextless
            return RLTrainer(SRLModel(num_clients, conf))
        
        if model_name in ['contextlessExp3Kmeans', 'exp3Kmeans', 'exp3KmeansAutoEncoder']:
            return Exp3Kmeans(num_clients, conf)
        
        if model_name == 'stackingModel':
            return StackingModelsServer(conf['models'])
        
        if model_name in ['contextlessClusterTrainer', 'SLClusterTrainer', 'AEClusterTrainer']:
            return ClusterTrainer(conf, create_model(num_clients, helper_model))
        
        print(model_name)