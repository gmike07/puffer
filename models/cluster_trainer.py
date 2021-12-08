from models.context_model import ContextModel
from models.cluster_model import ClusterModel
import numpy as np


class ClusterTrainer:
    def __init__(self, model_config, helper_model):       
        self.context_model = ContextModel(model_config)
        self.prediction_model = helper_model
        self.contexts = []
        self.config = model_config
        print('created clusterTrainer')

    def load(self):
        self.prediction_model.load()
        print('loaded clusterTrainer')
    
    def save(self):
        pass

    def update(self, state):
        self.contexts.append(self.context_model.generate_context(state['state']).reshape(-1))

    def predict(self, state):
        return self.prediction_model.predict(state)
    
    def done(self):
        cluster = ClusterModel(self.config)
        cluster.fit(np.array(self.contexts))
        cluster.save()
        print('done clusterTrainer')
        print(np.array(self.contexts).shape)

    def clear(self):
        pass

    def update_helper_model(self, helper_model):
        self.prediction_model = helper_model
        self.load()