from models.context_model import ContextModel
import numpy as np


class ClusterTrainer:
    def __init__(self, model_config, helper_model):       
        self.context_model = ContextModel(model_config)
        self.prediction_model = helper_model
        self.contexts = np.array([])
        print('created clusterTrainer')

    def load(self):
        self.prediction_model.load()
        print('loaded clusterTrainer')
    
    def save(self):
        pass

    def update(self, state):
        self.contexts = np.append(self.contexts, state['state'].reshape(1, -1))

    def predict(self, state):
        return self.prediction_model.predict(state)
    
    def done(self):
        self.context_model.fit(self.contexts)
        self.context_model.save()
        print('done clusterTrainer')

    def clear(self):
        pass

    def update_helper_model(self, helper_model):
        self.prediction_model = helper_model
        self.load()