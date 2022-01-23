from config_creator import CONFIG
from models.cluster_model import ClusterModel
from models.helper_functions import get_updated_config_model, fill_default_key, fill_default_key_conf
from models.exp3 import Exp3
from models.helper_functions import get_config
import numpy as np
import time
import json

class Exp3Kmeans:
    def __init__(self, num_clients, model_config):
        self.cluster_model = ClusterModel(model_config)
        self.cluster_model.load()
        exp3_config = get_updated_config_model('exp3', model_config)
        save_name = fill_default_key(model_config, 'save_name', f"exp3_{self.cluster_model.cluster_name[len('clusters_'):]}_scoring_{get_config()['buffer_length_coef']}")
        self.exp3_contexts = [Exp3(num_clients, exp3_config) for _ in range(self.cluster_model.num_clusters)]
        for i in range(self.cluster_model.num_clusters):
            path = fill_default_key_conf(model_config, 'exp3_model_path')
            self.exp3_contexts[i].save_path = f"{path}{save_name}_{i}.npy"
        self.cluster_counter = np.zeros(self.cluster_model.num_clusters)
        self.exp3_dct_path = f'{self.cluster_model.cluster_path}{self.cluster_model.cluster_name}_dct.json'
        print('created exp3Kmeans')

    def predict(self, state):
        id_ = self.cluster_model.get_cluster_id(state['state'])
        return self.exp3_contexts[id_].predict(state)

    def update(self, state):
        id_ = self.cluster_model.get_cluster_id(state['state'])
        if not get_config()['test']:
            self.cluster_counter[id_] += 1
        self.exp3_contexts[id_].update(state)

    def save_json(self):
        if CONFIG['test']:
            return
        dct = {str(i): list(self.exp3_contexts[i].weights) for i in range(len(self.exp3_contexts))}
        dct['counter'] = list(self.cluster_counter)
        with open(self.exp3_dct_path, 'w') as fp:
            fp.write(json.dumps(dct))
        print(self.exp3_dct_path, dct)

    def clear(self):
        for exp3 in self.exp3_contexts:
            exp3.clear()
        self.save_json()

    def save(self):
        for exp3 in self.exp3_contexts:
            exp3.save()
        self.save_json()

    def load(self):
        for exp3 in self.exp3_contexts:
            exp3.load()
        # self.cluster_counter = np.load(self.cluster_counter_path)
        print('loaded exp3Kmeans')

    def done(self):
        self.save()