import pickle

from numpy.core.defchararray import mod
from models.context_model import ContextModel
from sklearn.cluster import KMeans
from models.helper_functions import fill_default_key, fill_default_key_conf, get_config
import torch

class ClusterModel:
    def __init__(self, model_config):       
        self.context_model = ContextModel(model_config)
        self.context_model.eval()
        self.get_cluster_id = self.get_cluster_id_default
        if self.context_model.is_custom_context:
            self.get_cluster_id = lambda x: self.context_model.generate_context(x)
            self.num_clusters = int(len(self.context_model.actions) ** 3) + len(self.context_model.actions)
            context_layers = []
        else:
            context_layers = fill_default_key_conf(model_config, 'context_layers')
            self.num_clusters = fill_default_key_conf(model_config, 'num_clusters')
            self.kmeans = KMeans(n_clusters=self.num_clusters)
        self.cluster_name = fill_default_key(model_config, 'cluster_name', f"clusters_{self.context_model.context_type}_{get_config()['abr']}_{self.num_clusters}_{context_layers}")
        self.cluster_path = fill_default_key_conf(model_config, 'saving_cluster_path')

        print('created cluster model')

    def load(self):
        if self.context_model.is_custom_context:
            return
        with open(f"{self.cluster_path}{self.cluster_name}.pkl", 'rb') as f:
            self.kmeans = pickle.load(f)
        print('loaded cluster model')
    
    def save(self):
        if self.context_model.is_custom_context:
            return
        with open(f"{self.cluster_path}{self.cluster_name}.pkl", 'wb') as f:
            pickle.dump(self.kmeans, f)

    def get_cluster_id_default(self, x):
        with torch.no_grad():
            context = self.context_model.generate_context(x).reshape(-1)
        answer = self.kmeans.predict([context])
        return answer[0]

    def fit(self, X):
        if self.context_model.is_custom_context:
            return
        self.kmeans.fit(X)