from .context import Context
import numpy as np
import os
import shutil

class Exp3KMeans:
    def __init__(self, num_of_arms, kmeans):
        self._kmeans = kmeans
        self._version = 0

        # create contexts
        self._contexts = []
        for cluster in self._kmeans.cluster_centers_:
            context = Context(num_of_arms, cluster)
            self._contexts.append(context)

    def predict(self, datapoint):
        context_idx = self._kmeans.predict(datapoint)
        context_idx = context_idx[0]
        arm = self._contexts[context_idx].predict()
        return arm

    def update(self, datapoint, last_arm, reward):
        context_idx = self._kmeans.predict(datapoint)
        context_idx = context_idx[0]
        self._contexts[context_idx].update(reward, last_arm)

    def save(self, folder_path):
        old_dir = f'{folder_path}/{self._version - 1}'
        if os.path.isdir(old_dir):
            shutil.rmtree(old_dir)

        if os.path.isdir(f'{folder_path}/{self._version}'):
            shutil.rmtree(f'{folder_path}/{self._version}')

        os.mkdir(f'{folder_path}/{self._version}')
        for i, context in enumerate(self._contexts):
            base_dir = f'{folder_path}/{self._version}/{i}'
            os.mkdir(base_dir)
            np.savetxt(f'{base_dir}/cluster.txt', context.cluster)
            np.savetxt(f'{base_dir}/weights.txt', context.weights)
            np.savetxt(f'{base_dir}/gamma.txt', np.array([context.gamma]))

        self._version += 1
