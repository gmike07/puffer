from numpy.lib.npyio import save
from .context import Context
import numpy as np
import os
import shutil


class Exp3KMeans:
    def __init__(self, num_of_arms, kmeans, save_path, checkpoint=10):
        self._kmeans = kmeans
        self._version = 0
        self._contexts = []
        self._num_of_arms = num_of_arms
        self._checkpoint = checkpoint
        self._time2save = checkpoint
        self.save_path = save_path

    def predict(self, datapoint):
        context_idx = self._kmeans.predict(datapoint)
        context_idx = context_idx[0]
        arm = self._contexts[context_idx].predict()
        return arm

    def update(self, datapoint, last_arm, reward, version):
        if version < self._version:
            return False
        context_idx = self._kmeans.predict(datapoint)
        context_idx = context_idx[0]
        self._contexts[context_idx].update(reward, last_arm)

        self._time2save -= 1
        if self._time2save <= 0:
            self._time2save = self._checkpoint
            self.save()
        return True

    def save(self):
        old_dir = f'{self.save_path}/{self._version - 1}'
        if os.path.isdir(old_dir):
            shutil.rmtree(old_dir)

        if os.path.isdir(f'{self.save_path}/{self._version}'):
            shutil.rmtree(f'{self.save_path}/{self._version}')

        os.mkdir(f'{self.save_path}/{self._version}')
        for i, context in enumerate(self._contexts):
            base_dir = f'{self.save_path}/{self._version}/{i}'
            os.mkdir(base_dir)
            np.savetxt(f'{base_dir}/cluster.txt', context.cluster)
            np.savetxt(f'{base_dir}/weights.txt', context.weights)
            np.savetxt(f'{base_dir}/gamma.txt', np.array([context.gamma]))

        self._version += 1

    def load(self):
        if len(os.listdir(self.save_path)) == 0:
            for cluster in self._kmeans.cluster_centers_:
                context = Context(self._num_of_arms, cluster)
                self._contexts.append(context)
            self.save()
        else:
            base_exp3_dir = f'{self.save_path}/{os.listdir(self.save_path)[-1]}'
            self._version = int(os.listdir(self.save_path)[-1]) + 1
            for dir in os.listdir(base_exp3_dir):
                base_context_dir = f'{base_exp3_dir}/{dir}'
                cluster = np.loadtxt(f'{base_context_dir}/cluster.txt')
                weights = np.loadtxt(f'{base_context_dir}/weights.txt')
                gamma = np.loadtxt(f'{base_context_dir}/gamma.txt')
                self._contexts.append(
                    Context(self._num_of_arms, cluster, gamma, weights))