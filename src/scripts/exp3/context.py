import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

class Context:
    MAX_HISTORY_POINTS = 500

    def __init__(self, num_of_arms, cluster, gamma=None, weights=None):
        self.num_of_arms = num_of_arms
        self.cluster = cluster
        if gamma is None:
            self.gamma = min(1.0, np.sqrt(np.log(num_of_arms) / num_of_arms))
            self.t = 1
        else:
            self.gamma = gamma
            self.t = int(np.log(num_of_arms) /
                         (num_of_arms * np.square(gamma)))

        if weights is None:
            self.weights = np.zeros(num_of_arms)
        else:
            self.weights = np.array(weights)
        self.last_arm = 0
        self.weights_history = [np.copy(self.weights)]
        self.weights_history_scale = 1

    def predict(self):
        if self.gamma != 1.0:
            c = logsumexp(self.gamma * self.weights)
            prob_dist = np.exp((self.gamma * self.weights) - c)
        else:
            prob_dist = np.ones(self.num_of_arms) / np.float(self.num_of_arms)
        arm = np.random.choice(a=self.num_of_arms, p=prob_dist)
        self.last_arm = arm
        return arm

    def update(self, reward, last_arm=None):
        if last_arm == None:
            last_arm = self.last_arm
        loss = 1 - reward
        if self.gamma != 1.0:
            c = logsumexp(self.gamma * self.weights)
            prob = np.exp((self.gamma * self.weights[last_arm]) - c)
        else:
            prob = 1.0 / np.float(self.num_of_arms)
        prob=1
        assert prob > 0, "Assertion error"

        estimated = loss / prob
        print(f'weights changed {estimated}')
        self.weights[last_arm] += estimated
        self.t += 1
        self.gamma = min(1.0, np.sqrt(
            np.log(self.num_of_arms) / (self.num_of_arms * self.t)))
        self.weights_history.append(self.weights)

    def add_history(self, weights):
        self.weights_history.append(weights)
        if len(self.weights_history) > Context.MAX_HISTORY_POINTS:
            self.weights_history = self.weights_history[::2]
            self.weights_history_scale *= 2


    def plot_weights(self, save_dir):
        fig, ax = plt.subplots()
        for i in range(self.num_of_arms):
            arm_i_weights = np.array(list(map(lambda x: x[i], 100*self.weights_history)))
            scale = self.weights_history_scale * np.arange(len(arm_i_weights))
            ax.plot(arm_i_weights, scale)
        fig.savefig(save_dir)
        
