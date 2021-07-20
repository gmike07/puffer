import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt


class Context:
    MAX_HISTORY_POINTS = 100
    MAX_WEIGHT = 1e6

    def __init__(self, num_of_arms, cluster, gamma=None, weights=None, lr=None):
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
            self.weights = np.ones(num_of_arms)
        else:
            self.weights = np.array(weights)
        self.lr = lr
        self.last_arm = 0
        self.weights_history = [np.copy(self.weights)]
        self.weights_history_scale = 1

    def predict(self):
        c = logsumexp(self.weights)
        prob_dist = np.exp(self.weights - c)
        prob_dist = (1 - self.lr) * prob_dist + \
            self.lr * (1 / self.num_of_arms)
        arm = np.random.choice(a=self.num_of_arms, p=prob_dist)
        self.last_arm = arm
        return arm

    def update(self, reward, last_arm=None):
        if last_arm == None:
            last_arm = self.last_arm

        c = logsumexp(self.weights)
        prob = np.exp(self.weights[last_arm] - c)
        prob = (1 - self.lr) * prob + \
            self.lr * (1 / self.num_of_arms)

        loss = reward / prob
        # print(np.exp(self.lr * loss / self.num_of_arms))
        self.weights[last_arm] *= np.exp(self.lr *
                                         loss / self.num_of_arms)

        # normalize weights
        if self.weights.max() > Context.MAX_WEIGHT:
            self.weights /= Context.MAX_WEIGHT

        # self.t += 1

        self.save_history()

        print(
            f', weight={self.weights[last_arm]}')

    def save_history(self):
        self.weights_history.append(self.weights)
        if len(self.weights_history) > Context.MAX_HISTORY_POINTS:
            self.weights_history = self.weights_history[::2]
            self.weights_history_scale *= 2

    def plot_weights(self, save_dir):
        fig, ax = plt.subplots()
        all_weights = np.stack(self.weights_history, axis=1)
        for i in range(self.num_of_arms):
            arm_i_weights = all_weights[i, :]
            scale = self.weights_history_scale * np.arange(len(arm_i_weights))
            ax.plot(scale, arm_i_weights)
        ax.set_ylabel('weights scale')
        ax.set_xlabel('iteration number')
        fig.savefig(save_dir)
        plt.close('all')
