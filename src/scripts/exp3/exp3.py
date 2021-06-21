from .context import Context


class Exp3KMeans:
    def __init__(self, num_of_arms, kmeans):
        self._kmeans = kmeans
        
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
        self._contexts[context_idx].update(reward, last_arm)
