import torch
import numpy as np
from queue import Queue
import time

from models.helper_functions import fill_default_key_conf, get_config


class RLTrainer(torch.nn.Module):
    def __init__(self, model):
        super(RLTrainer, self).__init__()
        self.model = model
        self.CONFIG = self.model.CONFIG
        self.model_name = self.model.model_name
        self.actions = self.model.actions
        self.num_clients = self.model.num_clients
        self.measurements = self.model.measurements
        self.gamma = self.model.gamma
        self.model_path = self.model.model_path
        self.optimizer = self.model.optimizer
        self.input_size = self.model.input_size
        self.config = model.config

        self.logs_file = fill_default_key_conf(self.config, 'logs_file')
        self.logs_path = fill_default_key_conf(self.config, 'logs_path')
        self.rounds_to_save = fill_default_key_conf(model.config, 'rounds_to_save')

        self.sleep_sec = fill_default_key_conf(model.config, 'sleep_sec')
        self.rl_min_measuremets = fill_default_key_conf(model.config, 'rl_min_measuremets')
        self.rl_batch_size = fill_default_key_conf(model.config, 'rl_batch_size')

        print('created RLTrainer')
    
    def forward(self, x):
        return self.model.forward(x)

    def load(self):
        dct = torch.load(self.model_path + self.model_name)
        self.model.load_state_dict(dct['model_state_dict'])
        print('loaded RLTrainer')


    def predict(self, sent_state):
        probabilities = self.forward(torch.from_numpy(sent_state['state']))
        return np.random.choice(self.actions, p=probabilities.detach().cpu().numpy().reshape(-1))

    def save(self, path=''):
        if path == '':
            path = self.model_name
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, f"{self.model_path}{path}")
    
    def update(self, state):
        if get_config()['test']:
            return
        self.measurements[state['server_id']].put(state)

    def clear(self):
        self.measurements = [Queue() for _ in range(self.num_clients)]
    
    def done(self):
        self.save()

    def clear_client_history(self, client_id):
        self.measurements[client_id] = Queue()

    def get_log_highest_probability(self, state):
        probs = self.forward(torch.from_numpy(state))
        highest_prob_action = np.random.choice(self.actions, p=probs.detach().cpu().numpy().reshape(-1))
        return torch.log(probs.squeeze(0)[highest_prob_action])

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []

        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards)

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()


def train_rl(model, event, rl_type='rl'):
    total_measurements = [[] for _ in range(len(model.measurements))]
    rounds_to_save = model.rounds_to_save
    gradients = 0
    while not event.is_set():
        time.sleep(model.sleep_sec)
        if event.is_set():
            break
        for i, client_measures in enumerate(model.measurements):
            while not client_measures.empty() and client_measures.qsize() < model.rl_min_measuremets:
                total_measurements[i].append(client_measures.get())
        
            rounds_to_save -= 1

            if len(total_measurements[i]) < model.rl_min_measuremets:
                continue

            # select batch and update weights
            measures = np.array(total_measurements[i])
            indices = np.random.choice(np.arange(measures.size), model.rl_batch_size)
            measures_batch = measures[indices]
            measures = np.delete(measures, indices)

            total_measurements[i] = list(measures)

            states = np.array(list(map(lambda s: s["state"], measures_batch)))
            rewards = np.array(list(map(lambda s: s["normalized qoe"], measures_batch)))

            log_probs = []
            for state in states:
                log_prob = model.get_log_highest_probability(state)
                log_probs.append(log_prob)
            model.update_policy(rewards, log_probs)
            gradients += 1

            # save weights
            if rounds_to_save <= 0:
                print(f'saving {rl_type}...')                
                model.save()
                total_measurements[i] = []
                model.clear_client_history(i)
                rounds_to_save = model.rounds_to_save
                with open(f'{model.logs_path}{model.logs_file}', 'w') as logs_file:
                    logs_file.write(f"Num of calculated gradients: {gradients}.")