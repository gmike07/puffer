from queue import Queue
from models.helper_functions import fill_default_key_conf, get_updated_config_model
from models.ae_model import AE_Model
from ..config_creator import get_config
import torch
import time

class AETrainer:
    def __init__(self, config, helper_model):
        self.clean_data = Queue()
        self.prediction_model = helper_model
        if not get_config()['test']:
            self.model = AE_Model(get_updated_config_model('ae', config))
        else:
            self.model = None
        self.sleep_time = fill_default_key_conf(config, 'ae_sleep_sec')
        self.rounds_to_sleep = fill_default_key_conf(config, 'ae_rounds_to_save')
        self.logs_file = fill_default_key_conf(config, 'ae_logs_file')
        self.training = not get_config()['test']

    def predict(self, state):
        return self.prediction_model.predict(state)

    def update(self, state):
        if not self.training:
            return
        torched_state = torch.from_numpy(state.reshape(1,-1))
        self.clean_data.put((torched_state, torched_state))

    def clear(self):
        pass

    def save(self):
        if self.model is not None:
            self.model.save()

    def load(self):
        self.prediction_model.load()
    
    def done(self):
        pass


def train_ae(model, event):
    CONFIG = get_config()
    rounds_to_save = model.rounds_to_sleep
    gradients = 0
    while not event.is_set():
        time.sleep(model.sleep_time)
        if event.is_set():
            break
        if model.clean_data.qsize() < CONFIG['batch_size']:
            continue
        inputs, outputs = [], []
        for _ in range(CONFIG['batch_size']):
            input, output = model.clean_data.get()
            inputs.append(input)
            outputs.append(output)
        inputs = torch.cat(inputs)
        outputs = torch.cat(outputs)

        predictions = model.model(inputs)
        loss = model.model.loss_metrics(predictions, outputs)
        model.model.optimizer.zero_grad()
        loss.backward()
        model.model.optimizer.step()

        rounds_to_save -= 1
        gradients += 1
        # save weights
        if rounds_to_save <= 0:
            print(f'saving ae...')
            model.model.save()
            rounds_to_save = model.rounds_to_sleep
            with open(model.logs_file, 'w') as logs_file:
                logs_file.write(f"Num of calculated gradients: {gradients}.")