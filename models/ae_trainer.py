from queue import Queue
from models.helper_functions import fill_default_key_conf, get_updated_config_model, get_config, get_batch_size
from models.ae_model import AutoEncoder
import torch
import time


class AETrainer:
    def __init__(self, config, helper_model):
        self.clean_data = Queue()
        self.prediction_model = helper_model
        if not get_config()['test']:
            self.model = AutoEncoder(get_updated_config_model('ae', config))
        else:
            self.model = None
        self.sleep_time = fill_default_key_conf(config, 'sleep_sec')
        self.rounds_to_sleep = fill_default_key_conf(config, 'rounds_to_save')
        self.logs_file = fill_default_key_conf(config, 'logs_file')
        self.logs_path = fill_default_key_conf(config, 'logs_path')
        self.training = not get_config()['test']
        print('created AETrainer')

    def predict(self, state):
        return self.prediction_model.predict(state)

    def update(self, state):
        if not self.training:
            return
        torched_state = torch.from_numpy(state['state'].reshape(1, -1))
        self.clean_data.put((torched_state, torched_state))

    def clear(self):
        pass

    def save(self):
        if self.model is not None:
            self.model.save()

    def load(self):
        self.prediction_model.load()
        print('loaded AETrainer')
    
    def done(self):
        self.save()

    def update_helper_model(self, helper_model):
        self.prediction_model = helper_model
        self.load()


def train_ae(model, event, type_trainer='ae', f=None):
    CONFIG = get_config()
    rounds_to_save = model.rounds_to_sleep
    gradients = 0
    CONFIG['batch_size'] = get_batch_size()
    while not event.is_set():
        if model.clean_data.qsize() < CONFIG['batch_size'] / 2:
            time.sleep(model.sleep_time)
        if event.is_set():
            break
        if model.clean_data.qsize() < CONFIG['batch_size']:
            continue
        inputs, outputs = [], []
        for _ in range(CONFIG['batch_size']):
            input_, output = model.clean_data.get()
            inputs.append(input_)
            outputs.append(output)
        inputs = torch.cat(inputs)
        outputs = torch.cat(outputs)

        predictions = model.model(inputs)
        loss = model.model.loss_metric(predictions, outputs)
        model.model.optimizer.zero_grad()
        loss.backward()
        model.model.optimizer.step()

        rounds_to_save -= 1
        gradients += 1
        # save weights
        if rounds_to_save <= 0:
            print(f'saving {type_trainer}...', model.clean_data.qsize(), len(inputs), loss.item())
            if f is not None:
                f.write(' '.join([f'saving {type_trainer}... ', str(model.clean_data.qsize()), str(len(inputs)), str(loss.item())]) + '\n')
                f.flush()

            model.model.save()
            rounds_to_save = model.rounds_to_sleep
            with open(f'{model.logs_path}{model.logs_file}', 'w') as logs_file:
                logs_file.write(f"Num of calculated gradients: {gradients}.")