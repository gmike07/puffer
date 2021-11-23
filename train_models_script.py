from argument_parser import parse_arguments
from config_creator import get_config
from tqdm import tqdm
import torch
from data_iterator import DataIterator
from models import AutoEncoder, SL_Model
import time
import numpy as np


def save_cpp_model(model, model_path, CONFIG, input_size):
    example = torch.rand(1, input_size).double()
    traced_script_module = torch.jit.trace(model, example, check_trace=False)
    traced_script_module.save(model_path)


def train_sl(model, loader):
    CONFIG = get_config()
    for epoch in range(CONFIG['epochs']):
        pbar = tqdm(iterable=iter(loader), ncols=200)
        for (chunks, metrics) in pbar:
            predictions = model(chunks)
            loss = model.loss_metrics(predictions, metrics)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            pbar.set_description_str('epoch #{}'.format(epoch))
            pbar.set_postfix(loss=loss.mean().item())
        pbar.close()
        filename = f"sl_weights_{str(epoch)}_abr_{CONFIG['abr']}_v{str(CONFIG['version'])}_{CONFIG['scoring_function_type']}.pt"
        torch.save({
            'model_state_dict': model.model.state_dict()
        }, f"{CONFIG['weights_path']}{filename}")
        save_cpp_model(model.model, f"{CONFIG['weights_cpp_path']}{filename}", CONFIG, CONFIG['nn_input_size'])


def train_ae(model, loader):
    CONFIG = get_config()
    for epoch in range(CONFIG['epochs']):
        pbar = tqdm(iterable=iter(loader), ncols=200)
        for (chunks, _) in pbar:
            predictions = model(chunks)
            loss = model.loss_metrics(predictions, chunks)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            pbar.set_description_str('epoch #{}'.format(epoch))
            pbar.set_postfix(loss=loss.mean().item())
        pbar.close()
        filename = f"ae_weights_{str(epoch)}_abr_{CONFIG['abr']}_v{str(CONFIG['version'])}_{CONFIG['scoring_function_type']}.pt"
        torch.save({
            'model_state_dict': model.encoder_model.state_dict()
        }, f"{CONFIG['ae_weights_path']}{filename}")
        save_cpp_model(model.encoder_model, f"{CONFIG['weights_cpp_path']}{filename}", CONFIG, CONFIG['nn_input_size'] - len(CONFIG['ccs']))


def train_rl(model, event, rl_type='rl'):
    CONFIG = get_config()
    total_measurements = [[] for _ in range(len(model.measurements))]
    rounds_to_save = CONFIG['rl_rounds_to_save'] 
    gradients = 0
    while not event.is_set():
        time.sleep(CONFIG['rl_sleep_sec'])
        if event.is_set():
            break
        for i, client_measures in enumerate(model.measurements):
            while not client_measures.empty() and client_measures.qsize() < CONFIG['rl_min_measuremets']:
                total_measurements[i].append(client_measures.get())
        
            rounds_to_save -= 1

            if len(total_measurements[i]) < CONFIG['rl_min_measuremets']:
                continue

            # select batch and update weights
            measures = np.array(total_measurements[i])
            indices = np.random.choice(np.arange(measures.size), CONFIG['rl_batch_size'])
            measures_batch = measures[indices]
            measures = np.delete(measures, indices)

            total_measurements[i] = list(measures)

            states = np.array(list(map(lambda s: s["state"], measures_batch)))
            rewards = np.array(list(map(lambda s: s["qoe"], measures_batch)))

            log_probs = []
            for state in states:
                log_prob = model.get_log_highest_probability(state)
                log_probs.append(log_prob)

            model.update_policy(rewards, log_probs)
            gradients += 1

            # save weights
            if rounds_to_save <= 0:
                print(f'saving {rl_type}...')
                filename = f"{rl_type}_weights_abr_{CONFIG['abr']}_v{str(CONFIG['version'])}_{CONFIG['scoring_function_type']}.pt"
                
                torch.save({
                    'model_state_dict': model.model.state_dict()
                }, f"{CONFIG[f'{rl_type}_weights_path']}{filename}")
                # save_cpp_model(model, f"{CONFIG['weights_cpp_path']}{filename}", CONFIG, model.input_size)

                total_measurements[i] = []
                model.clear_client_history(i)
                rounds_to_save = CONFIG['rl_rounds_to_save']
                with open(CONFIG['rl_logs_file'], 'w') as logs_file:
                    logs_file.write(f"Num of calculated gradients: {gradients}.")


if __name__ == '__main__':
    parse_arguments()
    if get_config()['model_name'] == 'ae':
        model = AutoEncoder(get_config()['all_models_config']['ae'])
        training_func = train_ae
    else:
        print('reached')
        model = SL_Model(get_config()['all_models_config']['sl'])   
        training_func = train_sl
    is_ae =  (get_config()['model_name'] == 'ae')
    iterator = DataIterator(remove_bad=False, output_type='ssim', remove_action=is_ae)
    print('training all files...')
    training_func(model, iterator)
    iterator = DataIterator(remove_bad=True, output_type='ssim', remove_action=is_ae)
    print('training good files...')
    training_func(model, iterator)
