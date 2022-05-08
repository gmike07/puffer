import torch
import yaml
import os
import copy
import pathlib


CONFIG = {}
BATCH_SIZE = -1

def requires_helper_model(model_name):
    return model_name in ['DNNTrainer', 'AETrainer', 'inputClusterTrainer', 'DNNClusterTrainer', 'AEClusterTrainer']

def is_threaded_model(model_name):
    return model_name in ['DRL', 'REINFORCE', 'REINFORCE_AE'] or requires_helper_model(model_name)


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def write_yaml_settings(experiments_dct):
    with open(CONFIG['yaml_path'] + 'default.yml', 'r') as f:
        default_dictionary = yaml.safe_load(f)
        default_dictionary.update(experiments_dct)
        if 'default_path' in CONFIG:
            default_dictionary['log_dir'] = f"{CONFIG['default_path']}/src/monitoring"
            default_dictionary['media_dir'] = f"{CONFIG['default_path']}/media-181230"
        with open(CONFIG['yml_output_dir'] + 'settings.yml', 'w') as outfile:
            yaml.dump(default_dictionary, outfile)
        with open(CONFIG['yml_output_dir'] + 'settings_offline.yml', 'w') as outfile:
            yaml.dump(default_dictionary, outfile)


def add_server_index_to_config(config):
    for i in range(len(config['experiments'])):
        config['experiments'][i]['fingerprint']['cc_config']['server_id'] = i
    return config



def create_setting_yaml(test=False):
    copy_fingerprint = copy.deepcopy(CONFIG['fingerprint'])
    copy_fingerprint['cc_config']['random_cc'] = False
    experiments_dct = {'experiments': [{'fingerprint': copy.deepcopy(copy_fingerprint), 
                                        'num_servers': 1} for _ in range(CONFIG['num_clients'])]
                    }


    for i in range(CONFIG['num_clients']):
        fingerprint = experiments_dct['experiments'][i]['fingerprint']
        model_name = CONFIG['model_name']
        if test:
            model_config = CONFIG['all_models_config'][CONFIG['models'][i]]
            fingerprint['cc_config']['model_name'] = model_config['model_name']
            model_name = model_config['model_name']
        
        if requires_helper_model(model_name):
            fingerprint['cc_config']['start_stateless'] = True
    
        fingerprint['cc'] = 'bbr'
        if 'constant' in model_name:
            fingerprint['cc'] = CONFIG['ccs'][int(model_name[len('constant_'):])]
        fingerprint['cc_config']['server_path'] = f"http://localhost:{CONFIG['server_port']}"
    write_yaml_settings(add_server_index_to_config(experiments_dct))



def update_epochs_conf(epochs, simulation_constants, test):
    if epochs != -1:
        CONFIG['mahimahi_epochs'] = epochs
    elif test:
        CONFIG['mahimahi_epochs'] = simulation_constants['epochs_testing']
    else:
        CONFIG['mahimahi_epochs'] = simulation_constants['epochs_training']


def update_constants_config():
    CONFIG['random_sample'] = CONFIG['random_sample_size']
    CONFIG['prediction_size'] = len(CONFIG['quality_cols']) - len(['file_index', 'chunk_index'])
    CONFIG['sample_size'] = len(set(CONFIG['cc_cols'] + CONFIG['ccs']) - set(CONFIG['deleted_cols']))
    CONFIG['nn_input_size'] = CONFIG['history_size'] * (CONFIG['random_sample'] * \
                    CONFIG['sample_size'] + CONFIG['prediction_size']) + len(CONFIG['ccs'])
    CONFIG['weights_decay'] = float(CONFIG['weights_decay'])
    CONFIG['lr'] = float(CONFIG['lr'])
    helper_split = CONFIG['betas'][1:-1].split(',')
    CONFIG['betas'] = (float(helper_split[0]), float(helper_split[1]))
    

def create_all_models_config(models_config_dct):
    mapping = {}
    for model_config in models_config_dct:
        mapping[model_config['model_name']] = model_config
    CONFIG['all_models_config'] = mapping


def update_clients(num_clients, simulation_constants, test, scoring_path):
    if test:
        create_dir(scoring_path[:scoring_path.rfind('/')])
    if num_clients == -1:
        CONFIG['num_clients'] = simulation_constants['clients']
    else:
        CONFIG['num_clients'] = num_clients


def update_scoring_config(scoring_path):
    path = CONFIG['cc_scoring_path']
    if scoring_path != '':
        path = scoring_path
    CONFIG.update({'scoring_path': path})


def update_key_not_empty(key, val):
    if val:
        CONFIG[key] = val

def create_config(yaml_input_path, abr='', num_clients=-1, test=False, eval=False, epochs=-1, model_name='', scoring_path=''):
    global BATCH_SIZE
    CONFIG.clear()
    with open(yaml_input_path + 'settings.yml', 'r') as f:
        yaml_dct = yaml.safe_load(f)
        headers = [header for header in yaml_dct if header != 'all_models']
        CONFIG.update(yaml_dct['fingerprint']['cc_config'])
        for dct_name in headers:
            CONFIG.update(yaml_dct[dct_name])
        update_constants_config()
        CONFIG.update({'test': test, 'yaml_path': yaml_input_path, 'eval': eval,
                        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        'fingerprint': yaml_dct['fingerprint']})
        
        update_epochs_conf(epochs, yaml_dct['simulation_constants'], test)
        update_key_not_empty('model_name', model_name)
        update_key_not_empty('abr', abr)
        
        CONFIG['settings'] = [(delay, loss) for delay in CONFIG['delays'] for loss in CONFIG['losses']]

        create_all_models_config(yaml_dct['all_models'])
        update_scoring_config(scoring_path)
        update_clients(num_clients, yaml_dct['simulation_constants'], test, CONFIG['scoring_path'])

        for path in [CONFIG['scoring_path'], CONFIG['exp3_model_path'], CONFIG['weights_path'], CONFIG['saving_cluster_path'], CONFIG['logs_path']]:
            create_dir(path)

        CONFIG['default_path'] = pathlib.Path().resolve()
        BATCH_SIZE = CONFIG['batch_size']
        


def get_config():
    return CONFIG

def get_batch_size():
    return BATCH_SIZE