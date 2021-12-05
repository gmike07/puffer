import torch
import yaml
import os
import copy


CONFIG = {}


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def write_yaml_settings(experiments_dct):
    with open(CONFIG['yaml_path'] + 'default.yml', 'r') as f:
        default_dictionary = yaml.safe_load(f)
        default_dictionary.update(experiments_dct)
        with open(CONFIG['yml_output_dir'] + 'settings.yml', 'w') as outfile:
            yaml.dump(default_dictionary, outfile)
        with open(CONFIG['yml_output_dir'] + 'settings_offline.yml', 'w') as outfile:
            yaml.dump(default_dictionary, outfile)

def add_server_index_to_config(config):
    for i in range(len(config['experiments'])):
        config['experiments'][i]['fingerprint']['cc_config']['server_id'] = i
    return config



def create_setting_yaml(generating_data='', test=False):
    copy_fingerprint = copy.deepcopy(CONFIG['fingerprint'])
    if generating_data == 'randomized':
        copy_fingerprint['cc_config']['cc_monitoring_path'] = CONFIG['cc_monitoring_path']
        copy_fingerprint['cc_config']['random_cc'] = True
        experiments_dct = {'experiments': [{'fingerprint': copy_fingerprint, 
                                            'num_servers': CONFIG['num_clients']}]
                        }
        write_yaml_settings(add_server_index_to_config(experiments_dct))
        return
    if generating_data == 'fixed':
        copy_fingerprint['cc_config']['cc_monitoring_path'] = CONFIG['cc_monitoring_path']
        copy_fingerprint['cc_config']['random_cc'] = False
        experiments_dct = {'experiments': [{'fingerprint': copy.deepcopy(copy_fingerprint), 
                                            'num_servers': 1} for _ in range(CONFIG['num_clients'])]
                        }
        for i in range(CONFIG['num_clients']):
            experiments_dct['experiments'][i]['fingerprint']['cc'] = CONFIG['ccs'][i % len(CONFIG['ccs'])]
        write_yaml_settings(add_server_index_to_config(experiments_dct))
        return
    

    if test:
        copy_fingerprint['cc_config']['cc_scoring_path'] = CONFIG['scoring_path']

    copy_fingerprint['cc_config']['random_cc'] = False
    experiments_dct = {'experiments': [{'fingerprint': copy.deepcopy(copy_fingerprint), 
                                        'num_servers': 1} for _ in range(CONFIG['num_clients'])]
                    }


    for i in range(CONFIG['num_clients']):
        fingerprint = experiments_dct['experiments'][i]['fingerprint']
        if test:
            model_config = CONFIG['all_models_config'][CONFIG['test_models'][i]]
            fingerprint['cc_config']['cc_scoring_path'] = CONFIG['scoring_path']
            fingerprint['cc_config']['model_name'] = model_config['model_name']
        fingerprint['cc'] = 'bbr'
        fingerprint['cc_config']['server_path'] = f"http://localhost:{CONFIG['server_port']}"
    write_yaml_settings(add_server_index_to_config(experiments_dct))


def create_config(yaml_input_path, abr='', num_clients=-1, test=False, eval=False, generate_data=False, contextless=False, model_name=''):
    with open(yaml_input_path + 'settings.yml', 'r') as f:
        yaml_dct = yaml.safe_load(f)
        for dct_name in ['servers', 'contexts', 'clusters', 'exp3', 'nn_model', 'fingerprint', 'paths', 'constants', 'simulation_constants', 
                            'ae_model', 'rl_model', 'srl_model', 'rl_training_settings']:
            CONFIG.update(yaml_dct[dct_name])
        CONFIG.update(yaml_dct['fingerprint']['cc_config'])
        CONFIG['random_sample'] = CONFIG['random_sample_size']
        CONFIG['weights_cpp_path'] = CONFIG['cpp_weights_path']
        CONFIG['prediction_size'] = len(CONFIG['quality_cols']) - len(['file_index', 'chunk_index'])
        CONFIG['sample_size'] = len(set(CONFIG['cc_cols'] + CONFIG['ccs']) - set(CONFIG['deleted_cols']))
        CONFIG['nn_input_size'] = CONFIG['history_size'] * (CONFIG['random_sample'] * \
                        CONFIG['sample_size'] + CONFIG['prediction_size']) + len(CONFIG['ccs'])
        CONFIG.update({'training': not test, 'test': test, 'num_clients': num_clients, 
                        'yaml_path': yaml_input_path, 'eval': eval, 'generate_data': generate_data,
                        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        'fingerprint': yaml_dct['fingerprint'], 'cluster_contextless': contextless})
        
        if num_clients == -1:
            CONFIG['num_clients'] = yaml_dct['simulation_constants']['clients']

        if generate_data:
            CONFIG['mahimahi_epochs'] = yaml_dct['simulation_constants']['epochs_generating_data']
        elif test:
            CONFIG['mahimahi_epochs'] = yaml_dct['simulation_constants']['epochs_testing']
        else:
            CONFIG['mahimahi_epochs'] = yaml_dct['simulation_constants']['epochs_training']

        if model_name:
            CONFIG['model_name'] = model_name
        

        if abr != '':
            CONFIG.update({'abr': abr})
        if generate_data:
            CONFIG['test'] = False

        path = CONFIG['cc_monitoring_path']
        CONFIG.update({'input_dir': path[:path.rfind('/') + 1]})

        path = CONFIG['cc_scoring_path']
        index = path.rfind('/')
        scoring_path = path[:index] + '/' + CONFIG['abr'] + '_' + CONFIG['model_name'] + path[index:]
        CONFIG.update({'scoring_path': scoring_path})
        
        CONFIG['settings'] = [(delay, loss) for delay in CONFIG['delays'] for loss in CONFIG['losses']]

        CONFIG['weights_decay'] = float(CONFIG['weights_decay'])
        CONFIG['lr'] = float(CONFIG['lr'])
        helper_split = CONFIG['betas'][1:-1].split(',')
        CONFIG['betas'] = (float(helper_split[0]), float(helper_split[1]))

        mapping = {}
        for model_config in yaml_dct['all_models']:
            mapping[model_config['model_name']] = model_config
        CONFIG['all_models_config'] = mapping


        if CONFIG['test']:
            create_dir(scoring_path[:scoring_path.rfind('/')])
            CONFIG['num_clients'] = len(CONFIG['test_models'])
        


def get_config():
    return CONFIG