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



def create_setting_yaml(generating_data='', test=False):
    copy_fingerprint = copy.deepcopy(CONFIG['fingerprint'])
    if generating_data == 'randomized':
        copy_fingerprint['cc_config']['cc_monitoring_path'] = CONFIG['cc_monitoring_path']
        copy_fingerprint['cc_config']['random_cc'] = True
        experiments_dct = {'experiments': [{'fingerprint': copy_fingerprint, 
                                            'num_servers': CONFIG['num_clients']}]
                        }
        write_yaml_settings(experiments_dct)
        return
    if generating_data == 'fixed':
        copy_fingerprint['cc_config']['cc_monitoring_path'] = CONFIG['cc_monitoring_path']
        copy_fingerprint['cc_config']['random_cc'] = False
        experiments_dct = {'experiments': [{'fingerprint': copy.deepcopy(copy_fingerprint), 
                                            'num_servers': 1} for _ in range(CONFIG['num_clients'])]
                        }
        for i, cc in enumerate(CONFIG['ccs']):
            experiments_dct['experiments'][i]['fingerprint']['cc'] = cc
        write_yaml_settings(experiments_dct)
        return
    
    fixed_ccs = CONFIG['ccs'] if test else []
    if test:
        copy_fingerprint['cc_config']['cc_scoring_path'] = CONFIG['scoring_path']
    
    copy_fingerprint['cc_config']['random_cc'] = False
    experiments_dct = {'experiments': [{'fingerprint': copy.deepcopy(copy_fingerprint), 
                                        'num_servers': 1} for _ in range(CONFIG['num_clients'])]
                    }
    for i, cc in enumerate(fixed_ccs):
        experiments_dct['experiments'][i]['fingerprint']['cc'] = cc
    for i in range(len(fixed_ccs), CONFIG['num_clients']):
        experiments_dct['experiments'][i]['fingerprint']['cc'] = 'bbr'
        experiments_dct['experiments'][i]['fingerprint']['cc_config']['server_path'] = f"http://localhost:{CONFIG['server_port']}"
    write_yaml_settings(experiments_dct)


def create_config(yaml_input_path, abr='', num_clients=5, test=False, eval=False, generate_data=False):
    with open(yaml_input_path + 'settings.yml', 'r') as f:
        yaml_dct = yaml.safe_load(f)
        for dct_name in ['servers', 'contexts', 'clusters', 'exp3', 'nn_model', 'fingerprint', 'paths', 'constants', 'simulation_constants']:
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
                        'fingerprint': yaml_dct['fingerprint']})
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

        if CONFIG['test']:
            create_dir(scoring_path)

        if CONFIG['test']:
            CONFIG['num_clients'] = len(CONFIG['ccs']) + CONFIG['num_models']
        
        CONFIG['settings'] = [(delay, loss) for delay in CONFIG['delays'] for loss in CONFIG['losses']]

        CONFIG['weights_decay'] = float(CONFIG['weights_decay'])
        CONFIG['lr'] = float(CONFIG['lr'])
        helper_split = CONFIG['betas'][1:-1].split(',')
        CONFIG['betas'] = (float(helper_split[0]), float(helper_split[1]))


def get_config():
    return CONFIG