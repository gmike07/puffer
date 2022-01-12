from config_creator import get_config as helper_get_config
from config_creator import get_batch_size as helper_get_batch_size
import copy
import torch
import numpy as np


def fill_default(variable, default):
    if variable:
        return variable
    else:
        return default


def fill_default_key(dct, key, default):
    if key in dct and dct[key]:
            return dct[key]
    return default

def fill_default_key_conf(dct, key):
    return fill_default_key(dct, key, get_config()[key])


def get_updated_config_model(model_name, config):
    conf = copy.deepcopy(get_config()['all_models_config'][model_name])
    conf.update(config)
    return conf


def create_actions():
    ccs = get_config()['ccs']
    batch_size = get_config()['batch_size']
    return [torch.from_numpy(np.vstack([np.arange(len(ccs)) == i for _ in range(batch_size)]).astype(np.int)) for i in range(len(ccs))]


def merge_state_actions(state, action):
    return torch.from_numpy(np.hstack([state, action]))

def get_config():
    return helper_get_config()

def get_batch_size():
    return helper_get_batch_size()