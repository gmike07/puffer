#!/usr/bin/env python3
from concurrent.futures import thread
import time
import os
import subprocess
import signal
import numpy as np
import pandas as pd
import os
from requests.api import get
import requests
import json
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(grandparentdir)
from argument_parser import parse_arguments
from config_creator import get_config, create_setting_yaml, requires_helper_model, create_config
import signal
import sys
import pathlib
import random


plist = []
simulationDct = {}

def get_trace_path(trace_dir, file):
    return os.path.join(pathlib.Path().resolve(), trace_dir, file)

def get_delay_loss():
    delay, loss = get_config()['delays'], get_config()['losses']
    if get_config()['loss'] != -1.0:
        loss = get_config()['loss']
    if get_config()['delay'] != -1.0:
        delay = [get_config()['delay']]
    return delay, loss


def run_offline_media_servers():
    run_server_html_cmd = './helper_scripts/python_migrate.sh'
    p1 = subprocess.Popen(run_server_html_cmd, shell=True, preexec_fn=os.setsid)
    time.sleep(5)
    run_servers_cmd = './helper_scripts/python_run_servers.sh'
    p2 = subprocess.Popen(run_servers_cmd, shell=True, preexec_fn=os.setsid)
    time.sleep(5)
    return p1, p2


def clear_and_kill_all():
    kill_all_proccesses()
    subprocess.check_call("rm -rf ./*.profile", shell=True, executable='/bin/bash')


def signal_handler(sig, frame):
    clear_and_kill_all()
    sys.exit(0)


def get_mahimahi_command(trace_dir, filename, trace_index, delays, losses, seed=0, ms=5000):
    remote_port = get_config()['remote_base_port'] + trace_index
    port = get_config()['base_port'] + trace_index
    # format: delay1,...,delayn#seed-ms
    mahimahi_chrome_cmd = "mm-delay {}#{}-{} ".format(','.join(str(delay) for delay in delays), seed, ms)
    if losses:
        mahimahi_chrome_cmd += "mm-loss downlink {}#{}-{} ".format(','.join(str(loss) for loss in losses), seed, ms)
    mahimahi_chrome_cmd += "mm-link "
    # mahimahi_chrome_cmd += "--meter-uplink "
    # mahimahi_chrome_cmd += "--meter-downlink "

    mahimahi_chrome_cmd += f"{pathlib.Path().resolve()}/src/media-server/12mbps "
    mahimahi_chrome_cmd += get_trace_path(trace_dir, filename) + " "
    # mahimahi_chrome_cmd += "--downlink-log={} ".format('./uplink/uplink_1.up')
    mahimahi_chrome_cmd += "-- sh -c 'chromium-browser disable-infobars --disable-gpu --disable-software-rasterizer --headless --enable-logging=true --v=1 --remote-debugging-port={} http://$MAHIMAHI_BASE:8080/player/?wsport={} --user-data-dir=./{}.profile'".format(
                        remote_port, port, port)
    # print(mahimahi_chrome_cmd)
    helper_command = f'export PATH="{pathlib.Path().resolve()}/mahimahi/src/frontend:$PATH"'
    return helper_command + " && " + mahimahi_chrome_cmd


def send_dct_to_server(dct):
    try:
        requests.post(f"http://localhost:{get_config()['server_port']}", json.dumps(dct))
    except Exception as e:
        pass
    time.sleep(3)


def send_test_to_server():
    send_dct_to_server({'test': get_config()['test']})

def send_clear_to_server():
    send_dct_to_server({'clear': 'clear'})


def send_done_to_server():
    send_dct_to_server({'done': 'done'})


def send_switch_to_server(model_name, should_load, helper_model='', models=None):
    if models is None:
        models = []
    send_dct_to_server({'switch_model': 'switch_model', 'model_name': model_name, 'load': should_load, 'helper_model': helper_model, 'models': models})


def kill_proccesses(plist, sleep_time=3):
    for p in plist:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        time.sleep(sleep_time)


def kill_all_proccesses():
    global plist
    kill_proccesses(plist, 0)


def get_traces():
    trace_dir = get_config()['trace_dir'] + 'test/' if get_config()['test'] else get_config()['trace_dir'] + 'train/'
    traces = os.listdir(trace_dir)
    traces = list(sorted(traces))
    return trace_dir, traces


def run_single_simulation(num_clients, clients, f, seed, ms=5000):
    p1, p2 = run_offline_media_servers()
    plist = [p1, p2]
    sleep_time = 3
    delays, losses = get_delay_loss()

    traces, trace_dir = simulationDct['traces'], simulationDct['trace_dir']
    for i in range(1, clients + 1):
        index = f if get_config()['test'] else f + i - 1
        time.sleep(sleep_time)
        mahimahi_chrome_cmd = get_mahimahi_command(trace_dir, traces[index % len(traces)], i, delays, losses, seed, ms)
        p = subprocess.Popen(mahimahi_chrome_cmd, shell=True, preexec_fn=os.setsid)
        plist.append(p)

    time.sleep(60*11.5) # - sleep_time * clients

    kill_proccesses(plist[2:], sleep_time)
    kill_proccesses(plist[:2])

    send_clear_to_server()
    subprocess.check_call("rm -rf ./*.profile", shell=True,
                            executable='/bin/bash')
    with open(get_config()['logs_path'] + get_config()['train_test_log_file'], 'w') as f_log:
        f_log.write(f"{get_config()['model_name']}:\n")
        if 'helper_string' in simulationDct and simulationDct['helper_string']:
            f_log.write(simulationDct['helper_string'] + "\n")
        f_log.write(f"epoch: {simulationDct['epoch']} / {get_config()['mahimahi_epochs']}\n")
        f_log.write(f"trace: {f / num_clients} / {len(traces) // num_clients}")


def start_maimahi_clients(clients, exit_condition):
    global plist
    plist = []
    try:
        trace_dir, traces = get_traces()
        test_seperated = 'test_seperated' in get_config() and get_config()['test_seperated']
        num_clients = 1 if get_config()['test'] and not test_seperated else clients
        simulationDct['traces'] = traces
        simulationDct['trace_dir'] = trace_dir
        for epoch in range(get_config()['mahimahi_epochs']):
            simulationDct['epoch'] = epoch
            for f in range(0, len(traces), num_clients):
                index = (epoch * int(len(traces) / num_clients)) + int(f / num_clients)
                run_single_simulation(num_clients, clients, f, index)
                if exit_condition(index):
                    break
    except Exception as e:
        print("exception: " + str(e))
    finally:
        kill_proccesses(plist)
        subprocess.check_call("rm -rf ./*.profile", shell=True,
            executable='/bin/bash')


def create_arr(filedir, abr, i, ccs, func):
    arr = np.empty((len(ccs)))
    files = os.listdir(filedir)
    for j in range(len(ccs)):
        print(f'abr_{abr}_{i}_{ccs[j]}.txt')
        file = [f for f in files if f.find(f'abr_{abr}_{i}_{ccs[j]}.txt') != -1][0]
        lines = open(filedir + '/' + file, 'r').readlines()
        if len(lines) < 100:
            return None
        arr[j] = func(np.array([float(x) for x in lines]))
    return arr


def show_table(filedir, abr, max_iter, func, is_max=True, to_print=True):
    dfs = []
    ccs = get_config()['models']
    dct = {'constant_0': 0, 'constant_1': 1, 'constant_2': 2}
    ccs_named = list(map(lambda x: get_config()['ccs'][dct[x]] if x in dct else x, ccs))
    for i in range(max_iter):
        a = create_arr(filedir, abr, i, ccs, func)
        if a is not None:
            dct = {'exp': [i]}
            dct.update({ccs_named[i]: a[i] for i in range(len(ccs_named))})
            dfs.append(pd.DataFrame(dct))
    df = pd.concat(dfs)
    arr = df['exp']
    df = df.drop(['exp'], 1)
    delays = np.array([tup[0] for tup in get_config()['settings']])
    losses = np.array([tup[1] for tup in get_config()['settings']])
    df.insert(0, "loss", losses[(arr - 1) % len(losses)])
    df.insert(0, "delay", delays[(arr - 1) % len(delays)])
    if to_print and is_max:
        print(df)
        print(df.describe())
        dct = {'delay': delays, 'loss': losses}
        dct.update({cc: [np.mean(df[arr%16==i][cc]) for i in range(len(get_config()['settings']))] for cc in ccs_named})
        print(pd.DataFrame(dct))
    return df


def find_index(filename):
    filename = filename[filename.rfind(f"_{get_config()['abr']}_") + len(f"_{get_config()['abr']}_"):]
    filename = filename[:filename.find('_')]
    return int(filename)


def generate_table_models(filedir, models, threshold=np.inf, abr=''):
    if abr == '':
        abr == get_config()['abr']
    models_qoe = {}
    files = os.listdir(filedir)
    max_num = max(find_index(file) for file in files if file.endswith('.txt'))
    for model in models:
        models_qoe[model] = np.array([None] * max_num)
        for i in range(max_num):
            file = [f for f in files if f.find(f'abr_{abr}_{i}_{model}.txt') != -1][0]
            lines = open(filedir + '/' + file, 'r').readlines()
            if len(lines) < 100:
                continue
            qoe = np.array([float(x) for x in lines]).mean()
            if qoe < threshold:
                models_qoe[model][i] = qoe
    ccs = get_config()['ccs']
    mapping_conventions = {f'constant_{i}': ccs[i] for i in range(ccs)}
    mapping_conventions.update({'exp3KmeansCustom': 'exp3CustomContext'})

    f = lambda model: mapping_conventions[model] if model in mapping_conventions else model
    final_dct = {f(model): models_qoe[model] for model in models}
    return pd.DataFrame(final_dct)

     
def eval_scores2():
    print('mean')
    df = generate_table_models(get_config()['scoring_path'], get_config()['models'])
    df = df.dropna()
    print(df.mean())


def eval_scores_model2(models, filedir, threshold=np.inf):
    print('mean')
    df = generate_table_models(filedir, models, threshold=threshold)
    print(df.mean())


def eval_scores():
    files = os.listdir(get_config()['scoring_path'])
    print(get_config()['scoring_path'], files)
    max_num = max(find_index(file) for file in files if file.endswith('.txt'))
    print('mean')
    show_table(get_config()['scoring_path'], get_config()['abr'], max_num, np.mean)


def prepare_env(args=None):
    if get_config()['eval']:
        if args is not None:
            create_config(args.yaml_input_dir, args.abr, args.clients, args.test, args.eval, args.epochs, 'stackingModel', args.scoring_path)
            get_config()['models'] = list(map(lambda x: x.model_name, args.models))
        eval_scores()
        exit()
    subprocess.check_call('sudo sysctl -w net.ipv4.ip_forward=1', shell=True)
    subprocess.check_call("rm -rf ./*.profile", shell=True,
                                      executable='/bin/bash')


def run_simulation(model_name, should_load, f=lambda _: False, helper_model='', models=None):
    create_setting_yaml(test=get_config()['test'])
    send_test_to_server()
    send_clear_to_server()
    send_switch_to_server(model_name, should_load, helper_model, models)
    start_maimahi_clients(get_config()['num_clients'], f)
    print('finished part simulation!')


def train_simulation(model_name):
    if not requires_helper_model(model_name):
        # _, traces = get_traces()
        # exit_condition = lambda x: x % (len(traces) // get_config()['num_clients']) == (2 - 1)
        exit_condition = lambda x: False
        run_simulation(model_name, False, f=exit_condition)
        send_done_to_server()
        return
    epochs = get_config()['mahimahi_epochs']
    get_config()['mahimahi_epochs'] = 1
    for epoch in range(epochs):
        if model_name == 'inputClusterTrainer':
            exit_condition = lambda setting_number: setting_number == (4 - 1) # 4 iterations
        else:
            exit_condition = lambda _: False
        
        # exit_condition = lambda x: x == (2 - 1)

        simulationDct['helper_string'] = f'epoch: {epoch} / {epochs}, random: True'
        run_simulation(model_name, bool(epoch != 0), f=exit_condition, helper_model='random')
        if model_name == 'inputClusterTrainer':
            exit_condition = lambda setting_number: setting_number == (1 - 1) # 1 iterations
        else:
            exit_condition = lambda setting_number: setting_number == (3 - 1) # 3 iterations

        simulationDct['helper_string'] = f'epoch: {epoch} / {epochs}, random: False'
        # exit_condition = lambda x: x == (2 - 1)
        run_simulation(model_name, True, f=exit_condition, helper_model='idModel')
        simulationDct['helper_string'] = ''
    get_config()['mahimahi_epochs'] = epochs
    send_done_to_server()


def test_simulation():
    run_simulation('stackingModel', True, models=get_config()['models'])
    send_done_to_server()
    eval_scores()


def eval_scores_model(models, filedir, threshold=np.inf):
    files = os.listdir(get_config()['scoring_path'])
    dct = {}
    abr = get_config()['abr']
    for model in models:
        model_files = list(filter(lambda x: x.find(f'abr_{abr}_') and x.find(model) != -1 and x.find('.txt') != -1, files))
        for file in model_files:
            print(file)
            lines = open(filedir + '/' + file, 'r').readlines()
            if len(lines) < 100:
                continue
            if model not in dct:
                dct[model] = []
            qoe = np.array([float(x) for x in lines]).mean()
            if qoe < threshold:
                dct[model].append(qoe)
    dct = {key: np.array([np.mean(dct[key][setting]) for setting in range(len(get_config()['settings']))]) for key in dct}
    final_dct = {}
    final_dct['delay'] = np.array([delay for (delay, loss) in get_config()['settings']])[np.arange(len(get_config()['settings']))]
    final_dct['loss'] = np.array([loss for (delay, loss) in get_config()['settings']])[np.arange(len(get_config()['settings']))]
    final_dct.update(dct)
    conversion = {'constant_0': 'bbr', 'constant_1': 'vegas', 'constant_2': 'cubic', 'exp3KmeansCustom': 'exp3CustomContext'}
    f = lambda x: x if x not in conversion else conversion[x]
    final_dct = {f(key): final_dct[key] for key in final_dct}
    print('mean')
    print(pd.DataFrame(final_dct).mean())
    for per in [0.05, 0.1, 0.25]:
        print(f'{per}% precentile')
        print(pd.DataFrame(final_dct).quantile(per))


def test_simulation_model(models):
    get_config()['test_seperated'] = True
    for model in models:
        get_config()['models'] = [model] * get_config()['num_clients']
        run_simulation('stackingModel', True, models=get_config()['models'])
        send_done_to_server()
    eval_scores_model(models, get_config()['scoring_path'][:get_config()['scoring_path'].rfind('/') + 1])


if __name__ == '__main__':
    parse_arguments()
    signal.signal(signal.SIGINT, signal_handler)
    prepare_env()
    if get_config()['test']:
        test_simulation()
    else:
        train_simulation(get_config()['model_name'])