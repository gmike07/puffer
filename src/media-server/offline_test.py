#!/usr/bin/env python3
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


plist = []

def get_delay_loss(index):
    delay, loss = get_config()['settings'][index % len(get_config()['settings'])]
    if get_config()['loss'] != -1.0:
        loss = get_config()['loss']
    if get_config()['delay'] != -1.0:
        delay = get_config()['delay']
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


def get_mahimahi_command(trace_dir, filename, trace_index, delay, loss):
    remote_port = get_config()['remote_base_port'] + trace_index
    port = get_config()['base_port'] + trace_index
    mahimahi_chrome_cmd = "mm-delay {} ".format(int(delay))
    if loss != 0:
        mahimahi_chrome_cmd += "mm-loss uplink {} ".format(loss)
    mahimahi_chrome_cmd += "mm-link "
    # mahimahi_chrome_cmd += "--meter-downlink "

    mahimahi_chrome_cmd += f"{pathlib.Path().resolve()}/src/media-server/12mbps "
    # mahimahi_chrome_cmd += "/home/mike/puffer/src/media-server/12mbps "
    mahimahi_chrome_cmd += "{}/{} ".format(trace_dir, filename)
    # mahimahi_chrome_cmd += "--downlink-log={} ".format('./uplink/uplink_{}.up'.format(str(f+i)))
    mahimahi_chrome_cmd += "-- sh -c 'chromium-browser disable-infobars --disable-gpu --headless --enable-logging=true --v=1 --remote-debugging-port={} http://$MAHIMAHI_BASE:8080/player/?wsport={} --user-data-dir=./{}.profile'".format(
                        remote_port, port, port)
    # print(mahimahi_chrome_cmd)
    return mahimahi_chrome_cmd


def send_dct_to_server(dct):
    try:
        requests.post(f"http://localhost:{get_config()['server_port']}", json.dumps(dct))
    except Exception as e:
        pass
    time.sleep(3)


def send_clear_to_server():
    send_dct_to_server({'clear': 'clear'})


def send_done_to_server():
    send_dct_to_server({'done': 'done'})


def send_switch_to_server(model_name, should_load, helper_model='', models=None):
    if models is None:
        models = []
    send_dct_to_server({'switch_model': 'switch_model', 'model_name': model_name, 'load': should_load, 'helper_model': helper_model, 'models': models})


def create_failed_files(filedir, setting):
    if not get_config()['test']:
        return
    arr = get_config()['test_models']
    for i in range(len(arr)):
        filepath = f"{filedir}cc_score_{i+1}_abr_{get_config()['abr']}_{arr[i]}_{setting}.txt"
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                pass


def kill_proccesses(plist, sleep_time=3):
    for p in plist:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        time.sleep(sleep_time)


def kill_all_proccesses():
    global plist
    kill_proccesses(plist, 0)


def start_maimahi_clients(clients, filedir, exit_condition):
    global plist
    plist = []
    try:
        trace_dir = get_config()['trace_dir'] + 'test/' if get_config()['test'] else get_config()['trace_dir'] + 'train/'
        traces = os.listdir(trace_dir)
        traces = list(sorted(traces))
        
        test_seperated = 'test_seperated' in get_config() and get_config()['test_seperated']

        for epoch in range(get_config()['mahimahi_epochs']):
            num_clients = 1 if get_config()['test'] and not test_seperated else clients
            for f in range(0, len(traces), num_clients):

                p1, p2 = run_offline_media_servers()
                plist = [p1, p2]
                sleep_time = 3
                setting = (epoch * int(len(traces) / num_clients)) + int(f / num_clients)
                delay, loss = get_delay_loss(setting)


                for i in range(1, clients + 1):
                    index = f if get_config()['test'] else f + i - 1
                    time.sleep(sleep_time)
                    mahimahi_chrome_cmd = get_mahimahi_command(trace_dir, traces[index % len(traces)], i, delay, loss)
                    p = subprocess.Popen(mahimahi_chrome_cmd, shell=True,
                                         preexec_fn=os.setsid)
                    plist.append(p)

                time.sleep(60*10) # - sleep_time * clients

                kill_proccesses(plist[2:], sleep_time)
                kill_proccesses(plist[:2])
                
                # create_failed_files(filedir, setting)
                send_clear_to_server()
                subprocess.check_call("rm -rf ./*.profile", shell=True,
                                      executable='/bin/bash')
                with open(get_config()['logs_path'] + get_config()['train_test_log_file'], 'w') as f_log:
                    f_log.write(f"{get_config()['model_name']}:\n")
                    f_log.write(f"epoch: {epoch} / {get_config()['mahimahi_epochs']}\n")
                    f_log.write(f"trace: {f / num_clients} / {len(traces) // num_clients}")
                if exit_condition(setting):
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
    df = df[df['bbr'] < 17]
    if to_print and is_max:
        print(df)
        print(df.describe())
    return df


def find_index(filename):
    filename = filename[filename.rfind(f"_{get_config()['abr']}_") + len(f"_{get_config()['abr']}_"):]
    filename = filename[:filename.find('_')]
    return int(filename)


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
            get_config()['models'] = [model[0] for model in args.models]
            # get_config()['models'] = ['constant_0', 'constant_1', 'constant_2', 'exp3KmeansCustom', 'random', 'idModel']
        eval_scores()
        exit()
    subprocess.check_call('sudo sysctl -w net.ipv4.ip_forward=1', shell=True)
    subprocess.check_call("rm -rf ./*.profile", shell=True,
                                      executable='/bin/bash')


def run_simulation(model_name, should_load, f=lambda _: False, helper_model='', models=None):
    create_setting_yaml(test=get_config()['test'])
    send_clear_to_server()
    send_switch_to_server(model_name, should_load, helper_model, models)
    scoring_dir = get_config()['scoring_path'][:get_config()['scoring_path'].rfind('/') + 1]
    start_maimahi_clients(get_config()['num_clients'], scoring_dir, f)
    print('finished part simulation!')


def train_simulation(model_name):
    if not requires_helper_model(model_name):
        run_simulation(model_name, False)
        send_done_to_server()
        return
    epochs = get_config()['mahimahi_epochs']
    get_config()['mahimahi_epochs'] = 1
    for epoch in range(epochs):
        run_simulation(model_name, bool(epoch != 0), helper_model='random')
        exit_condition = lambda setting_number: setting_number == (3 - 1) # 3 iterations
        run_simulation(model_name, True, f=exit_condition, helper_model='idModel')
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
                dct[model] = [[] for _ in range(len(get_config()['settings']))]
            qoe = np.array([float(x) for x in lines]).mean()
            setting = find_index(file)
            if qoe <  threshold:
                dct[model][setting % len(get_config()['settings'])].append(qoe)
    dct = {key: np.array([np.mean(dct[key][setting]) for setting in range(len(get_config()['settings']))]) for key in dct}
    final_dct = {}
    final_dct['delay'] = np.array([delay for (delay, loss) in get_config()['settings']])[np.arange(len(get_config()['settings']))]
    final_dct['loss'] = np.array([loss for (delay, loss) in get_config()['settings']])[np.arange(len(get_config()['settings']))]
    final_dct.update(dct)
    conversion = {'constant_0': 'bbr', 'constant_1': 'vegas', 'constant_2': 'cubic', 'exp3KmeansCustom': 'exp3CustomContext'}
    f = lambda x: x if x not in conversion else conversion[x]
    final_dct = {f(key): final_dct[key] for key in final_dct}
    print(pd.DataFrame(final_dct))


def test_simulation_model(models):
    get_config()['test_seperated'] = True
    for model in models:
        get_config()['models'] = [model] * get_config()['num_clients']
        run_simulation('stackingModel', True, models=[model] * get_config()['num_clients'])
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