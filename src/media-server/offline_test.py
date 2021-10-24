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
from config_creator import get_config, create_setting_yaml


CONFIG = {}


def get_delay_loss(index):
    delay, loss = CONFIG['settings'][index % len(CONFIG['settings'])]
    if CONFIG['loss'] != -1.0:
        loss = CONFIG['loss']
    if CONFIG['delay'] != -1.0:
        delay = CONFIG['delay']
    return delay, loss


def run_offline_media_servers():
    run_server_html_cmd = './helper_scripts/python_migrate.sh'
    p1 = subprocess.Popen(run_server_html_cmd, shell=True, preexec_fn=os.setsid)
    time.sleep(5)
    run_servers_cmd = './helper_scripts/python_run_servers.sh'
    p2 = subprocess.Popen(run_servers_cmd, shell=True, preexec_fn=os.setsid)
    time.sleep(5)
    return p1, p2


def get_mahimahi_command(trace_dir, filename, trace_index, delay, loss):
    remote_port = CONFIG['remote_base_port'] + trace_index
    port = CONFIG['base_port'] + trace_index
    mahimahi_chrome_cmd = "mm-delay {} ".format(int(delay))
    if loss != 0:
        mahimahi_chrome_cmd += "mm-loss uplink {} ".format(loss)
    mahimahi_chrome_cmd += "mm-link "
    # mahimahi_chrome_cmd += "--meter-downlink "
    mahimahi_chrome_cmd += "/home/mike/puffer/src/media-server/12mbps "
    mahimahi_chrome_cmd += "{}/{} ".format(trace_dir, filename)
    # mahimahi_chrome_cmd += "--downlink-log={} ".format('./uplink/uplink_{}.up'.format(str(f+i)))
    mahimahi_chrome_cmd += "-- sh -c 'chromium-browser disable-infobars --disable-gpu --headless --enable-logging=true --v=1 --remote-debugging-port={} http://$MAHIMAHI_BASE:8080/player/?wsport={} --user-data-dir=./{}.profile'".format(
                        remote_port, port, port)
    return mahimahi_chrome_cmd


def send_clear_to_server():
    if CONFIG['generate_data']:
        return
    try:
        requests.post(f"http://localhost:{CONFIG['server_port']}", json.dumps({'clear': 'clear'}))
    except Exception as e:
        pass
    time.sleep(3)


def create_failed_files(filedir, setting):
    if not CONFIG['test']:
        return
    arr = get_config()['test_models']
    for i in range(len(arr)):
        filepath = f"{filedir}cc_score_{i+1}_abr_{CONFIG['abr']}_{arr[i]}_{setting}.txt"
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                pass


def kill_proccesses(plist, sleep_time=3):
    for p in plist:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        time.sleep(sleep_time)


def start_maimahi_clients(clients, filedir, exit_condition):
    plist = []
    try:
        trace_dir = CONFIG['trace_dir'] + 'test/' if CONFIG['test'] else CONFIG['trace_dir'] + 'train/'
        traces = os.listdir(trace_dir)
        traces = list(sorted(traces))
        
        for epoch in range(CONFIG['mahimahi_epochs']):
            num_clients = 1 if CONFIG['test'] else clients
            for f in range(0, len(traces), num_clients):

                p1, p2 = run_offline_media_servers()
                plist = [p1, p2]
                sleep_time = 3
                setting = (epoch * int(len(traces) / num_clients)) + int(f / num_clients)
                delay, loss = get_delay_loss(setting)


                for i in range(1, clients + 1):
                    index = f if CONFIG['test'] else f + i - 1
                    time.sleep(sleep_time)
                    mahimahi_chrome_cmd = get_mahimahi_command(trace_dir, traces[index], i, delay, loss)
                    p = subprocess.Popen(mahimahi_chrome_cmd, shell=True,
                                         preexec_fn=os.setsid)
                    plist.append(p)

                time.sleep(60*10 - sleep_time * clients)

                kill_proccesses(plist[2:], sleep_time)
                kill_proccesses(plist[:2])
                
                create_failed_files(filedir, setting)
                send_clear_to_server()
                subprocess.check_call("rm -rf ./*.profile", shell=True,
                                      executable='/bin/bash')
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
        file = [f for f in files if f.find(f'abr_{abr}_{ccs[j]}_{i}.txt') != -1][0]
        lines = open(filedir + '/' + file, 'r').readlines()
        if len(lines) < 100:
            return None
        arr[j] = func(np.array([float(x) for x in lines]))
    return arr


def show_table(filedir, abr, max_iter, func, is_max=True, to_print=True):
    dfs = []
    ccs = get_config()['test_models']
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
    # if is_max:
    #     df['max-nn'] = np.max(df, axis=1) - df['nn']
    # else:
    #     df['min-nn'] = df['nn'] - np.min(df, axis=1)
    delays = np.array([tup[0] for tup in CONFIG['settings']])
    losses = np.array([tup[1] for tup in CONFIG['settings']])
    df.insert(0, "loss", losses[(arr - 1) % len(losses)])
    df.insert(0, "delay", delays[(arr - 1) % len(delays)])
    df = df[df['bbr'] < 17]
    if to_print and is_max:
        print(df)
    return df


def eval_scores():
    path = CONFIG['scoring_path'][:CONFIG['scoring_path'].rfind('/')]
    files = os.listdir(path)
    max_num = max(int(file[file.rfind('_') + 1:-len('.txt')]) for file in files if file.endswith('.txt'))
    print('mean')
    show_table(path, CONFIG['abr'], max_num, np.mean)
    print('='*60)
    print('var')
    show_table(path, CONFIG['abr'], max_num, np.var, is_max=False)


if __name__ == '__main__':
    parse_arguments()
    CONFIG.update(get_config())


    if CONFIG['eval']:
        eval_scores()
        exit()


    # create settings file
    if CONFIG['generate_data']:
        create_setting_yaml(generating_data='randomized')
    else:
        create_setting_yaml(test=CONFIG['test'])
    
    send_clear_to_server()

    # special call for the code to work
    subprocess.check_call('sudo sysctl -w net.ipv4.ip_forward=1', shell=True)
    subprocess.check_call("rm -rf ./*.profile", shell=True,
                                      executable='/bin/bash')
    if not os.path.exists('../cc_monitoring'):
        os.mkdir('../cc_monitoring')

    scoring_dir = CONFIG['scoring_path'][:CONFIG['scoring_path'].rfind('/') + 1]
    start_maimahi_clients(CONFIG['num_clients'], scoring_dir, lambda _ : False)
    print('finished generating data')

    if CONFIG['test']:
        eval_scores()
    elif CONFIG['generate_data']:
        create_setting_yaml(generating_data='fixed')
        exit_condition = lambda setting_number: setting_number == (3 - 1) # 3 iterations
        start_maimahi_clients(CONFIG['num_clients'], scoring_dir, exit_condition)