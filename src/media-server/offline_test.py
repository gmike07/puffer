#!/usr/bin/env python3
import time
import os
import subprocess
import signal
import numpy as np
import pandas as pd
import os
import yaml
import argparse
import copy
import random


EPOCHS = 1
LOGS_FILE = './weights/logs.txt'
CCS = ['bbr', 'vegas', 'cubic']

BASE_PORT = 9360
REMOTE_BASE_PORT = 9222

DELAYS = {1: 30, 2: 40, 3: 60, 0: 100}
LOSSES = {1: 0.02, 2: 0.05, 3: 0.01, 0: 0.00}


def get_delay_loss(args, index):
    index = index % (len(DELAYS) * len(LOSSES))
    return DELAYS[int(index % len(DELAYS))], LOSSES[int(index / len(DELAYS))]


def run_offline_media_servers():
    run_server_html_cmd = './helper_scripts/python_migrate.sh'
    p1 = subprocess.Popen(run_server_html_cmd, shell=True, preexec_fn=os.setsid)
    time.sleep(5)
    run_servers_cmd = './helper_scripts/python_run_servers.sh'
    p2 = subprocess.Popen(run_servers_cmd, shell=True, preexec_fn=os.setsid)
    time.sleep(5)
    return p1, p2

def get_mahimahi_command(trace_dir, filename, trace_index, delay, loss):
    remote_port = REMOTE_BASE_PORT + trace_index
    port = BASE_PORT + trace_index
    mahimahi_chrome_cmd = "mm-delay {} ".format(delay)
    if loss != 0:
        mahimahi_chrome_cmd += "mm-loss uplink {} ".format(loss)
    mahimahi_chrome_cmd += "mm-link "
    # mahimahi_chrome_cmd += "--meter-downlink "
    mahimahi_chrome_cmd += "/home/mike/puffer/src/media-server/12mbps "
    mahimahi_chrome_cmd += "{}/{} ".format(trace_dir, filename)
    mahimahi_chrome_cmd += "-- sh -c 'chromium-browser disable-infobars --disable-gpu --headless --enable-logging=true --v=1 --remote-debugging-port={} http://$MAHIMAHI_BASE:8080/player/?wsport={} --user-data-dir=./{}.profile'".format(
                        remote_port, port, port)
    return mahimahi_chrome_cmd

def start_maimahi_clients(args, filedir, abr):
    logs_file = open(LOGS_FILE, 'w')
    plist = []
    try:
        trace_dir = args.trace_dir + 'test/' if args.test else args.trace_dir + 'train/'
        traces = os.listdir(trace_dir)
        for epoch in range(EPOCHS):
            for f in range(0, len(traces), args.clients):
                logs_file.write(
                    f"Epoch: {epoch}/{EPOCHS}. Files: {f}/{len(traces)}\n")
                logs_file.flush()

                p1, p2 = run_offline_media_servers()
                plist = [p1, p2]
                sleep_time = 3
                delay, loss = get_delay_loss(args, (int(f / args.clients)))
                for i in range(1, args.clients + 1):
                    index = f + 2 if args.test else f + i - 1
                    time.sleep(sleep_time)
                    mahimahi_chrome_cmd  = get_mahimahi_command(trace_dir, traces[index], i, delay, loss)
                    p = subprocess.Popen(mahimahi_chrome_cmd, shell=True,
                                         preexec_fn=os.setsid)
                    plist.append(p)

                time.sleep(60*10 - sleep_time * args.clients)
                for p in plist[2:]:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    time.sleep(sleep_time)
                for p in plist[:2]:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    time.sleep(3)
                if args.test:
                    arr = CCS + ['nn']
                    for i in range(len(arr)):
                        filepath = f'{filedir}{i+1}_abr_{abr}_{arr[i]}_{int(f / args.clients)}.txt'
                        if not os.path.exists(filepath):
                            with open(filepath, 'w') as f:
                                pass
                subprocess.check_call("rm -rf ./*.profile", shell=True,
                                      executable='/bin/bash')
                if int(f / args.clients) == len(DELAYS) * len(LOSSES) and args.test:
                    break
                if args.count_iter != -1 and int(f / args.clients) >= args.count_iter:
                    break
    except Exception as e:
        print("exception: " + str(e))
    finally:
        logs_file.close()
        for p in plist:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            subprocess.check_call("rm -rf ./*.profile", shell=True,
                                  executable='/bin/bash')


def main(args, filedir, abr):
    subprocess.check_call('sudo sysctl -w net.ipv4.ip_forward=1', shell=True)
    subprocess.check_call("rm -rf ./*.profile", shell=True,
                                      executable='/bin/bash')
    if not os.path.exists('../cc_monitoring'):
        os.mkdir('../cc_monitoring')
    start_maimahi_clients(args, filedir, abr)


def delete_keys_dct(dct, keys):
    for key in keys:
        dct.pop(key, None)


def create_settings(args):
    default_dictionary = yaml.load(open(args.yml_input_dir + "default.yml", 'r'), Loader=yaml.FullLoader)
    cc_dictionary = yaml.load(open(args.yml_input_dir + "cc.yml", 'r'), Loader=yaml.FullLoader)
    abr_dictionary = yaml.load(open(args.yml_input_dir + "abr.yml", 'r'), Loader=yaml.FullLoader)
    abr = abr_dictionary['abr']
    cc = cc_dictionary['cc']
    delete_keys_dct(abr_dictionary, ['abr'])
    delete_keys_dct(cc_dictionary, ['cc', 'python_weights_path', 'cpp_weights_path'])
    cc_scoring_path = cc_dictionary['cc_scoring_path']
    cc_monitoring_path = cc_dictionary['cc_monitoring_path']
    fingerprint = {'cc': cc, 'abr': abr}
    if len(abr_dictionary) != 0:
        fingerprint['abr_config'] = abr_dictionary
    if args.test:
        delete_keys_dct(cc_dictionary, ['cc_monitoring_path'])
        cc_dictionary['random_cc'] = False 
        model_path = cc_dictionary['model_path']
        delete_keys_dct(cc_dictionary, ['model_path'])
        fingerprint.update({'cc_config': cc_dictionary})
        default_dictionary.update({'experiments': [{
            'num_servers': 1,
            'fingerprint': copy.deepcopy(fingerprint)
        } for _ in range(len(CCS) + 1)]}) 
        for i in range(len(CCS)):
            default_dictionary['experiments'][i]['fingerprint']['cc'] = CCS[i]
        default_dictionary['experiments'][-1]['fingerprint']['cc'] = 'bbr'
        default_dictionary['experiments'][-1]['fingerprint']['cc_config']['model_path'] = model_path
    else:
        delete_keys_dct(cc_dictionary, ['model_path', 'cc_scoring_path'])
        cc_dictionary['random_cc'] = True
        fingerprint['cc_config'] = cc_dictionary
        default_dictionary.update({'experiments': [{
            'num_servers': args.clients,
            'fingerprint': fingerprint
        }]})

    with open(args.yml_output_dir + 'settings.yml', 'w') as outfile:
        yaml.dump(default_dictionary, outfile)
    with open(args.yml_output_dir + 'settings_offline.yml', 'w') as outfile:
        yaml.dump(default_dictionary, outfile)
    return (cc_scoring_path, abr) if args.test else (cc_monitoring_path, abr)



def create_settings_not_random(input_yaml_dir, yaml_output_dir):
    default_dictionary = yaml.load(open(input_yaml_dir + "default.yml", 'r'), Loader=yaml.FullLoader)
    cc_dictionary = yaml.load(open(input_yaml_dir + "cc.yml", 'r'), Loader=yaml.FullLoader)
    abr_dictionary = yaml.load(open(input_yaml_dir + "abr.yml", 'r'), Loader=yaml.FullLoader)
    abr = abr_dictionary['abr']
    cc = cc_dictionary['cc']
    delete_keys_dct(abr_dictionary, ['abr'])
    delete_keys_dct(cc_dictionary, ['cc', 'python_weights_path', 'cpp_weights_path'])
    cc_monitoring_path = cc_dictionary['cc_monitoring_path']
    fingerprint = {'cc': cc, 'abr': abr}
    if len(abr_dictionary) != 0:
        fingerprint['abr_config'] = abr_dictionary
    
    delete_keys_dct(cc_dictionary, ['model_path', 'cc_scoring_path'])
    cc_dictionary['random_cc'] = False
    fingerprint['cc_config'] = cc_dictionary
    default_dictionary.update({'experiments': [{
        'num_servers': 3,
        'fingerprint': copy.deepcopy(fingerprint)
    } for _ in range(len(CCS))]})
    for i in range(len(CCS)):
        default_dictionary['experiments'][i]['fingerprint']['cc'] = CCS[i]

    with open(yaml_output_dir + 'settings.yml', 'w') as outfile:
        yaml.dump(default_dictionary, outfile)
    with open(yaml_output_dir + 'settings_offline.yml', 'w') as outfile:
        yaml.dump(default_dictionary, outfile)
    return cc_monitoring_path, abr


def create_arr(filedir, abr, i, ccs, func):
    arr = np.empty((len(ccs)))
    filedir = filedir[:filedir.rfind('/')]
    files = os.listdir(filedir)
    for j in range(len(ccs)):
        # print(i, ccs[j])
        file = [f for f in files if f.find(f'abr_{abr}_{ccs[j]}_{i}.txt') != -1][0]
        lines = open(filedir + '/' + file, 'r').readlines()
        if len(lines) < 100:
            return None
        arr[j] = func(np.array([float(x) for x in lines]))
    return arr


def show_table(filedir, abr, max_iter, func, is_max=True):
    dfs = []
    ccs = CCS + ['nn']
    for i in range(max_iter):
        a = create_arr(filedir, abr, i, ccs, func)
        if a is not None:
            dct = {'exp': [i]}
            dct.update({ccs[i]: a[i] for i in range(len(ccs))})
            dfs.append(pd.DataFrame(dct))
    df = pd.concat(dfs)
    arr = df['exp']
    df = df.drop(['exp'], 1)
    if is_max:
        df['max-nn'] = np.max(df, axis=1) - df['nn']
    else:
        df['min-nn'] = df['nn'] - np.min(df, axis=1)
    df.insert(0, "exp", arr)
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", default=30, type=int)
    parser.add_argument("--trace-dir", default='./traces/final_traces/')
    parser.add_argument("--count_iter", default=-1)
    parser.add_argument("-t", "--test", default=False, action='store_true')
    parser.add_argument("-yid", "--yml-input-dir", default='/home/mike/puffer/helper_scripts/')
    parser.add_argument("-yod", "--yml-output-dir", default='/home/mike/puffer/src/')
    
    args = parser.parse_args()


    if args.test:
        args.clients = len(CCS) + 1
    filedir, abr = create_settings(args)
    main(args, filedir, abr)

    if args.test:
        filedir = yaml.load(open(args.yml_input_dir + "cc.yml", 'r'), Loader=yaml.FullLoader)['cc_scoring_path']
        abr = yaml.load(open(args.yml_input_dir + "abr.yml", 'r'), Loader=yaml.FullLoader)['abr']
        print('mean')
        show_table(filedir, abr, 16, np.mean)
        print('='*60)
        print('var')
        show_table(filedir, abr, 16, np.var, is_max=False)
    else:
        create_settings_not_random(args.yml_input_dir, args.yml_output_dir)
        args.clients = 3 * len(CCS)
        args.count_iters = 3
        main(args, filedir, abr)