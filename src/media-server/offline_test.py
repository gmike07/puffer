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


DELAYS = [100, 30, 40, 60]
LOSSES = [0, 0.02, 0.05, 0.01]
CONFIG = {'settings': np.array([(delay, loss) for delay in DELAYS for loss in LOSSES]),
          'remote_base_port': 9222, 'base_port': 9360}
good_traces = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 18, 20, 21, 22, 23, 24, 25, 26, 27, 
                28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 52, 
                53, 54, 55, 57, 58, 60, 61, 62, 63, 65, 66, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 
                79, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 103, 
                104, 105]

def get_delay_loss(args, index):
    delay, loss = CONFIG['settings'][index % len(CONFIG['settings'])]
    if args.loss != -1.0:
        loss = args.loss
    if args.delay != -1.0:
        delay = args.delay
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


def start_maimahi_clients(args, clients, filedir, abr, exit_condition):
    plist = []
    try:
        trace_dir = args.trace_dir + 'test/' if args.test else args.trace_dir + 'train/'
        traces = os.listdir(trace_dir)
        traces = list(sorted(traces))

        # loop over only intresting files
        # if args.test and args.all:
        #     dir = args.all_path
        #     files = os.listdir(dir[:dir.rfind('/')])
        #     files = [file for file in files if file.endswith('.txt')]
        #     max_num = max(int(file[file.rfind('_') + 1:-len('.txt')]) for file in files if file.endswith('.txt'))
        #     df = show_table(dir, abr, max_num, np.mean, True, False)
        #     traces = [traces[i] for i in df['num_experiment']]
        #     delays, losses = df['delay'], df['loss']
        
        for epoch in range(args.epochs):
            num_clients = 1 if args.all else clients
            for f in range(0, len(traces), num_clients):

                p1, p2 = run_offline_media_servers()
                plist = [p1, p2]
                sleep_time = 3
                setting = (epoch * int(len(traces) / num_clients)) + int(f / num_clients)
                delay, loss = get_delay_loss(args, setting)
                # if args.test and args.all:
                #     delay, loss = delays.iloc[f], losses.iloc[f]
                for i in range(1, clients + 1):
                    index = f if args.test else f + i - 1
                    time.sleep(sleep_time)
                    mahimahi_chrome_cmd = get_mahimahi_command(trace_dir, traces[index], i, delay, loss)
                    p = subprocess.Popen(mahimahi_chrome_cmd, shell=True,
                                         preexec_fn=os.setsid)
                    plist.append(p)

                time.sleep(60*10 - sleep_time * clients)
                for p in plist[2:]:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    time.sleep(sleep_time)
                for p in plist[:2]:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    time.sleep(3)
                if args.test:
                    arr = CONFIG['ccs'] + ['nn'] + ['nn']
                    for i in range(len(arr)):
                        filepath = f'{filedir}{i+1}_abr_{abr}_{arr[i]}_{setting}.txt'
                        if not os.path.exists(filepath):
                            with open(filepath, 'w') as f:
                                pass
                subprocess.check_call("rm -rf ./*.profile", shell=True,
                                      executable='/bin/bash')
                if exit_condition(setting):
                    break
    except Exception as e:
        print("exception: " + str(e))
    finally:
        for p in plist:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            subprocess.check_call("rm -rf ./*.profile", shell=True,
                                  executable='/bin/bash')


def main(args, clients, filedir, abr, exit_condition):
    subprocess.check_call('sudo sysctl -w net.ipv4.ip_forward=1', shell=True)
    subprocess.check_call("rm -rf ./*.profile", shell=True,
                                      executable='/bin/bash')
    if not os.path.exists('../cc_monitoring'):
        os.mkdir('../cc_monitoring')
    start_maimahi_clients(args, clients, filedir, abr, exit_condition)


def delete_keys_dct(dct, keys):
    for key in keys:
        dct.pop(key, None)


def create_settings(args):
    default_dictionary = yaml.load(open(args.yml_input_dir + "default.yml", 'r'), Loader=yaml.FullLoader)
    cc_dictionary = yaml.load(open(args.yml_input_dir + "cc.yml", 'r'), Loader=yaml.FullLoader)
    abr_dictionary = yaml.load(open(args.yml_input_dir + "abr.yml", 'r'), Loader=yaml.FullLoader)
    abr = abr_dictionary['abr']
    cc = cc_dictionary['cc']
    predict_score = cc_dictionary['predict_score']
    delete_keys_dct(abr_dictionary, ['abr'])
    delete_keys_dct(cc_dictionary, ['cc', 'python_weights_path', 'cpp_weights_path', 'ccs', 'predict_score'])
    cc_scoring_path = cc_dictionary['cc_scoring_path']
    cc_monitoring_path = cc_dictionary['cc_monitoring_path']
    fingerprint = {'cc': cc, 'abr': abr, 'ccs': CONFIG['ccs']}
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
        } for _ in range(len(CONFIG['ccs']) + 1)]}) 
        for i in range(len(CONFIG['ccs'])):
            default_dictionary['experiments'][i]['fingerprint']['cc'] = CONFIG['ccs'][i]
        default_dictionary['experiments'][-1]['fingerprint']['cc'] = 'bbr'
        default_dictionary['experiments'][-1]['fingerprint']['cc_config']['model_path'] = model_path
        default_dictionary['experiments'][-1]['fingerprint']['cc_config']['predict_score'] = predict_score
        default_dictionary['experiments'].append(copy.deepcopy(default_dictionary['experiments'][-1]))
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



def create_settings_not_random(input_yaml_dir, yaml_output_dir, clients=1):
    default_dictionary = yaml.load(open(input_yaml_dir + "default.yml", 'r'), Loader=yaml.FullLoader)
    cc_dictionary = yaml.load(open(input_yaml_dir + "cc.yml", 'r'), Loader=yaml.FullLoader)
    abr_dictionary = yaml.load(open(input_yaml_dir + "abr.yml", 'r'), Loader=yaml.FullLoader)
    abr = abr_dictionary['abr']
    cc = cc_dictionary['cc']
    delete_keys_dct(abr_dictionary, ['abr'])
    delete_keys_dct(cc_dictionary, ['cc', 'python_weights_path', 'cpp_weights_path', 'ccs', 'predict_score'])
    cc_monitoring_path = cc_dictionary['cc_monitoring_path']
    fingerprint = {'cc': cc, 'abr': abr, 'ccs': CONFIG['ccs']}
    if len(abr_dictionary) != 0:
        fingerprint['abr_config'] = abr_dictionary
    
    delete_keys_dct(cc_dictionary, ['model_path', 'cc_scoring_path'])
    cc_dictionary['random_cc'] = False
    fingerprint['cc_config'] = cc_dictionary
    default_dictionary.update({'experiments': [{
        'num_servers': clients,
        'fingerprint': copy.deepcopy(fingerprint)
    } for _ in range(len(CONFIG['ccs']))]})
    for i in range(len(CONFIG['ccs'])):
        default_dictionary['experiments'][i]['fingerprint']['cc'] = CONFIG['ccs'][i]

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
        file = [f for f in files if f.find(f'abr_{abr}_{ccs[j]}_{i}.txt') != -1][0]
        lines = open(filedir + '/' + file, 'r').readlines()
        if len(lines) < 100:
            return None
        arr[j] = func(np.array([float(x) for x in lines]))
    return arr


def show_table(filedir, abr, max_iter, func, is_max=True, to_print=True):
    dfs = []
    ccs = CONFIG['ccs'] + ['nn'] + ['nn']
    ccs_named = CONFIG['ccs'] + ['nn1'] + ['nn2']
    for i in range(max_iter):
        a = create_arr(filedir, abr, i, ccs, func)
        if a is not None:
            dct = {'exp': [i]}
            dct.update({ccs_named[i]: a[i] for i in range(len(ccs_named))})
            dfs.append(pd.DataFrame(dct))
    df = pd.concat(dfs)
    df['nn'] = (df['nn1'] + df['nn2']) / 2
    arr = df['exp']
    df = df.drop(['exp', 'nn1', 'nn2'], 1)
    if is_max:
        df['max-nn'] = np.max(df, axis=1) - df['nn']
    else:
        df['min-nn'] = df['nn'] - np.min(df, axis=1)
    df.insert(0, "loss", CONFIG['settings'][(arr - 1) % len(CONFIG['settings']), 1])
    df.insert(0, "delay", CONFIG['settings'][(arr - 1) % len(CONFIG['settings']), 0])
    # df.insert(0, "num_experiment", arr)
    df = df[df['nn'] < 17]
    if to_print:
        print(df)
    return df
    # print(list(arr))


def eval_scores(args):
    filedir = yaml.load(open(args.yml_input_dir + "cc.yml", 'r'), Loader=yaml.FullLoader)['cc_scoring_path']
    abr = yaml.load(open(args.yml_input_dir + "abr.yml", 'r'), Loader=yaml.FullLoader)['abr']
    files = os.listdir(filedir[:filedir.rfind('/')])
    max_num = max(int(file[file.rfind('_') + 1:-len('.txt')]) for file in files if file.endswith('.txt'))
    print('mean')
    show_table(filedir, abr, max_num, np.mean)
    print('='*60)
    print('var')
    show_table(filedir, abr, max_num, np.var, is_max=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", default=5, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--trace-dir", default='./traces/final_traces/')
    parser.add_argument("--count_iter", default=-1)
    parser.add_argument("-t", "--test", default=False, action='store_true')
    parser.add_argument("-a", "--all", default=False, action='store_true')
    parser.add_argument("-v", "--eval", default=False, action='store_true')
    parser.add_argument("-yid", "--yml-input-dir", default='/home/mike/puffer/helper_scripts/')
    parser.add_argument("--all_path", default="/home/mike/cc_scoring/linear_bba_all_1/cc_score_")
    parser.add_argument("-yod", "--yml-output-dir", default='/home/mike/puffer/src/')
    parser.add_argument("--delay", default=-1.0)
    parser.add_argument("--loss", default=0.0)
    
    args = parser.parse_args()
    
    CONFIG['ccs'] = yaml.load(open(args.yml_input_dir + "cc.yml", 'r'), Loader=yaml.FullLoader)['ccs']

    if args.eval:
        eval_scores(args)
        exit()

    filedir, abr = create_settings(args)
    clients = len(CONFIG['ccs']) + 2 if args.test else args.clients
    exit_condition = lambda setting_number: args.test and (setting_number == (len(CONFIG['settings']) - 1)) and (not args.all)

    main(args, clients, filedir, abr, exit_condition)
    print('finished generating data')

    if args.test:
        eval_scores(args)
    else:
        create_settings_not_random(args.yml_input_dir, args.yml_output_dir)
        clients = len(CONFIG['ccs']) # 3 per client
        exit_condition = lambda setting_number: setting_number == (3 - 1) # 3 iterations
        main(args, clients, filedir, abr, exit_condition)