import os
import sys
import argparse
import subprocess
import signal
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(grandparentdir)
from config_creator import create_config, create_dir, get_config
from offline_test import clear_and_kill_all, prepare_env, train_simulation, test_simulation, eval_scores
import time


g_pserver = None
DEFAULT_SLEEP_TIME = 4 * 60


def signal_handler(sig, frame):
    if g_pserver is not None:
        os.killpg(os.getpgid(g_pserver.pid), signal.SIGKILL)
    clear_and_kill_all()
    sys.exit(0)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", default=-1, type=int, help="Specify the number of clients to train \ test with")
    parser.add_argument('-yid', "--yaml_input_dir", default='./helper_scripts/', help='Specify the directory of the yaml setting files')
    parser.add_argument("--abr", default='', help='Specify the abr if you want to work with a given one')
    parser.add_argument("-t", "--test", default=False, action='store_true', help='A flag to indicate that it is testing and not training')
    parser.add_argument("-v", "--eval", default=False, action='store_true', help='A flag to specify whether you want to show the results in of simulating')
    parser.add_argument("--epochs", default=-1, type=int, help='number of epochs to do training \ testing')
    parser.add_argument('-sp',"--scoring_path", default='', help='Specify the place to store the scoring data')
    parser.add_argument('-m', '--models', nargs='+', default=[])
    args = parser.parse_args()
    new_models = []
    for data in args.models:
        if data.isnumeric():
            new_models[-1] = (new_models[-1][0], int(data))
        else:
            new_models.append((data, args.epochs))
    args.models = new_models
    model_folder = '_'.join(sorted([model[0] for model in args.models]))
    args.clients = len(args.models) if args.clients == -1 else args.clients
    create_config(args.yaml_input_dir, args.abr, args.clients, args.test, args.eval, args.epochs, '', args.scoring_path)
    args.scoring_path = get_config()['scoring_path'] + model_folder + '/'
    if args.test:
        create_dir(args.scoring_path)
    get_config()['scoring_path'] = args.scoring_path
    return args


def start_server(args):
    server_command = 'python3 server_model.py'
    if args.test:
        server_command += ' -t'
    if args.scoring_path and args.test:
        server_command += f' -sp {args.scoring_path}'
    print(server_command)
    return subprocess.Popen(server_command, shell=True, preexec_fn=os.setsid)


def kill_previous_server():
    kill_command = """pgrep -f "model" | while read pid
do 
echo $pid
kill -9 $pid
done"""
    return subprocess.Popen(kill_command, shell=True, preexec_fn=os.setsid)


def main_train_test():
    global g_pserver
    args = parse_arguments()
    prepare_env(args)
    pid = kill_previous_server()
    time.sleep(10)
    pid.terminate()
    start_server(args)
    time.sleep(DEFAULT_SLEEP_TIME)
    if args.eval:
        create_config(args.yaml_input_dir, args.abr, args.clients, args.test, args.eval, args.epochs, 'stackingModel', args.scoring_path)
        get_config()['models'] = [model[0] for model in args.models]
        eval_scores()
    if args.test:
        create_config(args.yaml_input_dir, args.abr, args.clients, args.test, args.eval, args.epochs, 'stackingModel', args.scoring_path)
        get_config()['models'] = [model[0] for model in args.models]
        test_simulation()
    else:
        for (model_name, epochs) in args.models:
            create_config(args.yaml_input_dir, args.abr, args.clients, args.test, args.eval, epochs, model_name, args.scoring_path)
            train_simulation(model_name)
            time.sleep(DEFAULT_SLEEP_TIME)
    signal_handler(0, 0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main_train_test()