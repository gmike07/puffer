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
from offline_test import clear_and_kill_all, prepare_env, train_simulation, test_simulation, eval_scores, test_simulation_model, eval_scores_model
import time


g_pserver = None
DEFAULT_SLEEP_TIME = 30

class modelData:
    def __init__(self, model_name):
        self.model_name = model_name
        self.epochs = -1
        self.trace_sets = ''
        self.load_model = False

    def update_load(self):
        self.load_model = True

    def update_epochs(self, epochs: int):
        self.epochs = epochs
        self.trace_sets = 'l' * epochs

    def __repr__(self):
        return f'model: {self.model_name}, epochs: {self.epochs},  load: {self.load_model}\n'


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
    parser.add_argument("-ts", "--test_seperated", default=False, action='store_true', help='A flag to indicate that it is testing and not training')
    parser.add_argument("-v", "--eval", default=False, action='store_true', help='A flag to specify whether you want to show the results in of simulating')
    parser.add_argument("-vs", "--eval_seperated", default=False, action='store_true', help='A flag to specify whether you want to show the results in of simulating')
    parser.add_argument("--epochs", default=-1, type=int, help='number of epochs to do training \ testing')
    parser.add_argument('-sp',"--scoring_path", default='', help='Specify the place to store the scoring data')
    parser.add_argument('-m', '--models', nargs='+', default=[])
    args = parser.parse_args()
    new_models = []
    curr_model = None
    for data in args.models:
        if data.isnumeric():
            curr_model.update_epochs(int(data))
        elif data == 'load':
            curr_model.update_load()
        else:
            curr_model = modelData(data)
            new_models.append(curr_model)
            if args.epochs > 0:
                curr_model.update_epochs(args.epochs)
    args.models = new_models
    args.clients = len(args.models) if args.clients == -1 or args.test else args.clients
    create_config(args.yaml_input_dir, args.abr, args.clients, args.test or args.test_seperated, args.eval, args.epochs, '', args.scoring_path)
    model_folder = '_'.join(sorted([model_data.model_name for model_data in args.models])) + f"_scoring_{get_config()['buffer_length_coef']}"
    args.scoring_path = get_config()['scoring_path'] + model_folder + '/'
    if args.test:
        create_dir(args.scoring_path)
    get_config()['scoring_path'] = args.scoring_path
    return args


def start_server(args):
    server_command = 'python3 server_model.py'
    if args.test or args.test_seperated:
        server_command += ' -t'
    if args.scoring_path and (args.test or args.test_seperated):
        server_command += f' -sp {args.scoring_path}'
    if args.clients != -1:
        server_command += f' --clients {args.clients}'
    print(server_command)
    return subprocess.Popen(server_command, shell=True, preexec_fn=os.setsid)


def kill_previous_server():
    kill_command = """pgrep -f "model" | while read pid
do
echo $pid
kill -9 $pid
done"""
    return subprocess.Popen(kill_command, shell=True, preexec_fn=os.setsid)


def eval(args):
    if args.eval:
        create_config(args.yaml_input_dir, args.abr, args.clients, args.test, args.eval, args.epochs, 'stackingModel', args.scoring_path)
        get_config()['models'] = list(map(lambda x: x.model_name, args.models))
        eval_scores()
        return True
    elif args.eval_seperated:
        create_config(args.yaml_input_dir, args.abr, args.clients, args.test, args.eval, args.epochs, 'stackingModel', args.scoring_path)
        get_config()['models'] = list(map(lambda x: x.model_name, args.models))
        eval_scores_model(get_config()['models'], get_config()['scoring_path'][:get_config()['scoring_path'].rfind('/') + 1])
        return True
    return False



def main_train_test():
    global g_pserver
    args = parse_arguments()
    prepare_env(args)
    pid = kill_previous_server()
    time.sleep(10)
    pid.terminate()
    if eval(args):
        signal_handler(0, 0)
    start_server(args)
    time.sleep(DEFAULT_SLEEP_TIME)
    if args.test:
        create_config(args.yaml_input_dir, args.abr, args.clients, args.test, args.eval, args.epochs, 'stackingModel', args.scoring_path)
        get_config()['models'] = list(map(lambda x: x.model_name, args.models))
        test_simulation()
    elif args.test_seperated:
        create_config(args.yaml_input_dir, args.abr, args.clients, args.test_seperated, args.eval, args.epochs, 'stackingModel', args.scoring_path)
        get_config()['models'] = list(map(lambda x: x.model_name, args.models))
        test_simulation_model(get_config()['models'])
    else:
        for model_data in args.models:
            create_config(args.yaml_input_dir, args.abr, args.clients, args.test, args.eval, model_data.epochs, model_data.model_name, args.scoring_path)
            train_simulation(model_data.model_name)
            if 'Cluster' not in model_data.model_name:
                time.sleep(DEFAULT_SLEEP_TIME)
            else:
                time.sleep(15 * 60)
    signal_handler(0, 0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main_train_test()