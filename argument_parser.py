import argparse
from config_creator import create_config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", default=-1, type=int, help="Specify the number of clients to train \ test with")
    parser.add_argument('-yid', "--yaml_input_dir", default='/home/mike/puffer/helper_scripts/', help='Specify the directory of the yaml setting files')
    parser.add_argument("--abr", default='', help='Specify the abr if you want to work with a given one')
    parser.add_argument("-t", "--test", default=False, action='store_true', help='A flag to indicate that it is testing and not training')
    parser.add_argument("-v", "--eval", default=False, action='store_true', help='A flag to specify whether you want to show the results in of simulating')
    parser.add_argument("-gd", "--generate_data", default=False, action='store_true', help='A flag to specify whether you want to generate data or not in the simulation')
    parser.add_argument("--contextless", default=False, action='store_true', help='A flag to specify whether you want to not use context in clustering')
    parser.add_argument('-mn',"--model_name", default='', help='Specify the model name if you want to work with a given one')

    args = parser.parse_args()
    create_config(args.yaml_input_dir, args.abr, args.clients, args.test, args.eval, args.generate_data, args.contextless, args.model_name)
    return args

if __name__ == '__main__':
    parse_arguments()