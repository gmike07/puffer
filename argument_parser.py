import argparse
from config_creator import create_config

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", default=5, type=int, help="Specify the number of clients to train \ test with")
    parser.add_argument('-ip', "--input_dir", default='../cc_monitoring/', help='Specify the directory of the training input to train the supervised model')
    parser.add_argument('-yid', "--yaml_input_dir", default='/home/mike/puffer/helper_scripts/', help='Specify the directory of the yaml setting files')
    parser.add_argument("--abr", default='', help='Specify the abr if you want to work with a given one')
    parser.add_argument("--model_name", default='constant', help='Specify the model you want to train \ test in the server')
    parser.add_argument("-t", "--test", default=False, action='store_true', help='A flag to indicate that it is testing and not training')
    args = parser.parse_args()
    create_config(args.input_dir, args.yaml_input_dir, args.abr, args.model_name, args.clients, args.test)

if __name__ == '__main__':
    parse_arguments()