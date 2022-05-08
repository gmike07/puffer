import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Event
import sys
from models.model_creator import create_model
from models.ae_trainer import train_ae
from models.rl_trainer import train_rl
from models.dnn_trainer import train_dnn
from config_creator import get_config, requires_helper_model, is_threaded_model
from argument_parser import parse_arguments
from contextlib import redirect_stdout, redirect_stderr
import signal
import socket
from typing import *

httpd = None


def signal_handler(sig, frame):
    global httpd
    if httpd is not None:
        httpd.shutdown()
    sys.exit(0)


def get_lambda_trainer(model, model_name, event, f=None):
    if model_name == 'DNNTrainer':
        return lambda: train_dnn(model, event, f=f)
    if model_name == 'AETrainer':
        return lambda: train_ae(model, event, f=f)
    if model_name in ['DRL', 'REINFORCE', 'REINFORCE_AE']:
        return lambda: train_rl(model, event)

def get_server_model(models_lst):   
    other_thread = [None]
    event_thread = [None]
    model_name = [None]
    finished_server_ids = [set()]
    helper_logger = [open('./all_logs_server_model.txt', 'w')]
    class HandlerClass(BaseHTTPRequestHandler):
        def default_headers(self):
            self.send_response(200, 'OK')
            self.end_headers()


        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def handle_clear(self):
            with redirect_stdout(helper_logger[0]):
                print('clearing...')
            print('clearing...')
            if models_lst[0] is not None:
                models_lst[0].clear()
            finished_server_ids[0] = set()
            return self.default_headers()

        def handle_done(self):
            with redirect_stdout(helper_logger[0]):
                print('done...')
            print('done...')
            if models_lst[0] is not None:
                models_lst[0].done()
            finished_server_ids[0] = set()
            return self.default_headers()

        def handle_test(self, is_test):
            print('testing ', is_test)
            get_config()['test'] = is_test
            return self.default_headers()

        def send_OK_and_prediction(self, prediction):
            self.default_headers()
            self.wfile.write(json.dumps({'cc': str(prediction)}).encode('utf-8'))

        def handle_state(self, parsed_data):
            if parsed_data['server_id'] in finished_server_ids[0]:
                print('ignore state', parsed_data['server_id'] - 1)
                return self.send_OK_and_prediction(0)
            parsed_data['server_id'] -= 1
            parsed_data['state'] = np.array(parsed_data['state']).reshape(1, -1)
            print('predicting server', parsed_data['server_id'], end=' ')
            models_lst[0].update(parsed_data)
            prediction = models_lst[0].predict(parsed_data)
            print('prediction is', prediction)
            self.send_OK_and_prediction(prediction)

        def handle_stateless(self, parsed_data):
            if parsed_data['server_id'] in finished_server_ids[0]:
                print('ignore state', parsed_data['server_id'] - 1)
                return self.send_OK_and_prediction(0)
            parsed_data['server_id'] -= 1
            print('predicting server', parsed_data['server_id'], end=' ')
            prediction = models_lst[0].predict(parsed_data)
            print('prediction is', prediction)
            self.send_OK_and_prediction(prediction)


        def handle_switch(self, parsed_data):
            with redirect_stdout(helper_logger[0]):
                print(f"switching to model {parsed_data['model_name']} and load: {parsed_data['load']}")
            helper_logger[0].flush()
            print(f"switching to model {parsed_data['model_name']} and load: {parsed_data['load']}")
            finished_server_ids[0] = set()
            if model_name[0] == parsed_data['model_name'] and parsed_data['model_name'] != 'stackingModel':
                if requires_helper_model(model_name[0]):
                    with redirect_stdout(helper_logger[0]):
                        models_lst[0].update_helper_model(create_model(get_config()['num_clients'], parsed_data['helper_model']))
                    helper_logger[0].flush()
                return self.default_headers()
            get_config()['batch_size'] = 1
            if models_lst[0] is not None:
                if event_thread[0] is not None:
                    event_thread[0].set()
                models_lst[0].save()
            
            models_lst[0] = None
            other_thread[0] = None
            event_thread[0] = None
            if len(parsed_data['models']) != 0:
                get_config()['models'] = parsed_data['models']
            with redirect_stdout(helper_logger[0]):
                models_lst[0] = create_model(get_config()['num_clients'], parsed_data['model_name'], parsed_data['helper_model'])
            helper_logger[0].flush()
            if parsed_data['load'] is True:
                models_lst[0].load()
            elif not get_config()['test'] and is_threaded_model(parsed_data['model_name']):
                event_thread[0] = Event()
                other_thread[0] = Thread(target=get_lambda_trainer(models_lst[0], parsed_data['model_name'], event_thread[0], helper_logger[0]))
                other_thread[0].start()
            model_name[0] = parsed_data['model_name']
            return self.default_headers()

        def hanlde_message(self, parsed_data):
            if parsed_data['message'] == 'sock finished':
                finished_server_ids[0] |= {parsed_data['server_id']}
                print('server', parsed_data['server_id'] - 1, 'finished')

        def do_POST(self):
            try:
                content_len = int(self.headers.get('Content-Length'))
                data = self.rfile.read(content_len)
                parsed_data = json.loads(data)
                if 'state' in parsed_data:
                    self.handle_state(parsed_data)
                elif 'stateless' in parsed_data:
                    self.handle_stateless(parsed_data)
                elif 'clear' in parsed_data:
                    self.handle_clear()
                elif 'done' in parsed_data:
                    self.handle_done()
                elif 'test' in parsed_data:
                    self.handle_test(parsed_data['test'])
                elif 'switch_model' in parsed_data:
                    self.handle_switch(parsed_data)
                elif 'message' in parsed_data:
                    self.hanlde_message(parsed_data)
                else:
                    print(parsed_data)
                    helper_logger[0].write("400-dct is " + str(parsed_data) +'\n')
                    helper_logger[0].flush()
                    self.send_response(400)
                    self.end_headers()
            except Exception as e:
                print('exception', e)
                helper_logger[0].write("400-exception is " + str(e) +'\n')
                helper_logger[0].flush()
                self.send_response(400, "error occurred")
                self.end_headers()
    return HandlerClass, helper_logger[0]



def run_server(server_handler, addr, port, server_class=HTTPServer):
    global httpd
    server_address = (addr, port)
    handler, f = server_handler()
    httpd = server_class(server_address, handler)
    f.write(f"Starting httpd server on {addr}:{port} with model None\n")
    f.flush()
    print(f"Starting httpd server on {addr}:{port} with model None")
    httpd.serve_forever()


if __name__ == '__main__':
    parse_arguments()
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)
    run_server(lambda: get_server_model([None]), 'localhost', get_config()['server_port'])
