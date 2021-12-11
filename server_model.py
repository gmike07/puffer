import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Event

from models.model_creator import create_model
from models.ae_trainer import train_ae
from models.rl_trainer import train_rl
from models.sl_trainer import train_sl
from config_creator import get_config, requires_helper_model, is_threaded_model
from argument_parser import parse_arguments

from typing import *

def get_lambda_trainer(model, model_name, event):
    if model_name == 'SLTrainer':
        return lambda: train_sl(model, event)
    if model_name == 'AETrainer':
        return lambda: train_ae(model, event)
    if model_name in ['rl', 'srl']:
        return lambda: train_rl(model, event)


def get_server_model(models_lst):   
    other_thread = [None]
    event_thread = [None]
    model_name = [None]
    finished_server_ids = [set()]
    class HandlerClass(BaseHTTPRequestHandler):
        def default_headers(self):
            self.send_response(200, 'OK')
            self.end_headers()


        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def handle_clear(self):
            print('clearing...')
            if models_lst[0] is not None:
                models_lst[0].clear()
            finished_server_ids[0] = []
            return self.default_headers()

        def handle_done(self):
            print('done...')
            if models_lst[0] is not None:
                models_lst[0].done()
            finished_server_ids[0] = set()
            return self.default_headers()

        def send_OK_and_prediction(self, prediction):
            self.default_headers()
            self.wfile.write(json.dumps({'cc': str(prediction)}).encode('utf-8'))

        def handle_state(self, parsed_data):
            if parsed_data['server_id'] in finished_server_ids[0]:
                print('ignore state')
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
                print('ignore state', parsed_data['server_id'])
                return self.send_OK_and_prediction(0)
            parsed_data['server_id'] -= 1
            print('predicting server', parsed_data['server_id'], end=' ')
            prediction = models_lst[0].predict(parsed_data)
            print('prediction is', prediction)
            self.send_OK_and_prediction(prediction)


        def handle_switch(self, parsed_data):
            print(f"switching to model {parsed_data['model_name']} and load: {parsed_data['load']}")
            finished_server_ids[0] = set()
            if model_name[0] == parsed_data['model_name']:
                if requires_helper_model(model_name[0]):
                    models_lst[0].update_helper_model(create_model(get_config()['num_clients'], parsed_data['helper_model']))
                return self.default_headers()
            if models_lst[0] is not None:
                if event_thread[0] is not None:
                    event_thread[0].set()
                models_lst[0].save()
            
            models_lst[0] = None
            other_thread[0] = None
            event_thread[0] = None

            models_lst[0] = create_model(get_config()['num_clients'], parsed_data['model_name'], parsed_data['helper_model'])
            if parsed_data['load'] is True:
                models_lst[0].load()
            elif not get_config()['test'] and is_threaded_model(parsed_data['model_name']):
                event_thread[0] = Event()
                other_thread[0] = Thread(target=get_lambda_trainer(models_lst[0], parsed_data['model_name'], event_thread[0]))
                other_thread[0].start()
            model_name[0] = parsed_data['model_name']
            return self.default_headers()

        def hanlde_message(self, parsed_data):
            if parsed_data['message'] == 'sock finished':
                finished_server_ids[0] |= {parsed_data['server_id']}
                print('server', parsed_data['server_id'], 'finished')

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
                elif 'switch_model' in parsed_data:
                    self.handle_switch(parsed_data)
                elif 'message' in parsed_data:
                    self.hanlde_message(parsed_data)
                else:
                    print(parsed_data)
                    self.send_response(400)
                    self.end_headers()
            except Exception as e:
                print('exception', e)
                self.send_response(400, "error occurred")
                self.end_headers()
    return HandlerClass


def run_server(server_handler, addr, port, server_class=HTTPServer):
    server_address = (addr, port)
    handler = server_handler()
    httpd = server_class(server_address, handler)
    print(f"Starting httpd server on {addr}:{port} with model None")
    httpd.serve_forever()


if __name__ == '__main__':
    parse_arguments()
    get_config()['batch_size'] = 1
    run_server(lambda: get_server_model([None]), 'localhost', get_config()['server_port'])
