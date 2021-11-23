import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Event

from models import create_model
from config_creator import CONFIG, get_config
from train_models_script import train_rl
from argument_parser import parse_arguments


def is_rl(name):
    return name in ['rl', 'srl']


def get_server_model(models_lst):   
    other_thread = [None]
    event_thread = [None]
    class HandlerClass(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            try:
                content_len = int(self.headers.get('Content-Length'))
                data = self.rfile.read(content_len)
                parsed_data = json.loads(data)
                if 'state' in parsed_data:
                    parsed_data['server_id'] -= 1
                    parsed_data['state'] = np.array(parsed_data['state']).reshape(1, -1)
                    # print('state length', parsed_data['state'].shape)
                    models_lst[0].update(parsed_data)
                    self.send_response(406 + models_lst[0].predict(parsed_data))
                    self.end_headers()
                elif 'clear' in parsed_data:
                    print('clearing...')
                    if models_lst[0] is not None:
                        models_lst[0].clear()
                    self.send_response(200, 'OK')
                    self.end_headers()
                elif 'switch_model' in parsed_data:
                    print(f"switching to model {parsed_data['model_name']} and load: {parsed_data['load']}")
                    if models_lst[0] is not None:
                        if event_thread[0] is not None:
                            event_thread[0].set()
                        models_lst[0].save()
                    
                    models_lst[0] = None
                    other_thread[0] = None
                    event_thread[0] = None

                    models_lst[0] = create_model(get_config()['num_clients'], parsed_data['model_name'])
                    if parsed_data['load'] is True:
                        models_lst[0].load()
                    elif is_rl(parsed_data['model_name']) and get_config()['training']:
                        event_thread[0] = Event()
                        other_thread[0] = Thread(target=lambda: train_rl(models_lst[0], event_thread[0], parsed_data['model_name']))
                        other_thread[0].start()
                    self.send_response(200, 'OK')
                    self.end_headers()
                else:
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


#create_model(get_config()['num_clients'], 'rl')
if __name__ == '__main__':
    parse_arguments()
    get_config()['batch_size'] = 1
    run_server(lambda: get_server_model([None]), 'localhost', get_config()['server_port'])
