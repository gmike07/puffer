import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Event

from models.model_creator import create_model
from models.ae_trainer import AETrainer, train_ae
from models.rl_trainer import RLTrainer, train_rl
from models.sl_trainer import SLTrainer, train_sl
from config_creator import CONFIG, get_config
from argument_parser import parse_arguments


def get_lambda_trainer(model):
    if isinstance(model, SLTrainer):
        event = Event()
        return Thread(lambda: train_sl(model, event)), event
    if isinstance(model, RLTrainer):
        event = Event()
        return Thread(lambda: train_rl(model, event)), event
    if isinstance(model, AETrainer):
        event = Event()
        return Thread(lambda: train_ae(model, event)), event
    return None, None
    


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
                    prediction = models_lst[0].predict(parsed_data)
                    print('predicting server', parsed_data['server_id'], 'prediction is', prediction)
                    self.send_response(406 + prediction)
                    self.end_headers()
                elif 'clear' in parsed_data:
                    print('clearing...')
                    if models_lst[0] is not None:
                        models_lst[0].clear()
                    self.send_response(200, 'OK')
                    self.end_headers()
                elif 'done' in parsed_data:
                    print('clearing...')
                    if models_lst[0] is not None:
                        models_lst[0].done()
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
                    other_thread[0], event_thread[0] = get_lambda_trainer(models_lst[0])
                    if parsed_data['load'] is True:
                        models_lst[0].load()
                    elif not get_config()['test'] and other_thread[0] is not None:
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
