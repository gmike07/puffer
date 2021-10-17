import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from models import ConstantModel, Exp3Kmeans, Exp3, SL_Model, Exp3Server
from config_creator import get_config
from argument_parser import parse_arguments


def get_server_model(model):
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
                    parsed_data['state'] = np.array(parsed_data['state'])
                    model.update(parsed_data)
                    self.send_response(406 + model.predict(parsed_data))
                    self.end_headers()
                elif 'clear' in parsed_data:
                    print('clearing...')
                    model.clear()
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
    print(f"Starting httpd server on {addr}:{port} with model {get_config()['model_name']}")
    httpd.serve_forever()



def start_server(model, addr, port, is_rl=False):
    if not is_rl:
        run_server(lambda: get_server_model(model), addr, port)
        return
    
    server_thread = Thread(target=lambda: run_server(lambda: get_server_model(model), addr, port))
    model_thread = Thread(target=lambda: model.train_rl())
    model_thread.start()
    server_thread.start()
    model_thread.join()
    server_thread.join()


def create_model(model_name='exp3', training=True, num_clients=5):
    if model_name == 'exp3Kmeans':
        return Exp3Kmeans(num_clients, should_clear_weights=False, is_training=training)
    if model_name == 'exp3':
        return Exp3(num_clients, should_clear_weights=False, is_training=training)

    if model_name == 'exp3Server':
        return Exp3Server(num_clients)
    if model_name == 'resetingExp3Kmeans':
        return Exp3Kmeans(num_clients, should_clear_weights=True, is_training=training)
    if model_name == 'constant':
        return ConstantModel()
    if model_name == 'sl':
        return SL_Model()


if __name__ == '__main__':
    parse_arguments()

    model = create_model(get_config()['model_name'], get_config()['training'], get_config()['num_clients'])
    if not get_config()['training']:
        model.load()
    is_rl = get_config()['model_name'] in ['rl', 'srl']
    start_server(model, 'localhost', get_config()['server_port'], is_rl)
