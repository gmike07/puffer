import torch
import yaml


DELETED_COLS = ['file_index', 'chunk_index', 'max_pacing_rate', 'reordering',
                'advmss', 'pmtu', 'fackets', 'snd_mss', 'probes']
TO_DELETE_COLS = [x for x in DELETED_COLS if x != 'chunk_index']


QUALITIES = ['426x240', '640x360', '854x480', '1280x720', '1920x1080']
QUALITY_COLS = ['file_index', 'chunk_index', 'ssim', 'video_buffer',
                'cum_rebuffer', 'media_chunk_size', 'trans_time'] + \
                ['qoe', 'ssim_quality', 'ssim_change', 'rebuffer', 'bit_quality', 'bit_change']

CC_COLS = ['file_index', 'chunk_index', 'sndbuf_limited', 'rwnd_limited', 'busy_time',
            'delivery_rate', 'data_segs_out', 'data_segs_in', 'min_rtt', 'notsent_bytes',
            'segs_in', 'segs_out', 'bytes_received', 'bytes_acked', 'max_pacing_rate',
            'total_retrans', 'rcv_space', 'rcv_rtt', 'reordering', 'advmss',
            'snd_cwnd', 'snd_ssthresh', 'rttvar', 'rtt', 'rcv_ssthresh', 'pmtu',
            'last_ack_recv', 'last_data_recv', 'last_data_sent', 'fackets',
            'retrans', 'lost', 'sacked', 'unacked', 'rcv_mss', 'snd_mss', 'ato',
            'rto', 'backoff', 'probes', 'ca_state', 'timestamp']

CONFIG = {'epochs': 5, 'model_network_sizes': [500, 200, 100, 80, 50, 40, 30, 20],
          'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          'batch_size': 16, 'lr': 1e-4, 'betas': (0.5, 0.999),
          'weights_decay': 1e-4, 'version': 2.0, 'history_size': 7,
          'random_sample': 40,
          'prediction_size': len(set(QUALITY_COLS) - {'file_index', 'chunk_index'})}


def create_config(input_dir, yaml_input_path, abr):
    with open(yaml_input_path + 'abr.yml', 'r') as f:
        CONFIG['abr'] = yaml.safe_load(f)['abr']
    if abr != '':
        CONFIG['abr'] = abr
    with open(yaml_input_path + 'cc.yml', 'r') as f:
        cc_dct = yaml.safe_load(f)
        CONFIG.update({key: cc_dct[key] for key in 
                    ['history_size', 'ccs', 'scoring_function_type', 'buffer_length_coef', 'quality_change_qoef']})

        CONFIG['random_sample'] = cc_dct['random_sample_size']
        CONFIG['weights_path'] = cc_dct['python_weights_path']
        CONFIG['weights_cpp_path'] = cc_dct['cpp_weights_path']
        CONFIG['scoring_func'] = cc_dct['predict_score']

        CONFIG['sample_size'] = len(set(CC_COLS + CONFIG['ccs']) - set(DELETED_COLS))
        CONFIG['nn_input_size'] = CONFIG['history_size'] * (CONFIG['random_sample'] * \
                        CONFIG['sample_size'] + CONFIG['prediction_size']) + len(CONFIG['ccs'])
    CONFIG.update({'input_dir': input_dir})


def get_config():
    return CONFIG