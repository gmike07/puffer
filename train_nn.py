import argparse
from numpy.core.fromnumeric import squeeze
import yaml
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from normalization_functions import normalize_chunks, normalize_answers, DELETED_COLS
import torch
import torch.nn.functional as F
import numpickle as npl


QUALITIES = ['426x240', '640x360', '854x480', '1280x720', '1920x1080']
QUALITY_COLS = ['file_index', 'chunk_index', 'ssim', 'video_buffer',
                'cum_rebuffer', 'media_chunk_size', 'trans_time']

CC_COLS = ['file_index', 'chunk_index', 'sndbuf_limited', 'rwnd_limited', 'busy_time',
            'delivery_rate', 'data_segs_out', 'data_segs_in', 'min_rtt', 'notsent_bytes',
            'segs_in', 'segs_out', 'bytes_received', 'bytes_acked', 'max_pacing_rate',
            'total_retrans', 'rcv_space', 'rcv_rtt', 'reordering', 'advmss',
            'snd_cwnd', 'snd_ssthresh', 'rttvar', 'rtt', 'rcv_ssthresh', 'pmtu',
            'last_ack_recv', 'last_data_recv', 'last_data_sent', 'fackets',
            'retrans', 'lost', 'sacked', 'unacked', 'rcv_mss', 'snd_mss', 'ato',
            'rto', 'backoff', 'probes', 'ca_state', 'timestamp']

CONFIG = {'epochs': 20, 'network_sizes': [500, 200, 100, 80, 50, 40, 30, 20],
          'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          'batch_size': 16, 'lr': 1e-4, 'betas': (0.5, 0.999),
          'weights_decay': 1e-4, 'version': 2.0, 'history_size': 7,
          'random_sample': 40,
          'prediction_size': len(set(QUALITY_COLS) - {'file_index', 'chunk_index'})}
MIN_SSIM = 0.0
MAX_SSIM = 60.0

class DataLoaderRandom:
    def __init__(self, answers):
        self.answers = answers
        self.chunks = None
        self.current_df = -1
        self.offset = 0
        self.perm = None
        
        mask = (self.answers['ssim'] != -1)
        self.answers['ssim'][~mask] = MAX_SSIM
        helper = np.maximum(MIN_SSIM, np.minimum(MAX_SSIM, -10 * np.log10(1 - self.answers['ssim'][mask])))
        self.answers['ssim'][mask] = helper
        self.answers['ssim'] /= MAX_SSIM
        self.answers['video_buffer'] /= 20.0
        file_indexes = self.answers['file_index'].unique()
        good_files = []
        for file_index in file_indexes:
            file_answers = self.answers[self.answers['file_index'] == file_index]
            next_answers, curr_answers = file_answers[1:], file_answers[:-1]
            qoe = MAX_SSIM * next_answers['ssim'] - MAX_SSIM * CONFIG['scoring_lambda'] * np.abs(next_answers['ssim'] - curr_answers['ssim']) - \
                CONFIG['scoring_mu'] * np.maximum(0, answers['trans_time'] - answers['video_buffer'] * 20.0)
            if np.mean(qoe) < 17:
                good_files.append(file_index)
        self.answers = self.answers[self.answers.file_index.isin(good_files)]
        self.N = len(self.answers)
        self.items = np.sum(self.answers['chunk_index'] >= CONFIG['history_size']) \
                     - len(self.answers['file_index'].unique())


    def __len__(self):
        return int(self.items / CONFIG['batch_size'])

    def __iter__(self):
        self.offset = 0
        self.perm = np.arange(self.N)
        return self

    def get_history_chunk(self, chunk_index):
        chunks = self.chunks
        mask1 = (chunk_index - CONFIG['history_size']) < chunks['chunk_index']
        mask2 = chunks['chunk_index'] <= chunk_index
        chunk_history = chunks[mask1 & mask2]
        helper_arr = np.array([])
        helper_answer = self.answers.drop(['chunk_index', 'file_index'], 1)
        for history in range(CONFIG['history_size'] - 1, -1, -1):
            mask = chunk_history['chunk_index'] == (chunk_index - history)
            chunk_i = chunk_history[mask].drop(['chunk_index'], 1)
            random_indexes = np.random.choice(np.arange(len(chunk_i)),
                                              CONFIG['random_sample'],
                                              replace=CONFIG['resample'])
            # chunk_i = chunk_i.iloc[random_indexes].to_numpy()
            helper_arr = np.append(helper_arr, chunk_i.iloc[random_indexes].to_numpy().reshape(-1))
            helper_arr = np.append(helper_arr, helper_answer.iloc[self.offset - history].to_numpy().reshape(-1))
        mask = chunks['chunk_index'] == (chunk_index + 1)
        next_ccs = chunks[mask].iloc[0][-len(CONFIG['ccs']):]
        return np.append(helper_arr, next_ccs.to_numpy().reshape(-1))

    def apply_scoring(self, answers):
        # ['ssim', 'video_buffer', 'cum_rebuffer', 'media_chunk_size', 'trans_time']
        rebuffer_time = np.maximum(0, answers[:, 9] - answers[:, 6] * 20.0) # next trans_time - next video_buffer
        # next_quality, curr_quality = 0, 0
        curr_quality1 = MAX_SSIM * answers[:, 0]
        next_quality1 = MAX_SSIM * answers[:, 5]
        # curr_quality2 = (answers[:, 3] * 100.0 * 100000.0) / (1000 * 1000 * answers[:, 9])
        # next_quality2 = (answers[:, 8] * 100.0 * 100000.0) / (1000 * 1000 * answers[:, 9])
        return pd.DataFrame({
                        'quality1': next_quality1, 
                        'diff_quality1':  CONFIG['scoring_lambda'] * abs(next_quality1 - curr_quality1), 
                        'rebuffer': CONFIG['scoring_mu'] * rebuffer_time,
                        # 'quality2': next_quality2, 
                        # 'diff_quality2':  CONFIG['scoring_lambda'] * abs(next_quality2 - curr_quality2),
                        }).to_numpy()
        return next_quality - CONFIG['scoring_lambda'] * abs(next_quality - curr_quality) - CONFIG['scoring_mu'] * rebuffer_time

    def __next__(self):
        batch_size = CONFIG['batch_size']
        answer_chunks = np.empty((batch_size, CONFIG['input_size']))
        # answer_metrics_size = 1 if CONFIG['scoring_func'] else 2 * CONFIG['prediction_size']
        answer_metrics = np.empty((batch_size, 2 * CONFIG['prediction_size']))
        i = 0
        drop_lst = ['file_index', 'chunk_index']
        while i < batch_size:
            if self.offset >= self.N - 1:
                raise StopIteration()
            answer = self.answers.iloc[self.perm[self.offset]]
            next_answer = self.answers.iloc[self.offset + 1]
            self.offset += 1
            file_index, chunk_index = answer['file_index'], answer['chunk_index']
            answer = answer.drop(drop_lst)
            if chunk_index < CONFIG['history_size'] or \
                    next_answer['chunk_index'] != chunk_index + 1:
                continue
            if file_index != self.current_df:
                self.current_df = file_index
                df_path = f"{CONFIG['input_dir']}dfs/chunks_{int(file_index)}.npy"
                self.chunks = npl.load_numpickle(df_path)
            answer_chunks[i] = self.get_history_chunk(chunk_index)
            next_answer = next_answer.drop(drop_lst)
            answer_metrics[i] = np.append(answer, next_answer)
            i += 1
        answer_chunks = torch.from_numpy(answer_chunks).to(CONFIG['device'])
        if CONFIG['scoring_func']:
            answer_metrics = self.apply_scoring(answer_metrics).reshape((batch_size, -1)) / 100
        answer_metrics = torch.from_numpy(answer_metrics).to(CONFIG['device'])
        return answer_chunks, answer_metrics


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        sizes = CONFIG['network_sizes']
        output_size = 3 if CONFIG['scoring_func'] else 2 * CONFIG['prediction_size']
        self.model = torch.nn.Sequential(
            torch.nn.Linear(CONFIG['input_size'], sizes[0]),
            torch.nn.ReLU(),
            # torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(sizes[0], sizes[1]),
            # torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            # torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(sizes[1], sizes[2]),
            torch.nn.ReLU(),
            # torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(sizes[2], sizes[3]),
            torch.nn.ReLU(),
            # torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(sizes[3], sizes[4]),
            torch.nn.ReLU(),
            # torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(sizes[4], sizes[5]),
            torch.nn.ReLU(),
            # torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(sizes[5], sizes[6]),
            torch.nn.ReLU(),
            # torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(sizes[6], sizes[7]),
            torch.nn.ReLU(),
            # torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(sizes[7], output_size)
        ).double().to(CONFIG['device'])
        self.loss_quality = torch.nn.CrossEntropyLoss().to(
            device=CONFIG['device'])
        self.loss_metrics = torch.nn.MSELoss().to(device=CONFIG['device'])
        # self.loss_metrics = torch.nn.L1Loss().to(device=CONFIG['device'])

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=CONFIG['lr'],
                                          weight_decay=CONFIG['weights_decay'])

    def forward(self, x):
        return self.model(x)


def save_cpp_model(model, model_path):
    example = torch.rand(1, CONFIG['input_size']).double()
    traced_script_module = torch.jit.trace(model.model, example, check_trace=False)
    traced_script_module.save(model_path)


def train(model, loader):
    for epoch in range(CONFIG['epochs']):
        pbar = tqdm(iterable=iter(loader), ncols=200)
        indexes = np.array([0, 3, 5, 6, 8, 9])
        for (chunks, metrics) in pbar:
            predictions = model(chunks)
            # 'ssim', 'video_buffer', 'cum_rebuffer', 'media_chunk_size', 'trans_time'] --> 0, 3, 5
            if not CONFIG['scoring_func']:
                predictions = predictions[:, indexes]
                metrics = metrics[:, indexes]
            loss = model.loss_metrics(predictions, metrics)
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            pbar.set_description_str('epoch #{}'.format(epoch))
            pbar.set_postfix(loss=loss.mean().item())
        pbar.close()
        filename = f"weights_{str(epoch)}_abr_{CONFIG['abr']}_v{str(CONFIG['version'])}_{CONFIG['scoring_type']}.pt"
        torch.save({
            'model_state_dict': model.model.state_dict()
        }, f"{CONFIG['weights_path']}{filename}")
        save_cpp_model(model, f"{CONFIG['weights_cpp_path']}{filename}")


def filter_path(x: str):
    return x.endswith('.txt') and x.find(f"abr_{CONFIG['abr']}_") != -1


def generate_csv_from_file(file, file_index, f_chunk, f_answer, vector_cc,
                           skip=1):
    chunk_index, counter = 0, 0
    chunks = []
    for line in file:
        try:
            line = line.strip()
            if line == '':
                continue
            if line.startswith('new run,'):
                chunk_index = 0
                continue
            data = line.split(',')
            if data[0] == 'audio':
                continue
                # chunks = []
            elif data[0] == 'video':
                f_chunk.write('\n'.join(chunks) + '\n')
                chunks = []
                answer = ','.join([str(file_index), str(chunk_index)]) + ','
                answer += ','.join(str(i) for i in data[1:-3]) + ','
                answer += str(float(int(data[-3]) / 100000.0)) + ','
                answer += str(float(int(data[-2]) / 1000.0)) + '\n'
                f_answer.write(answer)
                chunk_index += 1
            else:
                if counter % skip == 0:
                    chunk = ','.join([str(file_index), str(chunk_index)]) + ','
                    chunk += ','.join(str(i) for i in data[1:]) + ','
                    chunk += ','.join(str(i) for i in vector_cc[data[0]])
                    chunks.append(chunk)
                counter = (counter + 1) % skip
        except Exception as e:
            print('shit', e, type(e))
            return
    return


def generate_dfs(files):
    chunks_csv_path = f"{CONFIG['input_dir']}chunks.csv"
    answers_csv_path = f"{CONFIG['input_dir']}answers.csv"
    vector_cc = {CONFIG['ccs'][i]: (np.arange(len(CONFIG['ccs'])) == i).astype(np.uint64)
                 for i in range(len(CONFIG['ccs']))}
    files = list(files)
    f_answer = open(answers_csv_path, 'w')
    f_answer.write(','.join(QUALITY_COLS) + '\n')
    if not os.path.exists(f"{CONFIG['input_dir']}dfs"):
        os.mkdir(f"{CONFIG['input_dir']}dfs")
    for i, file in enumerate(tqdm(iterable=files)):
        f = open(f"{CONFIG['input_dir']}{file}", 'r')
        f_chunk = open(chunks_csv_path, 'w')
        f_chunk.write(','.join(CC_COLS + CONFIG['ccs']) + '\n')
        generate_csv_from_file(f, i, f_chunk, f_answer, vector_cc)
        f.close()
        f_chunk.close()
        lst_df = [df.astype(np.uint64) for df in
                  pd.read_csv(chunks_csv_path, header=0, chunksize=5000)]
        df_chunks = pd.concat(lst_df).astype(np.uint64)
        npl.save_numpickle(normalize_chunks(df_chunks),
                           f"{CONFIG['input_dir']}dfs/chunks_{i}.npy",
                           all_numeric=True)
    f_answer.close()
    lst_df = [df.astype(np.float64) for df in
              tqdm(pd.read_csv(answers_csv_path, header=0, chunksize=5000))]
    df_answers = pd.concat(lst_df).astype(np.float64)
    npl.save_numpickle(normalize_answers(df_answers),
                       f"{CONFIG['input_dir']}dfs/answers.npy",
                       all_numeric=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', "--input_dir", default='../cc_monitoring/')
    parser.add_argument('-yid', "--yaml_input_dir", default='/home/mike/puffer/helper_scripts/')
    parser.add_argument("--abr", default='')
    parser.add_argument("-df", "--generate_dfs", default=False, action='store_true')
    parser.add_argument("-rs", "--resample", default=False, action='store_true')

    args = parser.parse_args()

    with open(args.yaml_input_dir + 'abr.yml', 'r') as f:
        CONFIG['abr'] = yaml.safe_load(f)['abr']
    if args.abr != '':
        CONFIG['abr'] = args.abr
    with open(args.yaml_input_dir + 'cc.yml', 'r') as f:
        cc_dct = yaml.safe_load(f)
        CONFIG.update({key: cc_dct[key] for key in 
                    ['history_size', 'ccs', 'scoring_type', 'scoring_mu', 'scoring_lambda']})

        CONFIG['random_sample'] = cc_dct['sample_size']
        CONFIG['weights_path'] = cc_dct['python_weights_path']
        CONFIG['weights_cpp_path'] = cc_dct['cpp_weights_path']
        CONFIG['scoring_func'] = cc_dct['predict_score']

        CONFIG['sample_size'] = len(set(CC_COLS + CONFIG['ccs']) - set(DELETED_COLS))
        CONFIG['input_size'] = CONFIG['history_size'] * (CONFIG['random_sample'] * \
                        CONFIG['sample_size'] + CONFIG['prediction_size']) + len(CONFIG['ccs'])
    CONFIG.update({'input_dir': args.input_dir, 'resample': args.resample})
    if args.generate_dfs:
        files = list(filter(filter_path, os.listdir(CONFIG['input_dir'])))
        print('generating chunks...')
        generate_dfs(files)
    answer_path = f"{CONFIG['input_dir']}dfs/answers.npy"
    answers = npl.load_numpickle(answer_path)
    loader = DataLoaderRandom(answers)
    model = Model()
    print(answers.describe())
    print('training...')
    train(model, loader)


if __name__ == '__main__':
    main()
