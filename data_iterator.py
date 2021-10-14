import numpy as np
import pandas as pd
import numpickle as npl
from config_creator import get_config
from models import OUTPUT_DIMS
import torch

MAX_SSIM = 60.0

class DataIterator:
    def __init__(self, answers, remove_bad=True, output_type='qoe'):
        self.answers = answers
        self.chunks = None
        self.current_df = -1
        self.offset = 0
        self.perm = None
        self.CONFIG = get_config()

        self.output_type = output_type
        self.output_size = OUTPUT_DIMS[self.output_type]
        self.answers['ssim_quality'] /= MAX_SSIM
        self.answers['ssim_change'] /= MAX_SSIM

        if remove_bad:
            good_files = self.answers[self.answers['qoe'] < 17]['file_index'].unique()
            self.answers = self.answers[self.answers.file_index.isin(good_files)]
        
        self.N = len(self.answers)
        self.items = np.sum(self.answers['chunk_index'] >= self.CONFIG['history_size']) \
                     - len(self.answers['file_index'].unique())


    def __len__(self):
        return int(self.items / self.CONFIG['batch_size'])

    def __iter__(self):
        self.offset = 0
        self.perm = np.arange(self.N)
        return self

    def get_history_chunk(self, chunk_index):
        chunks = self.chunks
        mask1 = (chunk_index - self.CONFIG['history_size']) < chunks['chunk_index']
        mask2 = chunks['chunk_index'] <= chunk_index
        chunk_history = chunks[mask1 & mask2]
        helper_arr = np.array([])
        helper_answer = self.answers.drop(['chunk_index', 'file_index'], 1)
        for history in range(self.CONFIG['history_size'] - 1, -1, -1):
            mask = chunk_history['chunk_index'] == (chunk_index - history)
            chunk_i = chunk_history[mask].drop(['chunk_index', 'file_index'], 1)
            random_indexes = np.random.choice(np.arange(len(chunk_i)),
                                              self.CONFIG['random_sample'],
                                              replace=True)
            # chunk_i = chunk_i.iloc[random_indexes].to_numpy()
            helper_arr = np.append(helper_arr, chunk_i.iloc[random_indexes].to_numpy().reshape(-1))
            helper_arr = np.append(helper_arr, helper_answer.iloc[self.offset - history].to_numpy().reshape(-1))
        mask = chunks['chunk_index'] == (chunk_index + 1)
        next_ccs = chunks[mask].iloc[0][-len(self.CONFIG['ccs']):]
        return np.append(helper_arr, next_ccs.to_numpy().reshape(-1))

    def apply_scoring(self, answer):
        if self.output_type == 'qoe':
            return answer.loc['qoe']
        if self.output_type == 'ssim':
            return answer.loc[['ssim_quality', 'ssim_change', 'rebuffer']]
        if self.output_type == 'bit_rate':
            return answer.loc[['bit_quality', 'bit_change', 'rebuffer']]
        # all
        return answer.loc[['ssim_quality', 'ssim_change', 'rebuffer', 'bit_quality', 'bit_change']]

    def __next__(self):
        batch_size = self.CONFIG['batch_size']
        answer_chunks = np.empty((batch_size, self.CONFIG['nn_input_size']))
        answer_metrics = np.empty((batch_size, self.output_size))
        i = 0
        while i < batch_size:
            if self.offset >= self.N - 1:
                raise StopIteration()
            answer = self.answers.iloc[self.perm[self.offset]]
            next_answer = self.answers.iloc[self.offset + 1]
            self.offset += 1
            file_index, chunk_index = answer['file_index'], answer['chunk_index']
            if chunk_index < self.CONFIG['history_size'] or \
                    next_answer['chunk_index'] != chunk_index + 1:
                continue
            if file_index != self.current_df:
                self.current_df = file_index
                df_path = f"{self.CONFIG['input_dir']}dfs/chunks_{int(file_index)}.npy"
                self.chunks = npl.load_numpickle(df_path)
            answer_chunks[i] = self.get_history_chunk(chunk_index)
            answer_metrics[i] = self.apply_scoring(answer)
            i += 1
        answer_chunks = torch.from_numpy(answer_chunks).to(self.CONFIG['device'])
        answer_metrics = answer_metrics / 100
        answer_metrics = torch.from_numpy(answer_metrics).to(self.CONFIG['device'])
        return answer_chunks, answer_metrics