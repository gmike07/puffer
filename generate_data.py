import argparse
import yaml
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import numpickle as npl
from config_creator import CC_COLS, QUALITY_COLS, create_config, get_config

def generate_csv_from_file(file, file_index, f_chunk, f_answer, skip=1):
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
                answer += ','.join(str(i) for i in data[1:]) + "\n"
                f_answer.write(answer)
                chunk_index += 1
            else:
                if counter % skip == 0:
                    chunk = ','.join([str(file_index), str(chunk_index)]) + ','
                    chunk += ','.join(str(i) for i in data)
                    chunks.append(chunk)
                counter = (counter + 1) % skip
        except Exception as e:
            print('shit', e, type(e))
            return


def generate_dfs(files, input_dir, ccs):
    chunks_csv_path = f"{input_dir}chunks.csv"
    answers_csv_path = f"{input_dir}answers.csv"
    files = list(files)
    f_answer = open(answers_csv_path, 'w')
    f_answer.write(','.join(QUALITY_COLS) + '\n')
    if not os.path.exists(f"{input_dir}dfs"):
        os.mkdir(f"{input_dir}dfs")
    for i, file in enumerate(tqdm(iterable=files)):
        f = open(f"{input_dir}{file}", 'r')
        f_chunk = open(chunks_csv_path, 'w')
        f_chunk.write(','.join(CC_COLS + ccs) + '\n')
        generate_csv_from_file(f, i, f_chunk, f_answer)
        f.close()
        f_chunk.close()
        lst_df = [df.astype(np.float64) for df in
                  pd.read_csv(chunks_csv_path, header=0, chunksize=5000)]
        df_chunks = pd.concat(lst_df).astype(np.float64)
        npl.save_numpickle(df_chunks, f"{input_dir}dfs/chunks_{i}.npy",
                           all_numeric=True)
    f_answer.close()
    lst_df = [df.astype(np.float64) for df in
              tqdm(pd.read_csv(answers_csv_path, header=0, chunksize=5000))]
    df_answers = pd.concat(lst_df).astype(np.float64)
    npl.save_numpickle(df_answers, f"{input_dir}dfs/answers.npy", all_numeric=True)


def filter_path(x: str, abr: str):
    return x.endswith('.txt') and x.find(f"abr_{abr}_") != -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', "--input_dir", default='../cc_monitoring/')
    parser.add_argument('-yid', "--yaml_input_dir", default='/home/mike/puffer/helper_scripts/')
    parser.add_argument("--abr", default='')

    args = parser.parse_args()
    create_config(args.input_dir, args.yaml_input_dir, args.abr)
    CONFIG = get_config()
    abr, ccs = CONFIG['abr'], CONFIG['ccs']
    files = list(filter(lambda x: filter_path(x, abr), os.listdir(args.input_dir)))
    print('generating chunks...')
    generate_dfs(files, args.input_dir, ccs)
    answer_path = f"{args.input_dir}dfs/answers.npy"
    answers = npl.load_numpickle(answer_path)
    print(answers.describe())


if __name__ == '__main__':
    main()
