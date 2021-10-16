import os
import numpy as np
from config_creator import get_config, create_config
from models import SL_Model, ContextModel
from data_iterator import DataIterator
from sklearn.cluster import KMeans
from tqdm import tqdm
import pickle
import argparse


def create_cluster(context_model, loader):
    CONFIG = get_config()
    data = []
    print('collecting data...')
    for (chunks, _) in tqdm(loader):
        data.append(context_model.generate_context(chunks).reshape(-1))
    data = np.array(data)
    print('training kmeans...')
    kmeans = KMeans(n_clusters=CONFIG['num_clusters'])
    kmeans.fit(data)
    print('saving...')
    with open(f"{CONFIG['saving_cluster_path']}clusters_{CONFIG['num_clusters']}_{CONFIG['context_layers']}.pkl", 'wb') as f:
        pickle.dump(kmeans, f)
    return kmeans


def load_cluster():
    CONFIG = get_config()
    kmeans = None
    with open(f"{CONFIG['saving_cluster_path']}clusters_{CONFIG['num_clusters']}_{CONFIG['context_layers']}.pkl", 'rb') as f:
        kmeans = pickle.load(f)
    return kmeans


def calc_min_centers_dist(cluster_centers):
    b = cluster_centers.reshape(
        cluster_centers.shape[0], 1, cluster_centers.shape[1])
    clusters_dist = np.sqrt(
        np.einsum('ijk, ijk->ij', cluster_centers-b, cluster_centers-b))
    np.fill_diagonal(clusters_dist, np.amax(clusters_dist))
    return np.amin(clusters_dist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', "--input_dir", default='../cc_monitoring/')
    parser.add_argument('-yid', "--yaml_input_dir", default='/home/mike/puffer/helper_scripts/')
    parser.add_argument("--abr", default='')
    args = parser.parse_args()
    create_config(args.input_dir, args.yaml_input_dir, args.abr)
    get_config()['batch_size'] = 1    
    
    base_model = SL_Model()
    base_model.load()
    context_model = ContextModel(base_model)
    context_model.eval()

    if not os.path.exists(get_config()['saving_cluster_path']):
        os.mkdir(get_config()['saving_cluster_path'])
 
    loader = DataIterator(remove_bad=get_config()['remove_bad_files'], output_type='ssim', remove_action=True)
    kmeans = create_cluster(context_model, loader)
    min_dist = calc_min_centers_dist(kmeans.cluster_centers_)
    print(f'min centers dist {min_dist}')


if __name__ == '__main__':
    main()