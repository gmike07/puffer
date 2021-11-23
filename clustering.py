import os
import numpy as np
from argument_parser import parse_arguments
from config_creator import get_config
from models import ClusterModel
from data_iterator import DataIterator
from tqdm import tqdm


def create_cluster(loader):
    CONFIG = get_config()
    model_config = {'cluster_contextless': CONFIG['cluster_contextless'], 
                    'num_clusters': 0, 'context_layers': [], 'cluster_name': '',
                    'sl_model_name': '', 'output_type': CONFIG['sl_output_type'], 
                    'ae_model_name': '', 'ae_sizes': [], 
                    'cluster_type': 'ae' if CONFIG['model_name'] == 'ae' else 'sl'}
    kmeansModel = ClusterModel(model_config)
    data = []
    print('collecting data...')
    for (chunks, _) in tqdm(loader):
        data.append(kmeansModel.get_context(chunks))
    data = np.array(data)
    print('training kmeans...')
    kmeansModel.fit(data)
    print('saving...')
    kmeansModel.save()
    return kmeansModel.kmeans


def calc_min_centers_dist(cluster_centers):
    b = cluster_centers.reshape(
        cluster_centers.shape[0], 1, cluster_centers.shape[1])
    clusters_dist = np.sqrt(
        np.einsum('ijk, ijk->ij', cluster_centers-b, cluster_centers-b))
    np.fill_diagonal(clusters_dist, np.amax(clusters_dist))
    return np.amin(clusters_dist)


if __name__ == '__main__':
    parse_arguments()
    get_config()['batch_size'] = 1    

    if not os.path.exists(get_config()['saving_cluster_path']):
        os.mkdir(get_config()['saving_cluster_path'])
 
    loader = DataIterator(remove_bad=get_config()['cluster_contextless'], output_type='ssim', remove_action=True)
    kmeans = create_cluster(loader)
    min_dist = calc_min_centers_dist(kmeans.cluster_centers_)
    print(f'min centers dist {min_dist}')