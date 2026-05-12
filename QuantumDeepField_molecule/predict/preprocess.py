#!/usr/bin/env python3

import argparse
from collections import defaultdict
import glob
import os
import pickle
import shutil
import sys

sys.path.append('../')
from train import preprocess as pp


def load_dict(filename):
    with open(filename, 'rb') as f:
        dict_load = pickle.load(f)
        dict_default = defaultdict(lambda: max(dict_load.values())+1)
        for k, v in dict_load.items():
            dict_default[k] = v
    return dict_default


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_trained')
    parser.add_argument('basis_set')
    parser.add_argument('radius', type=float)
    parser.add_argument('grid_interval', type=float)
    parser.add_argument('dataset_predict')
    parser.add_argument(
        '--backend',
        choices=['numpy', 'rust'],
        default='numpy',
        help="Forwarded to ``train.preprocess.create_dataset`` (see train/preprocess.py).",
    )
    parser.add_argument(
        '--rust-batch-size',
        type=int,
        default=64,
        help="Forwarded to ``train.preprocess.create_dataset`` when --backend rust.",
    )
    parser.add_argument(
        '--output-format',
        choices=['npy', 'shard', 'both'],
        default='npy',
        help="Forwarded to ``train.preprocess.create_dataset`` (npy / shard / both).",
    )
    args = parser.parse_args()
    dataset_trained = args.dataset_trained
    basis_set = args.basis_set
    radius = args.radius
    grid_interval = args.grid_interval
    dataset_predict = args.dataset_predict
    backend = args.backend
    rust_batch_size = args.rust_batch_size
    output_format = args.output_format

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dir_trained = os.path.join(project_root, 'dataset', dataset_trained) + os.sep
    dir_predict = os.path.join(project_root, 'dataset', dataset_predict) + os.sep

    filename = dir_trained + 'orbitaldict_' + basis_set + '.pickle'
    orbital_dict = load_dict(filename)
    N_orbitals = len(orbital_dict)

    print('Preprocess', dataset_predict, 'dataset.\n'
          'The preprocessed dataset is saved in', dir_predict, 'directory.\n'
          'If the dataset size is large, '
          'it takes a long time and consume storage.\n'
          'Wait for a while...')
    print('Backend:', backend,
          f'(rust_batch_size={rust_batch_size})' if backend == 'rust' else '',
          '  output:', output_format)
    print('-'*50)

    pp.create_dataset(dir_predict, 'test',
                      basis_set, radius, grid_interval, orbital_dict,
                      backend=backend, rust_batch_size=rust_batch_size,
                      output_format=output_format)
    if N_orbitals < len(orbital_dict):
        print('##################### Warning!!!!!! #####################\n'
              'The prediction dataset contains unknown atoms\n'
              'that did not appear in the training dataset.\n'
              'The parameters for these atoms have not been learned yet\n'
              'and must be randomly initialized at this time.\n'
              'Therefore, the prediction will be unreliable\n'
              'and we stop this process.\n'
              '#########################################################')
        for path in glob.glob(dir_predict + 'test_*'):
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)
    else:
        print('-'*50)
        print('The preprocess has finished.')
