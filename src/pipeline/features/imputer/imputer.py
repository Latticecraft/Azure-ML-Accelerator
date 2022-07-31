import os, argparse
import pickle
import numpy as np
import pandas as pd
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from sklearn.impute import KNNImputer, SimpleImputer


def main(ctx):
    # read in data
    dict_files = pd.read_pickle(ctx['args'].datasets_pkl + '/datasets.pkl')

    new_files = {}
    for imputation in ['mean', 'knn']:
        if imputation == 'mean':
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        else: # imputation == 'knn'
            imputer = KNNImputer(missing_values=np.nan, n_neighbors=5)

        for key in dict_files.keys():
            if 'X_train' in key:
                df_x = dict_files['X_train'] if 'X_train' in dict_files.keys() else dict_files['X_train_none']
                imputer.fit(df_x)

                new_files[get_key(key, 'X', 'train', imputation)] = pd.DataFrame(imputer.transform(dict_files[key]), columns=imputer.feature_names_in_)
                new_files[get_key(key, 'y', 'train', imputation)] = dict_files[get_key(key, 'y', 'train', '')]

                new_files[get_key(key, 'X', 'valid', imputation)] = pd.DataFrame(imputer.transform(dict_files[key.replace("train", "valid")]), columns=imputer.feature_names_in_)
                new_files[get_key(key, 'y', 'valid', imputation)] = dict_files[get_key(key, 'y', 'valid', '')]

                new_files[get_key(key, 'X', 'test', imputation)] = pd.DataFrame(imputer.transform(dict_files[key.replace("train", "test")]), columns=imputer.feature_names_in_)
                new_files[get_key(key, 'y', 'test', imputation)] = dict_files[get_key(key, 'y', 'test', '')]

    # save data to outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(new_files, f, protocol=pickle.HIGHEST_PROTOCOL)

    copy_tree('outputs', args.transformed_data)


def get_key(key, type, fold, imputer):
    arr = key.split('_')
    if imputer != '':
        if len(arr) == 2:
            return f'{type}_{fold}_{imputer}'
        elif len(arr) == 3:
            return f'{type}_{fold}_{imputer}_{arr[2]}'
        else:
            raise Exception('Unknown filename format')
    else:
        if len(arr) == 2:
            return f'{type}_{fold}'
        elif len(arr) == 3:
            return f'{type}_{fold}_{arr[2]}'
        else:
            raise Exception('Unknown filename format')


def start(args):
    os.makedirs('outputs', exist_ok=True)
    mlflow.start_run()
    mlflow.autolog()
    run = Run.get_context()
    tags = run.parent.get_tags()
    return {
        'args': args,
        'run': run,
        'tags': tags
    }


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--datasets-pkl', type=str, default='data')
    parser.add_argument('--transformed_data', type=str, help='Path of output data')

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == '__main__':
    # parse args
    args = parse_args()
    ctx = start(args)

    # run main function
    main(ctx)