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
    dict_orig = pd.read_pickle(ctx['args'].datasets_pkl + '/datasets.pkl')

    dict_new = {}
    for imputation in ['mean', 'knn']:
        if imputation == 'mean':
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        else: # imputation == 'knn'
            imputer = KNNImputer(missing_values=np.nan, n_neighbors=5)

        for key in dict_orig.keys():
            if 'X_train' in key:
                dict_new[get_key(key, 'X', 'train', imputation)] = pd.DataFrame(imputer.fit_transform(dict_orig[key]), columns=imputer.feature_names_in_)
                dict_new[get_key(key, 'y', 'train', imputation)] = dict_orig[get_key(key, 'y', 'train')]

                dict_new[get_key(key, 'X', 'valid', imputation)] = pd.DataFrame(imputer.transform(dict_orig[key.replace("train", "valid")]), columns=imputer.feature_names_in_)
                dict_new[get_key(key, 'y', 'valid', imputation)] = dict_orig[get_key(key, 'y', 'valid')]

                dict_new[get_key(key, 'X', 'test', imputation)] = pd.DataFrame(imputer.transform(dict_orig[key.replace("train", "test")]), columns=imputer.feature_names_in_)
                dict_new[get_key(key, 'y', 'test', imputation)] = dict_orig[get_key(key, 'y', 'test')]

    # save data to outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(dict_new, f, protocol=pickle.HIGHEST_PROTOCOL)

    copy_tree('outputs', args.transformed_data)


def get_key(key, type, fold, imputer=None):
    arr = key.split('_')
    if imputer != None:
        return f'{type}_{fold}_{imputer}_{arr[2]}'
    else:
        return f'{type}_{fold}_{arr[2]}'


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