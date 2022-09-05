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

    for imputation in ['mean', 'knn']:
        if imputation == 'mean':
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        else: # imputation == 'knn'
            imputer = KNNImputer(missing_values=np.nan, n_neighbors=5)
            
        imputer.fit_transform(dict_files['X_train'])
        dict_files[f'imputer____{imputation}'] = imputer 

    # save data to outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(dict_files, f, protocol=pickle.HIGHEST_PROTOCOL)

    copy_tree('outputs', args.transformed_data)


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