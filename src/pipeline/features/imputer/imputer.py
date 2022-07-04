# imports
import os, argparse
import pickle
import numpy as np
import pandas as pd
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from sklearn.impute import KNNImputer, SimpleImputer

# define functions
def main(ctx):
    # read in data
    dict_files = pd.read_pickle(ctx['args'].datasets_pkl + '/datasets.pkl')

    new_files = {}
    for imputation in ['mean', 'knn']:
        if imputation == 'mean':
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        else: # imputation == 'knn'
            imputer = KNNImputer(missing_values=np.nan, n_neighbors=5)

        imputer.fit(dict_files['X_train'])

        new_files[f'X_train_{imputation}'] = pd.DataFrame(imputer.transform(dict_files['X_train']), columns=imputer.feature_names_in_)
        new_files[f'y_train_{imputation}'] = dict_files['y_train']
        
        new_files[f'X_valid_{imputation}'] = pd.DataFrame(imputer.transform(dict_files['X_valid']), columns=imputer.feature_names_in_)
        new_files[f'y_valid_{imputation}'] = dict_files['y_valid']

        new_files[f'X_test_{imputation}'] = pd.DataFrame(imputer.transform(dict_files['X_test']), columns=imputer.feature_names_in_)
        new_files[f'y_test_{imputation}'] = dict_files['y_test']

    # save data to outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(new_files, f, protocol=pickle.HIGHEST_PROTOCOL)

    copy_tree('outputs', args.transformed_data)


def start(args):
    os.makedirs("outputs", exist_ok=True)
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
    parser.add_argument("--datasets-pkl", type=str, default='data')
    parser.add_argument("--transformed_data", type=str, help="Path of output data")

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()
    ctx = start(args)

    # run main function
    main(ctx)