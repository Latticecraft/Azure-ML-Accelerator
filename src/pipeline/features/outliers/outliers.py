# imports
import os, argparse
import pickle
import pandas as pd
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from pathlib import Path
from sklearn.ensemble import IsolationForest


# define functions
def main(ctx):
    # read in data
    dict_files = pd.read_pickle(ctx['args'].datasets_pkl + '/datasets.pkl')

    # determine possible outliers
    for key in dict_files.keys():
        if key.startswith('X_train'):
            df_train = dict_files[key]
            df_valid = dict_files[key.replace('train', 'valid')]
            df_test = dict_files[key.replace('train', 'test')]
            
            clf = IsolationForest(max_samples=100)
            clf.fit(df_train)
            df_train['OutlierIsoForest'] = clf.predict(df_train)
            df_valid['OutlierIsoForest'] = clf.predict(df_valid)
            df_test['OutlierIsoForest'] = clf.predict(df_test)

            dict_files[key] = df_train
            dict_files[key.replace('train', 'valid')] = df_valid
            dict_files[key.replace('train', 'test')] = df_test

    # print dfs
    for key in dict_files.keys():
        print(f'dict_files[{key}].head():')
        print(dict_files[key].head())

    # save data to outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(dict_files, f, protocol=pickle.HIGHEST_PROTOCOL)
    
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
    parser.add_argument("--transformed-data", type=str, default='data')

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