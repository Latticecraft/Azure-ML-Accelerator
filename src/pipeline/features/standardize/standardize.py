import os, argparse
import pandas as pd
import pickle
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from pathlib import Path
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler


def main(ctx):
    # read in data
    dict_files = pd.read_pickle(ctx['args'].datasets_pkl + '/datasets.pkl')

    df_X = dict_files['X_train']
    cols = list(df_X.columns)
    numeric_cols = [x for x in df_X.columns if 'float' in df_X[x].dtype.name or 'int' in df_X[x].dtype.name]

    ct = make_column_transformer((StandardScaler(), numeric_cols))
    dict_files['X_train'] = pd.DataFrame(ct.fit_transform(df_X), columns=cols)
    dict_files['X_valid'] = pd.DataFrame(ct.transform(dict_files['X_valid']), columns=cols)
    dict_files['X_test'] = pd.DataFrame(ct.transform(dict_files['X_test']), columns=cols)

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
    parser.add_argument('--transformed-data', type=str, default='data')

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