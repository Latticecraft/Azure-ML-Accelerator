# imports
import os, argparse, json
import pickle
import pandas as pd
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree


# define functions
def main(ctx):
    # read in data
    dict_files = pd.read_pickle(ctx['args'].datasets_pkl + '/datasets.pkl')
    df = dict_files['X_train']

    # log metrics
    metrics = {}
    for c in df.columns:
        if df[c].dtype.name == 'category':
            metrics[f'stats.{c}_mode'] = df[c].mode()
        if df[c].dtype.name.startswith('float') or df[c].dtype.name.startswith('int'):
            metrics[f'stats.{c}_mean'] = df[c].mean()

    # save data to outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(dict_files, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('outputs/metrics.json', 'w') as f:
        json.dump(metrics, f)

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
    parser.add_argument("--transformed-data", type=str, help="Path of output data")

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