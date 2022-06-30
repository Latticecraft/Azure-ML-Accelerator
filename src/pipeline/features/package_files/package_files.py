# imports
import os, argparse
import pickle
import pandas as pd
import mlflow

from azureml.core import Run
from pathlib import Path
from shutil import copyfile

# define functions
def main(ctx):
    # read in data
    df_feat = pd.read_pickle(ctx['args'].features_csv)
    df_label = pd.read_pickle(ctx['args'].label_csv)
    df_xtra = pd.read_pickle(ctx['args'].unwanted_csv)

    # perform task
    dict_files = {
        'X': df_feat,
        'y': df_label,
        'xtra': df_xtra
    }

    # write outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(dict_files, f, protocol=pickle.HIGHEST_PROTOCOL)

    copyfile('outputs/datasets.pkl', Path(ctx['args'].transformed_data) /'datasets.pkl')


def start(args):
    os.makedirs("outputs", exist_ok=True)
    mlflow.start_run()
    mlflow.autolog()
    run = Run.get_context()
    tags = run.parent.get_tags()
    return {
        'args': args,
        'run': run,
        'project': tags['project']
    }


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--features-csv", type=str, default='data')
    parser.add_argument("--label-csv", type=str, default='data')
    parser.add_argument('--unwanted-csv', type=str)
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