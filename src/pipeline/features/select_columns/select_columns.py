# imports
import os, argparse
import pandas as pd
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from pathlib import Path


# define functions
def main(ctx):
    # read in data
    df = pd.read_pickle(ctx['args'].marketing_csv)

    # print first 5 lines
    if ctx['args'].columns != 'None':
        cols = args.columns.replace('%20',' ').split(',')
        if ctx['args'].take_complement == 'False':
            df = df[cols]
        else:
            df = df[list(set(df.columns) - set(cols))]

    # save data to outputs
    df.to_pickle((Path('outputs') / 'datasets.pkl'))
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
    parser.add_argument("--marketing-csv", type=str, default='data/datasets.pkl')
    parser.add_argument("--transformed_data", type=str, help="Path of output data")
    parser.add_argument("--columns", type=str, default='Complain')
    parser.add_argument("--take_complement", type=str, default=False)

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