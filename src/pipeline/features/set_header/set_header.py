# imports
import os, argparse
import pandas as pd
import re
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from pathlib import Path


# define functions
def main(ctx):
    # read in data
    df = pd.read_csv(ctx['args'].marketing_csv)

    # print first 5 lines
    print(df.head())

    # remove spaces from header    
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', x) for x in df.columns]

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
    parser.add_argument("--transformed_data", type=str, default='data')

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