# imports
import os, argparse, json
import pandas as pd
import urllib.parse
import mlflow

from azureml.core import Run
from pathlib import Path


# define functions
def main(ctx):
    # read in data
    df = pd.read_pickle(ctx['args'].marketing_csv)

    # print first 5 lines
    print(df.head())

    # remove spaces from header
    if ctx['args'].replacements != 'placeholder':
        replacements = json.loads(urllib.parse.unquote(ctx['args'].replacements))
        df = df.replace(replacements)

    df = df.replace(',', ' ', regex=True)

    # save data to outputs
    df.to_pickle((Path('outputs') / 'datasets.pkl'))
    df.to_pickle((Path(ctx['args'].transformed_data) / 'datasets.pkl'))


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
    parser.add_argument("--marketing-csv", type=str, default='data/transformed.csv')
    parser.add_argument('--replacements', type=str)
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