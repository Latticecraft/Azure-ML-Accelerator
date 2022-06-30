# imports
import os, argparse
import numpy as np
import pandas as pd
import mlflow

from azureml.core import Run
from pathlib import Path


# define functions     
def main(ctx):
    # read in data
    sep = ','
    if args.separator == 'semicolon':
        sep = ';'
    elif args.separator == 'tab':
        sep = '\t'

    df = pd.read_csv(ctx['args'].input_csv, sep=sep)

    # print first 5 lines
    print( '\n'.join([f'{k}: {v}' for k, v in sorted(os.environ.items())]) )
    print(df.head())

    # log metrics
    mlflow.log_metric('dataframe rows', len(df))

    dict_missing = np.sum(df.isnull()).to_dict()
    for (k,v) in dict_missing.items():
        if dict_missing[k] > 0:
            mlflow.log_metric(f'missing.{k}', v)

    # save data to outputs
    df.to_csv("outputs/transformed.csv", index=False)
    df.to_csv((Path(ctx['args'].transformed_data) / "transformed.csv"), index=False)


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
    parser.add_argument("--input-csv", type=str, default='data/ifood_df.csv')
    parser.add_argument('--separator', type=str, default='comma')
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