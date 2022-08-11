import os, argparse
import numpy as np
import pandas as pd
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree


def main(ctx):
    # read in data
    sep = ','
    if args.separator == 'semicolon':
        sep = ';'
    elif args.separator == 'tab':
        sep = '\t'

    # read csv and coerce float64 columns to float32
    df = pd.read_csv(ctx['args'].input_csv, sep=sep, nrows=100)

    float_cols = [c for c in df if df[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}

    df = pd.read_csv(ctx['args'].input_csv, sep=sep, engine='c', dtype=float32_cols)

    # print env variables, first 5 rows and datatypes
    print( '\n'.join([f'{k}: {v}' for k, v in sorted(os.environ.items())]) )
    print(df.dtypes)
    if len(df) > 5:
        [print(df.iloc[x]) for x in [0,1,2,3,4]]

    # log metrics
    mlflow.log_metric('dataframe rows', len(df))

    dict_missing = np.sum(df.isnull()).to_dict()
    for (k,v) in dict_missing.items():
        if dict_missing[k] > 0:
            mlflow.log_metric(f'missing.{k}', v)

    # save data to outputs
    df.to_csv('outputs/transformed.csv', index=False)
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
    parser.add_argument('--input-csv', type=str, default='data/ifood_df.csv')
    parser.add_argument('--separator', type=str, default='comma')
    parser.add_argument('--transformed_data', type=str, default='data')

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