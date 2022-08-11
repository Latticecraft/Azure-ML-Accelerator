import os, argparse, json
import numpy as np
import pandas as pd
import urllib.parse
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def main(ctx):
    # read in data
    df = pd.read_pickle(ctx['args'].input_data)

    # perform any replacements specified
    if ctx['args'].replacements != 'None':
        replacements = json.loads(urllib.parse.unquote(ctx['args'].replacements))
        df = df.replace(replacements)

    if ctx['args'].datatypes != 'None':
        datatypes = json.loads(urllib.parse.unquote(ctx['args'].datatypes))
        for k,v in datatypes.items():
            df[k] = df[k].astype(v)

    # remove spaces from header
    df = df.replace(',', ' ', regex=True)

    # encode label
    le = LabelEncoder()
    df[ctx['args'].label] = le.fit_transform(df[ctx['args'].label])

    # replace inf with nan
    cols = [x for x in df.columns if df[x].dtype.name == 'int64' or df[x].dtype.name == 'float64']
    for x in cols:
        df.loc[np.isfinite(df[x]) == False, x] = np.nan
        

    # print debug
    print(df.dtypes)
    if len(df) > 5:
        [print(df.iloc[x]) for x in [0,1,2,3,4]]

    # save data to outputs
    df.to_pickle((Path('outputs') / 'datasets.pkl'))
    
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
    parser.add_argument('--input-data', type=str, default='data/transformed.csv')
    parser.add_argument('--label', type=str, default='None')
    parser.add_argument('--replacements', type=str, default='None')
    parser.add_argument('--datatypes', type=str, default='None')
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