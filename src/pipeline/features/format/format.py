import os, argparse, json
import numpy as np
import pandas as pd
import urllib.parse
import mlflow
import re

from azureml.core import Run
from distutils.dir_util import copy_tree
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def main(ctx):
    # read in data
    df = pd.read_pickle(ctx['args'].input_data)

    # remove spaces from header    
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', x) for x in df.columns]

    # set datatypes
    if ctx['args'].datatypes != 'None':
        datatypes = json.loads(urllib.parse.unquote(ctx['args'].datatypes))
        for k,v in datatypes.items():
            df[k] = df[k].astype(v)

    # replace inf with nan
    if eval(ctx['args'].convert_inf) == True:
        cols = [x for x in df.columns if 'int' in df[x].dtype.name or 'float' in df[x].dtype.name]
        for x in cols:
            df.loc[np.isfinite(df[x]) == False, x] = np.nan

    # set index
    if ctx['args'].index != 'None':
        df.set_index(ctx['args'].index)

    # process label
    if ctx['args'].label != 'None':
        # encode label
        le = LabelEncoder()
        df[ctx['args'].label] = le.fit_transform(df[ctx['args'].label])

        # if classification, ensure label is int
        if ctx['args'].type != 'Regression':
            df[ctx['args'].label] = df[ctx['args'].label].astype(int)

    # perform any replacements specified
    if ctx['args'].replacements != 'None':
        replacements = json.loads(urllib.parse.unquote(ctx['args'].replacements))
        df = df.replace(replacements)

    # remove bools if configured
    if eval(ctx['args'].drop_bools) == True:
        print('dropping bools...')
        to_drop = []
        for i in np.arange(len(df.columns)):
            if 'int' not in str(df.dtypes[i]) and 'float' not in str(df.dtypes[i]):
                to_drop.append(df.columns[i])

        for col in to_drop:
            del df[col]

    # remove commas from strings (can interfere with DFS)
    if eval(ctx['args'].remove_commas) == True:
        df = df.replace(',', ' ', regex=True)

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

    # inputs
    parser.add_argument('--input-data', type=str, default='data/datasets.pkl')
    parser.add_argument('--convert-inf', type=str, default='True')
    parser.add_argument('--datatypes', type=str, default='None')
    parser.add_argument('--drop-bools', type=str, default='False')
    parser.add_argument('--index', type=str, default='None')
    parser.add_argument('--label', type=str, default='None')
    parser.add_argument('--remove-commas', type=str, default='True')
    parser.add_argument('--replacements', type=str, default='None')
    parser.add_argument('--type', type=str, default='None')

    # outputs
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