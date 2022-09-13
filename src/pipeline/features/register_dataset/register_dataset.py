import os, argparse
import numpy as np
import pandas as pd
import pickle
import mlflow

from azureml.core import Run, Datastore, Dataset
from azureml.data.datapath import DataPath
from distutils.dir_util import copy_tree


def main(ctx):
    # read in data
    with open(args.datasets_pkl + '/datasets.pkl', 'rb') as f:
        dict_files = pickle.load(f)

    # write data out
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(dict_files, f, protocol=pickle.HIGHEST_PROTOCOL)

    copy_tree('outputs', args.transformed_data)

    # get imputers/balancers for tagging
    df_files = pd.DataFrame(list(pd.Series(list(dict_files.keys())).str.split('_')),
        columns=['type', 'fold', 'col1', 'col2', 'imputer', 'balancer'])

    imputers = ','.join([x for x in np.unique(df_files['imputer'].dropna()) if x != ''])
    balancers = ','.join([x for x in np.unique(df_files['balancer'].dropna()) if x != ''])

    # register dataset
    variant = ''
    variant = variant + '-downsample' if eval(ctx['args'].downsample) == True else variant
    variant = variant + '-dropbools' if eval(ctx['args'].drop_bools) == True else variant

    datastore = Datastore.get(ctx['run'].experiment.workspace, 'output')
    
    ds = Dataset.File.upload_directory(src_dir='outputs',
        target=DataPath(datastore, f'{ctx["args"].project}/gold{variant}'),
        overwrite=True)

    ds.register(ctx['run'].experiment.workspace, 
        f'{ctx["args"].project}_gold{variant}', 
        create_new_version=True, 
        tags={
            'imputers': imputers, 
            'balancers': balancers, 
            'downsample': ctx['args'].downsample, 
            'drop_bools': ctx['args'].drop_bools
        })


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
    parser.add_argument('--project', type=str, default='None')
    parser.add_argument('--downsample', type=str, default='False')
    parser.add_argument('--drop-bools', type=str, default='False')
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
