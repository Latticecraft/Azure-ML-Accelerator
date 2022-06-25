# imports
import os, argparse
import pickle
import mlflow

from azureml.core import Run, Datastore, Dataset
from azureml.data.datapath import DataPath
from pathlib import Path


# define functions
def main(ctx):
    # read in data
    with open(args.datasets_pkl + '/datasets.pkl', 'rb') as f:
        dict_files = pickle.load(f)

    # write data out
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(dict_files, f, protocol=pickle.HIGHEST_PROTOCOL)

    # register dataset
    datastore = Datastore.get(ctx['run'].experiment.workspace, 'output')
    ds = Dataset.File.upload_directory(src_dir='outputs',
        target=DataPath(datastore, f'{ctx["project"]}/gold'),
        overwrite=True)
    ds.register(ctx['run'].experiment.workspace, f'{ctx["project"]}/gold', create_new_version=True)


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
    parser.add_argument("--datasets-pkl", type=str, default='data')

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
