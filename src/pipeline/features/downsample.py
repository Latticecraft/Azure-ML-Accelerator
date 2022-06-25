# imports
import os, argparse
import pickle
import mlflow

from azureml.core import Run
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path
from shutil import copyfile


# define functions
def main(ctx):
    # read in data
    with open(ctx['args'].datasets_pkl + '/datasets.pkl', 'rb') as f:
        dict_files = pickle.load(f)
    
    if eval(ctx['args'].enable) == True:
        # perform downsample
        rus = RandomUnderSampler(sampling_strategy=ctx['args'].ratio)
        dict_files['X_train'], dict_files['y_train'] = rus.fit_resample(dict_files['X_train'], dict_files['y_train'])

    # save data to outputs
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
    parser.add_argument("--datasets-pkl", type=str, default='data')
    parser.add_argument('--enable', type=str, default='False')
    parser.add_argument('--ratio', type=float, default=0.5)
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