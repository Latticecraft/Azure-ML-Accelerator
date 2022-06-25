# imports
import os, argparse
import pickle
import mlflow
import random

from azureml.core import Run
from pathlib import Path
from shutil import copyfile
from sklearn.model_selection import train_test_split

# define functions
def main(ctx):
    # read in data
    with open(ctx['args'].datasets_pkl + '/datasets.pkl', 'rb') as f:
        dict_files = pickle.load(f)

    # perform split
    X_train, X_test, y_train, y_test = train_test_split(dict_files['X'], dict_files['y'], test_size=0.33, random_state=random.randint(0,1000000))
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=random.randint(0,1000000))

    # create new dictionary
    new_files = {}
    new_files['X_train'] = X_train
    new_files['X_valid'] = X_valid
    new_files['X_test'] = X_test
    new_files['y_train'] = y_train
    new_files['y_valid'] = y_valid
    new_files['y_test'] = y_test

    # log metrics
    dict_train = {
        f'y_train_{k}': v for k,v in y_train.iloc[:,0].value_counts().to_dict().items()
    }

    dict_valid = {
        f'y_valid_{k}': v for k,v in y_valid.iloc[:,0].value_counts().to_dict().items()
    }

    dict_test = {
        f'y_test_{k}': v for k,v in y_test.iloc[:,0].value_counts().to_dict().items()
    }

    dict_counts = {**dict_train, **dict_valid, **dict_test}
    for k in dict_counts.keys():
        mlflow.log_metric(k, dict_counts[k])

    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(new_files, f, protocol=pickle.HIGHEST_PROTOCOL)

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
