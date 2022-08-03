import os, argparse
import pickle
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from imblearn.under_sampling import RandomUnderSampler


def main(ctx):
    # read in data
    with open(ctx['args'].datasets_pkl + '/datasets.pkl', 'rb') as f:
        dict_orig = pickle.load(f)
    
    dict_new = {
        'X_train_none': dict_orig['X_train'],
        'y_train_none': dict_orig['y_train'],
        'X_valid_none': dict_orig['X_valid'],
        'y_valid_none': dict_orig['y_valid'],
        'X_test_none': dict_orig['X_test'],
        'y_test_none': dict_orig['y_test']
    }

    # apply random undersampler
    X_train_rus, y_train_rus = RandomUnderSampler().fit_resample(dict_new['X_train_none'], dict_new['y_train_none'])

    dict_new['X_train_rus'] = X_train_rus
    dict_new['y_train_rus'] = y_train_rus
    dict_new['X_valid_rus'] = dict_orig['X_valid']
    dict_new['y_valid_rus'] = dict_orig['y_valid']
    dict_new['X_test_rus'] = dict_orig['X_test']
    dict_new['y_test_rus'] = dict_orig['y_test']

    # save data to outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(dict_new, f, protocol=pickle.HIGHEST_PROTOCOL)

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
    parser.add_argument('--datasets-pkl', type=str, default='data')
    parser.add_argument('--transformed_data', type=str, help='Path of output data')

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