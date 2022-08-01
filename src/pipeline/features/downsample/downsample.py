import os, argparse
import pickle
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from imblearn.under_sampling import RandomUnderSampler


def main(ctx):
    # read in data
    with open(ctx['args'].datasets_pkl + '/datasets.pkl', 'rb') as f:
        dict_files = pickle.load(f)
    
    new_files = {
        'X_valid_none': dict_files['X_valid'],
        'y_valid_none': dict_files['y_valid'],
        'X_test_none': dict_files['X_test'],
        'y_test_none': dict_files['y_test']
    }

    df_x = dict_files['X_train']
    df_y = dict_files['y_train']

    new_files['X_train_none'] = df_x
    new_files['y_train_none'] = df_y

    if 'Undersample' in ctx['args'].balancer_mode:
        # apply under-samplers
        X_train_rus, y_train_rus = RandomUnderSampler().fit_resample(df_x, df_y)

        new_files['X_train_rus'] = X_train_rus
        new_files['y_train_rus'] = y_train_rus
        new_files['X_valid_rus'] = dict_files['X_valid']
        new_files['y_valid_rus'] = dict_files['y_valid']
        new_files['X_test_rus'] = dict_files['X_test']
        new_files['y_test_rus'] = dict_files['y_test']

    # save data to outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(new_files, f, protocol=pickle.HIGHEST_PROTOCOL)

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
    parser.add_argument('--balancer-mode', type=str, default='None')
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