import os, argparse
import pickle
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, ADASYN


def main(ctx):
    # read in data
    with open(ctx['args'].datasets_pkl + '/datasets.pkl', 'rb') as f:
        dict_files = pickle.load(f)

    if ctx['args'].type != 'Regression':
        new_files = {}
        for key in dict_files.keys():
            if key.startswith('X_train'):
                df_x = dict_files[key]
                df_y = dict_files[key.replace('X','y')]

                # get categorical indices
                cat_indices = [df_x.columns.get_loc(x) for x in df_x.columns if df_x[x].dtype.name == 'category' or df_x[x].dtype.name == 'boolean']

                # apply over-samplers
                X_train_ros, y_train_ros = RandomOverSampler().fit_resample(df_x, df_y)

                if len(cat_indices) > 0:
                    X_train_smote, y_train_smote = SMOTENC(categorical_features=cat_indices).fit_resample(df_x, df_y)
                else:
                    X_train_smote, y_train_smote = SMOTE().fit_resample(df_x, df_y)

                new_files[key+'_none'] = df_x
                new_files[key.replace('X','y')+'_none'] = df_y
                new_files[key+'_ros'] = X_train_ros
                new_files[key.replace('X','y')+'_ros'] = y_train_ros
                new_files[key+'_smote'] = X_train_smote
                new_files[key.replace('X','y')+'_smote'] = y_train_smote

        # add back in test validation set
        for key in dict_files.keys():
            if key.startswith('X_valid') or key.startswith('X_test'):
                new_files['{}'.format(key)] = dict_files[key]
                new_files['{}'.format(key.replace('X','y'))] = dict_files[key.replace('X','y')]
    else:
        new_files = dict_files

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
    parser.add_argument('--type', type=str)
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
