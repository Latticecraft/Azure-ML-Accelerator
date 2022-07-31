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

    if ctx['args'].balancer_mode == 'Oversample':
        new_files = {}
        for key in dict_files.keys():
            if key.startswith('X_train'):
                df_x = dict_files[key]
                df_y = dict_files[key.replace('X','y')]

                # get categorical indices
                cat_indices = [df_x.columns.get_loc(x) for x in df_x.columns if df_x[x].dtype.name == 'category' or df_x[x].dtype.name == 'boolean']

                # do nothing
                new_files[get_key(key, 'X', 'train', 'none')] = df_x
                new_files[get_key(key, 'y', 'train', 'none')] = df_y
                copy_valid_test(new_files, dict_files, key, 'none')

                # apply over-samplers
                X_train_ros, y_train_ros = RandomOverSampler().fit_resample(df_x, df_y)
                new_files[get_key(key, 'X', 'train', 'ros')] = X_train_ros
                new_files[get_key(key, 'y', 'train', 'ros')] = y_train_ros
                copy_valid_test(new_files, dict_files, key, 'ros')

                if len(cat_indices) > 0:
                    print('Categorical and/or boolean features found, using SMOTENC')
                    try:
                        X_train_smote, y_train_smote = SMOTENC(categorical_features=cat_indices).fit_resample(df_x, df_y)
                        new_files[get_key(key, 'X', 'train', 'smote')] = X_train_smote
                        new_files[get_key(key, 'y', 'train', 'smote')] = y_train_smote
                        copy_valid_test(new_files, dict_files, key, 'smote')
                    except:
                        print('Error running SMOTE')
                else:
                    print('Categorical and/or boolean features NOT found, using SMOTE/ADASYN')
                    try:
                        X_train_smote, y_train_smote = SMOTE().fit_resample(df_x, df_y)
                        new_files[get_key(key, 'X', 'train', 'smote')] = X_train_smote
                        new_files[get_key(key, 'y', 'train', 'smote')] = y_train_smote
                        copy_valid_test(new_files, dict_files, key, 'smote')
                    except:
                        print('Error running SMOTE')

                    try:
                        X_train_adasyn, y_train_adasyn = ADASYN().fit_resample(df_x, df_y)
                        new_files[get_key(key, 'X', 'train', 'adasyn')] = X_train_adasyn
                        new_files[get_key(key, 'y', 'train', 'adasyn')] = y_train_adasyn
                        copy_valid_test(new_files, dict_files, key, 'adasyn')
                    except:
                        print('Error running ADASYN')
    else:
        new_files = dict_files

    # save data to outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(new_files, f, protocol=pickle.HIGHEST_PROTOCOL)

    copy_tree('outputs', args.transformed_data)


def copy_valid_test(dict_new, dict_old, key, balancer):
    dict_new[get_key(key, 'X', 'valid', balancer)] = dict_old[get_key(key, 'X', 'valid', None)]
    dict_new[get_key(key, 'y', 'valid', balancer)] = dict_old[get_key(key, 'y', 'valid', None)]
    dict_new[get_key(key, 'X', 'test', balancer)] = dict_old[get_key(key, 'X', 'test', None)]
    dict_new[get_key(key, 'y', 'test', balancer)] = dict_old[get_key(key, 'y', 'test', None)]


def get_key(key, type, fold, balancer):
    arr = key.split('_')
    if balancer != None:
        if len(arr) == 3:
            return f'{type}_{fold}_{arr[2]}_{balancer}'
        else:
            raise Exception('Unknown filename format')
    else:
        if len(arr) == 3:
            return f'{type}_{fold}_{arr[2]}'
        else:
            raise Exception('Unknown filename format')


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
