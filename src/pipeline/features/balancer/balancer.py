import os, argparse
import pickle
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, ADASYN
from sklearn.pipeline import Pipeline


def main(ctx):
    # read in data
    with open(ctx['args'].datasets_pkl + '/datasets.pkl', 'rb') as f:
        dict_orig = pickle.load(f)

    #dict_new = {}
    keys = [x for x in dict_orig.keys()]
    for key in keys:
        if key.startswith('imputer'):
            arr = key.split('_')
            if arr[5] != 'rus':
                df_x = dict_orig['X_train_none']

                # get categorical indices
                cat_indices = [df_x.columns.get_loc(x) for x in df_x.columns if df_x[x].dtype.name == 'category' or df_x[x].dtype.name == 'boolean']

                df_x = dict_orig[f'imputer____{arr[4]}_none'].fit_transform(df_x)
                df_y = dict_orig['y_train_none']

                

                # apply over-samplers
                balancer = RandomOverSampler()
                balancer.fit_resample(df_x, df_y)
                dict_orig[f'balancer____{arr[4]}_ros'] = balancer
                #dict_new[get_key(key, 'X', 'train', 'ros')] = X_train_ros
                #dict_new[get_key(key, 'y', 'train', 'ros')] = y_train_ros
                #copy_valid_test(dict_new, dict_orig, key, 'ros')

                if len(cat_indices) > 0:
                    print('Categorical and/or boolean features found, using SMOTENC')
                    try:
                        balancer = SMOTENC(categorical_features=cat_indices)
                        balancer.fit_resample(df_x, df_y)
                        dict_orig[f'balancer____{arr[4]}_smote'] = balancer
                        #dict_new[get_key(key, 'X', 'train', 'smote')] = X_train_smote
                        #dict_new[get_key(key, 'y', 'train', 'smote')] = y_train_smote
                        #copy_valid_test(dict_new, dict_orig, key, 'smote')
                    except:
                        print('Error running SMOTE')
                else:
                    print('Categorical and/or boolean features NOT found, using SMOTE/ADASYN')
                    try:
                        balancer = SMOTE()
                        balancer.fit_resample(df_x, df_y)
                        dict_orig[f'balancer____{arr[4]}_smote'] = balancer
                        #dict_new[get_key(key, 'X', 'train', 'smote')] = X_train_smote
                        #dict_new[get_key(key, 'y', 'train', 'smote')] = y_train_smote
                        #copy_valid_test(dict_new, dict_orig, key, 'smote')
                    except:
                        print('Error running SMOTE')

                    try:
                        balancer = ADASYN()
                        balancer.fit_resample(df_x, df_y)
                        dict_orig[f'balancer____{arr[4]}_adasyn'] = balancer
                        #dict_new[get_key(key, 'X', 'train', 'adasyn')] = X_train_adasyn
                        #dict_new[get_key(key, 'y', 'train', 'adasyn')] = y_train_adasyn
                        #copy_valid_test(dict_new, dict_orig, key, 'adasyn')
                    except:
                        print('Error running ADASYN')

        #elif key.startswith('X_train') and '_rus' in key:
        #    dict_new[get_key(key, 'X', 'train', 'rus')] = dict_orig[get_key(key, 'X', 'train', 'rus')]
        #    dict_new[get_key(key, 'y', 'train', 'rus')] = dict_orig[get_key(key, 'y', 'train', 'rus')]
        #    dict_new[get_key(key, 'X', 'valid', 'rus')] = dict_orig[get_key(key, 'X', 'valid', 'rus')]
        #    dict_new[get_key(key, 'y', 'valid', 'rus')] = dict_orig[get_key(key, 'y', 'valid', 'rus')]
        #    dict_new[get_key(key, 'X', 'test', 'rus')] = dict_orig[get_key(key, 'X', 'test', 'rus')]
        #    dict_new[get_key(key, 'y', 'test', 'rus')] = dict_orig[get_key(key, 'y', 'test', 'rus')]

        elif key.startswith('imputer') or key.startswith('outliers'):
            dict_orig[key] = dict_orig[key]

    # save data to outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(dict_orig, f, protocol=pickle.HIGHEST_PROTOCOL)

    copy_tree('outputs', args.transformed_data)


def copy_valid_test(dict_new, dict_orig, key, balancer):
    dict_new[get_key(key, 'X', 'valid', balancer)] = dict_orig[get_key(key, 'X', 'valid')]
    dict_new[get_key(key, 'y', 'valid', balancer)] = dict_orig[get_key(key, 'y', 'valid')]
    dict_new[get_key(key, 'X', 'test', balancer)] = dict_orig[get_key(key, 'X', 'test')]
    dict_new[get_key(key, 'y', 'test', balancer)] = dict_orig[get_key(key, 'y', 'test')]


def get_key(key, type, fold, balancer='none'):
    arr = key.split('_')
    return f'{type}_{fold}_{arr[2]}_{balancer}'


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
