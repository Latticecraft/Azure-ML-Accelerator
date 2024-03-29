import os, argparse
import pickle
import featuretools as ft
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from featuretools.primitives import IsNull, Weekday


def main(ctx):
    # read in data
    with open(ctx['args'].datasets_pkl + '/datasets.pkl', 'rb') as f:
        dict_files = pickle.load(f)

    dict_predfs = {}
    for key in dict_files.keys():
        if 'X_train' in key:
            df_train = dict_files[key]
            df_valid = dict_files[key.replace('train', 'valid')]
            df_test = dict_files[key.replace('train', 'test')]
            dict_predfs[key] = df_train

            dataframes_train = {
                'main': (df_train, 'ID')
            }

            feature_matrix_train, feature_defs = ft.dfs(dataframes=dataframes_train, relationships=[], target_dataframe_name='main', trans_primitives=[IsNull, Weekday])
            feature_matrix_train, feature_defs = ft.encode_features(feature_matrix_train, feature_defs)

            if eval(ctx['args'].remove_low_info) == True:
                print('removing low information features...')
                feature_matrix_train, feature_defs = ft.selection.remove_low_information_features(feature_matrix_train, feature_defs)
            
            if eval(ctx['args'].remove_high_corr) == True:
                print('removing highly correlated features...')
                feature_matrix_train, feature_defs = ft.selection.remove_highly_correlated_features(feature_matrix_train, feature_defs)

            if eval(ctx['args'].remove_high_nan) == True:
                print('removing highly null features...')
                feature_matrix_train, feature_defs = ft.selection.remove_highly_null_features(feature_matrix_train, feature_defs)
            
            if eval(ctx['args'].remove_single_val) == True:
                print('removing single value features...')
                feature_matrix_train, feature_defs = ft.selection.remove_single_value_features(feature_matrix_train, feature_defs)

            # apply transformations to validation set
            dataframes_valid = {
                'main': (df_valid, 'ID')
            }

            feature_matrix_valid = ft.calculate_feature_matrix(dataframes=dataframes_valid, features=feature_defs)

            # HACK: for some reason boolean is used instead of bool which makes downstream training fail... convert boolean to bool
            bool_cols1 = {x: 'int' for x in feature_matrix_valid.columns if feature_matrix_valid[x].dtype.name == 'boolean'}
            bool_cols2 = {x: 'bool' for x in feature_matrix_valid.columns if feature_matrix_valid[x].dtype.name == 'boolean'}
            feature_matrix_valid = feature_matrix_valid.astype(bool_cols1)
            feature_matrix_valid = feature_matrix_valid.astype(bool_cols2)

            dataframes_test = {
                'main': (df_test, 'ID')
            }

            # apply transformations to test set
            feature_matrix_test = ft.calculate_feature_matrix(dataframes=dataframes_test, features=feature_defs)

            bool_cols1 = {x: 'int' for x in feature_matrix_test.columns if feature_matrix_test[x].dtype.name == 'boolean'}
            bool_cols2 = {x: 'bool' for x in feature_matrix_test.columns if feature_matrix_test[x].dtype.name == 'boolean'}
            feature_matrix_test = feature_matrix_test.astype(bool_cols1)
            feature_matrix_test = feature_matrix_test.astype(bool_cols2)

            # ensure columns are in same order as train
            feature_matrix_valid = feature_matrix_valid[list(feature_matrix_train.columns)]
            feature_matrix_test = feature_matrix_test[list(feature_matrix_train.columns)]

            # add dictionary files
            dict_files[key] = feature_matrix_train
            dict_files[key.replace('train', 'valid')] = feature_matrix_valid
            dict_files[key.replace('train', 'test')] = feature_matrix_test

    # log metrics
    key = 'X_train' if 'X_train' in dict_predfs.keys() else 'X_train_none'
    df = dict_predfs[key]
    dict_dtypes = df.dtypes.value_counts().to_dict()
    for (k,v) in dict_dtypes.items():
        mlflow.log_metric(f'dtypes.predfs.{k}', v)

    df = dict_files[key]
    dict_dtypes = df.dtypes.value_counts().to_dict()
    for (k,v) in dict_dtypes.items():
        mlflow.log_metric(f'dtypes.postdfs.{k}', v)

    # save data to outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(dict_files, f, protocol=pickle.HIGHEST_PROTOCOL)

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

    # input arguments
    parser.add_argument('--datasets-pkl', type=str, default='data')
    parser.add_argument('--remove-low-info', type=str, default='True')
    parser.add_argument('--remove-high-corr', type=str, default='True')
    parser.add_argument('--remove-high-nan', type=str, default='True')
    parser.add_argument('--remove-single-val', type=str, default='True')

    # output arguments
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