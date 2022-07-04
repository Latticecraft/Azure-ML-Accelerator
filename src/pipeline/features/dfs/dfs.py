# imports
import os, argparse
import pickle
import featuretools as ft
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from featuretools.primitives import IsNull, Weekday


# define functions
def main(ctx):
    # read in data
    with open(ctx['args'].datasets_pkl + '/datasets.pkl', 'rb') as f:
        dict_files = pickle.load(f)

    dict_predfs = {}
    df_train = dict_files['X_train']
    df_valid = dict_files['X_valid']
    df_test = dict_files['X_test']
    dict_predfs['X_train'] = df_train

    dataframes_train = {
        "customers" : (df_train, "ID")
    }

    feature_matrix_train, feature_defs = ft.dfs(dataframes=dataframes_train, relationships=[], target_dataframe_name='customers', trans_primitives=[IsNull, Weekday])
    feature_matrix_train, feature_defs = ft.encode_features(feature_matrix_train, feature_defs)
    feature_matrix_train, feature_defs = ft.selection.remove_low_information_features(feature_matrix_train, feature_defs)
    feature_matrix_train, feature_defs = ft.selection.remove_highly_correlated_features(feature_matrix_train, feature_defs)
    feature_matrix_train, feature_defs = ft.selection.remove_highly_null_features(feature_matrix_train, feature_defs)
    feature_matrix_train, feature_defs = ft.selection.remove_single_value_features(feature_matrix_train, feature_defs)

    dataframes_valid = {
        'customers' : (df_valid, 'ID')
    }

    feature_matrix_valid = ft.calculate_feature_matrix(dataframes=dataframes_valid, features=feature_defs)

    dataframes_test = {
        'customers' : (df_test, 'ID')
    }

    feature_matrix_test = ft.calculate_feature_matrix(dataframes=dataframes_test, features=feature_defs)

    # ensure columns are in same order as train
    feature_matrix_valid = feature_matrix_valid[list(feature_matrix_train.columns)]
    feature_matrix_test = feature_matrix_test[list(feature_matrix_train.columns)]

    # add dictionary files
    dict_files['X_train'] = feature_matrix_train
    dict_files['X_valid'] = feature_matrix_valid
    dict_files['X_test'] = feature_matrix_test

    # log metrics
    df = dict_predfs['X_train']
    dict_dtypes = df.dtypes.value_counts().to_dict()
    for (k,v) in dict_dtypes.items():
        mlflow.log_metric(f'dtypes.predfs.{k}', v)

    df = dict_files['X_train']
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