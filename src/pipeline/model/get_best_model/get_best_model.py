import sys, os, argparse
import json
import mlflow
import numpy as np
import pickle

from azureml.core import Run
from distutils.dir_util import copy_tree
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from lazy_eval import LazyEval


def main(ctx):
    # get run ids of sweep job and best run
    best_runid, best_score, sweepid = '', 0, ''
    with open(Path(ctx['args'].train_artifacts)/'best_run.json', 'r') as f:
        data = json.load(f)
        best_runid = data['runId']
        best_score = data['best_score']
        sweepid = data['sweepId']
        label = data['label']
        imputer = data['imputer']
        balancer = data['balancer']

    # transform data based on imputer/balancer selected
    with open(ctx['args'].datasets_pkl + '/datasets.pkl', 'rb') as handle:
        dict_files = pickle.load(handle)
        data = LazyEval(dict_files)

    X_train, y_train = data.get('train', imputer, balancer)
    X_valid, y_valid = data.get('valid', imputer, balancer)
    X_test, y_test = data.get('test', imputer, balancer)

    dict_new = {
        'X_train': X_train,
        'y_train': y_train,
        'X_valid': X_valid,
        'y_valid': y_valid,
        'X_test': X_test,
        'y_test': y_test
    }

    with open('outputs/datasets.pkl', 'wb') as handle:
        pickle.dump(dict_new, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # download best run artifacts to outputs
    best_run = Run.get(workspace=ctx['run'].experiment.workspace, run_id=best_runid)
    best_run.download_file('outputs/model.pkl', output_file_path='outputs')
    best_run.download_file('outputs/features_ranked.json', output_file_path='outputs')

    # get metrics and tags from sweep job
    sweep_run = Run.get(ctx['run'].experiment.workspace, sweepid)
    metrics = sweep_run.get_metrics(recursive=True)
    tags = sweep_run.get_tags()

    # create sweep dictionary
    dict_new = {
        'sweep_get_best_model_runid': best_runid,
        'sweep_run_ids': list(metrics.keys()),
        'sweep_primary_metric': [metrics[k][x] for k in metrics.keys() for x in metrics[k] if x == ctx['args'].primary_metric],
        'sweep_balancer': [json.loads(tags[k])['balancer'] for k in metrics.keys() if k in tags.keys()],
        'sweep_imputer': [json.loads(tags[k])['imputer'] for k in metrics.keys() if k in tags.keys()]
    }

    dict_new['num_balancer'] = len(np.unique(dict_new['sweep_balancer']))
    dict_new['num_imputer'] = len(np.unique(dict_new['sweep_imputer']))

    print(f'dict_new: {dict_new}')

    # log best run metric to parent pipeline
    ctx['run'].parent.set_tags({'best_score': best_score, 'label': label})
    
    # get metrics of best run
    for k in metrics[best_runid].keys():
        if type(metrics[best_runid][k]) is not list:
            dict_new[k] = metrics[best_runid][k]

    # get ranked features
    with open('outputs/features_ranked.json', 'r') as f:
        features_ranked = json.load(f)

    for k,v in features_ranked.items():
        dict_new[k] = features_ranked[k]

    # copy train artifacts to outputs
    copy_tree(ctx['args'].train_artifacts, 'outputs')

    # copy outputs
    copy_tree('outputs', args.transformed_data)

    # save additional metrics to outputs
    with open('outputs/metrics.json', 'w') as f:
        json.dump(dict_new, f)


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
    parser.add_argument('--sweep-name', type=str, default='None')
    parser.add_argument('--train-artifacts', type=str, default='data')
    parser.add_argument('--primary-metric', type=str, default='None')
    parser.add_argument('--transformed-data', type=str)

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
