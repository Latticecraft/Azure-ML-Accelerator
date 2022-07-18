import os, argparse
import json
import mlflow
import numpy as np

from azureml.core import Run
from distutils.dir_util import copy_tree
from pathlib import Path


def main(ctx):
    # get run ids of sweep job and best run
    best_runid, best_score, sweepid = '', 0, ''
    with open(Path(ctx['args'].train_artifacts)/'best_run.json', 'r') as f:
        data = json.load(f)
        best_runid = data['runId']
        best_score = data['best_score']
        sweepid = data['sweepId']
        label = data['label']
    
    # download best run artifacts to outputs
    best_run = Run.get(workspace=ctx['run'].experiment.workspace, run_id=best_runid)
    best_run.download_file('outputs/model.pkl', output_file_path='outputs')
    best_run.download_file('outputs/datasets.pkl', output_file_path='outputs')
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
