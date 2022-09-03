# imports
import os, argparse
import pandas as pd
import mlflow
import json

from azureml.core import Dataset, Datastore, Run
from azureml.data.datapath import DataPath
from dateutil import parser
from datetime import datetime, timedelta
from pathlib import Path


def main(ctx):
    # initialize metrics
    jm = {}
    jm['experiment'] = ctx['run'].experiment.name
    jm['runId'] = ctx['data'].tags['mlflow.parentRunId']
    jm['runDate'] = datetime.utcnow()
    jm['downsample'] = ctx['args'].downsample
    jm['drop_bools'] = ctx['args'].drop_bools

    # iterate across sibling jobs
    parent_run = Run.get(ctx['run'].experiment.workspace, ctx['data'].tags['mlflow.parentRunId'])
    child_runs = parent_run.get_children()
    for cr in child_runs:
        if cr.get_status() == 'Completed':
            properties = cr.get_properties()
            details = cr.get_details()

            # log is-reused
            jm[f'JOB_{cr.display_name}.isreused'] = properties['azureml.isreused'] if 'azureml.isreused' in properties else False

            # log time data
            duration = round((parser.parse(details['endTimeUtc']) - parser.parse(details['startTimeUtc'])) / timedelta(minutes=1), 2)
            jm[f'JOB_{cr.display_name}.startTime'] = details['startTimeUtc']
            jm[f'JOB_{cr.display_name}.duration'] = duration

            # log metrics
            metrics = cr.get_metrics()
            for k in metrics.keys():
                jm[k] = metrics[k]

            # merge metrics.json file if it exists
            try:
                metrics = json.loads(cr._download_artifact_contents_to_string('outputs/metrics.json'))
                for k in metrics.keys():
                    jm[k] = metrics[k]
                
                print(f'{cr.name}: found metrics.json file, merging')
            except:
                print(f'{cr.name}: no metrics.json file found')

    # write metrics to output
    df_metrics = pd.DataFrame.from_dict(jm, orient='index').T
    df_metrics.to_csv('outputs/transformed.csv', index=False)
    df_metrics.to_csv((Path(ctx['args'].transformed_data) / '{}.csv'.format(ctx['data'].tags['mlflow.parentRunId'])), index=False)

    # write metrics to central repo for project
    datastore = Datastore.get(ctx['run'].experiment.workspace, 'output')
    ds = Dataset.File.upload_directory(src_dir=args.transformed_data,
        target=DataPath(datastore, f'{ctx["args"].project}/{args.destination_folder}'),
        overwrite=True)


def start(args):
    os.makedirs('outputs', exist_ok=True)
    mlflow.start_run()
    mlflow.autolog()
    run = Run.get_context()
    tags = run.parent.get_tags()
    client = mlflow.tracking.MlflowClient()
    return {
        'args': args,
        'run': run,
        'tags': tags,
        'data': client.get_run(mlflow.active_run().info.run_id).data
    }


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--datasets-pkl', type=str, default='data')
    parser.add_argument('--project', type=str)
    parser.add_argument('--destination-folder', type=str)
    parser.add_argument('--downsample', type=str, default='False')
    parser.add_argument('--drop-bools', type=str, default='False')

    parser.add_argument('--transformed-data', type=str, help='Path of output data')

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
