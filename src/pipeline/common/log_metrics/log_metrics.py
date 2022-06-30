# imports
import os
import argparse
import numpy as np
import pandas as pd
import mlflow
import json

from dateutil import parser
from datetime import datetime, timedelta
from pathlib import Path
from azureml.core import Dataset, Datastore, Run
from azureml.data.datapath import DataPath


def main(ctx):
    # log metrics
    jm = {}
    jm['experiment'] = ctx['run'].experiment.name
    jm['projectname'] = ctx['project']
    jm['runId'] = ctx['data'].tags['mlflow.parentRunId']
    jm['runDate'] = datetime.utcnow()

    # log time data
    parent_run = Run.get(ctx['run'].experiment.workspace, ctx['data'].tags['mlflow.parentRunId'])

    child_runs = parent_run.get_children()
    for cr in child_runs:
        print(f'processing {cr.display_name}')
        if cr.get_status() == 'Completed':
            properties = cr.get_properties()
            jm[f'JOB_{cr.display_name}.isreused'] = properties['azureml.isreused'] if 'azureml.isreused' in properties else False

            details = cr.get_details()
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

    df_metrics = pd.DataFrame.from_dict(jm, orient='index').T
    df_metrics.to_csv('outputs/transformed.csv', index=False)
    df_metrics.to_csv((Path(ctx['args'].transformed_data) / '{}.csv'.format(ctx['data'].tags['mlflow.parentRunId'])), index=False)

    datastore = Datastore.get(ctx['run'].experiment.workspace, 'output')
    ds = Dataset.File.upload_directory(src_dir=args.transformed_data,
        target=DataPath(datastore, f'{ctx["project"]}/{args.destination_folder}'),
        overwrite=True)

    if args.destination_folder == 'runinfo':
        os.makedirs("temp", exist_ok=True)
        df_trainlog = pd.DataFrame({'runDate':[], 'get_best_model_runid':[]})
        df_trainlog.to_csv('temp/history.csv', index=False)

        ds = Dataset.File.upload_directory(src_dir='temp',
            target=DataPath(datastore, f'{ctx["project"]}/trainlog'),
            overwrite=False)


def start(args):
    os.makedirs("outputs", exist_ok=True)
    mlflow.start_run()
    mlflow.autolog()
    run = Run.get_context()
    tags = run.parent.get_tags()
    client = mlflow.tracking.MlflowClient()
    return {
        'args': args,
        'run': run,
        'project': tags['project'],
        'data': client.get_run(mlflow.active_run().info.run_id).data
    }


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--datasets-pkl", type=str, default='data')
    parser.add_argument('--destination-folder', type=str)
    parser.add_argument("--transformed-data", type=str, help="Path of output data")

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()
    ctx = start(args)

    # run main function
    main(ctx)
