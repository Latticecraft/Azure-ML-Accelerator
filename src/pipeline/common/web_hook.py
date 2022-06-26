# imports
import os, argparse
import mlflow
import requests

from dateutil import parser
from datetime import datetime, timedelta
from pathlib import Path
from azureml.core import Dataset, Datastore, Run
from azureml.data.datapath import DataPath


def main(ctx):
    if ctx['args'].url is not None:
        r = requests.post(f'{ctx["args"].url}&definitionId={ctx["args"].next_pipeline}', json={})
        if r.status_code != 200:
            raise ValueError('Unable to trigger next DevOps pipeline')


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
    parser.add_argument('--datasets-pkl', type=str)
    parser.add_argument('--url', type=str)
    parser.add_argument('--next-pipeline', type=str)

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
