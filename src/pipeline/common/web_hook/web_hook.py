# imports
import os, argparse
import mlflow
import requests

from azureml.core import Run


def main(ctx):
    if ctx['args'].url != 'None':
        r = requests.post(f'{ctx["args"].url}&definitionId={ctx["args"].param1}', json={})
        if r.status_code != 200:
            raise ValueError('Unable to trigger next DevOps pipeline')


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
        'data': client.get_run(mlflow.active_run().info.run_id).data
    }


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--datasets-pkl', type=str)
    parser.add_argument('--url', type=str)
    parser.add_argument('--param1', type=str)

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
