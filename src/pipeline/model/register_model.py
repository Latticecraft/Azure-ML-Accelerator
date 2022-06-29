# imports
import os
import argparse
import mlflow

from azureml.core import Run
from azureml.core.model import Model
from distutils.dir_util import copy_tree


def main(ctx):
    Model.register(model_path=ctx['args'].datasets_pkl + '/model.pkl',
                    model_name=ctx['project'],
                    tags={
                        'label': ctx['label'], 
                        'type': ctx['type'],
                        'primary_metric': ctx['primary_metric'],
                        'best_score': ctx['best_score']
                    },
                    description="LightGBM model to predict response",
                    workspace=ctx['run'].experiment.workspace)

    copy_tree(ctx['args'].datasets_pkl, args.transformed_data)


def start(args):
    os.makedirs("outputs", exist_ok=True)
    mlflow.start_run()
    mlflow.autolog()
    run = Run.get_context()
    tags = run.parent.get_tags()
    return {
        'args': args,
        'run': run,
        'project': tags['project'],
        'type': tags['type'],
        'primary_metric': tags['primary_metric'],
        'label': tags['label'],
        'best_score': tags['best_score']
    }


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--datasets-pkl", type=str, default='data')
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

