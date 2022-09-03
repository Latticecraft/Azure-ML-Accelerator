import os, argparse
import mlflow

from azureml.core import Run
from azureml.core.model import Model
from distutils.dir_util import copy_tree


def main(ctx):
    Model.register(model_path=ctx['args'].datasets_pkl + '/model.pkl',
                    model_name=ctx['args'].project,
                    tags={
                        'label': ctx['args'].label,
                        'primary_metric': ctx['args'].primary_metric,
                        'type': ctx['args'].type,
                        'source': ctx['args'].source,
                        'downsample': eval(ctx['args'].downsample),
                        'drop_bools': eval(ctx['args'].drop_bools),
                        'best_score': ctx['tags']['best_score']
                    },
                    description='',
                    workspace=ctx['run'].experiment.workspace)

    copy_tree(ctx['args'].datasets_pkl, args.transformed_data)


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

    # inputs
    parser.add_argument('--datasets-pkl', type=str, default='data')
    parser.add_argument('--project', type=str, default='None')
    parser.add_argument('--label', type=str, default='None')
    parser.add_argument('--primary-metric', type=str, default='None')
    parser.add_argument('--type', type=str, default='None')
    parser.add_argument('--source', type=str, default='None')
    parser.add_argument('--downsample', type=str, default='False')
    parser.add_argument('--drop-bools', type=str, default='False')

    # outputs
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

