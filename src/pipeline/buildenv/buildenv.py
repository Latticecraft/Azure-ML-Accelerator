import os, argparse

from azureml.core import Environment, Run


def main(args):
    run = Run.get_context()
    env = Environment(name='batch_training')
    build = env.build(run.experiment.workspace)
    build.wait_for_completion()


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments


    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == '__main__':
    # parse args
    args = parse_args()

    # run main function
    main(args)