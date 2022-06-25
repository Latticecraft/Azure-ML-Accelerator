import os, argparse
import subprocess


def main(args):
    command = f'az ml environment create --name {args.name} --conda-file {os.path.dirname(os.path.realpath(__file__))}/../../config/environment/env.yaml --image mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04'

    list_files = subprocess.run(command.split(' '))
    print('The exit code was: %d' % list_files.returncode)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--name', type=str, required=False, default='ltcftenv')
    
    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)