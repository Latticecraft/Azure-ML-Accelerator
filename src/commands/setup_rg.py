import os, argparse
import subprocess


def main(args):
    command = f'az deployment group create --resource-group {args.resource_group} --template-file {os.path.dirname(os.path.realpath(__file__))}/../../config/resourcegroup/azuredeploy-accelerator.json'

    list_files = subprocess.run(command.split(' '))
    print('The exit code was: %d' % list_files.returncode)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--resource-group', type=str, required=True)
    
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