import os, argparse
import subprocess


def main(args):
    command = f'az ml job create --file {os.path.dirname(os.path.realpath(__file__))}/../../config/pipeline/train.yaml --web --set tags.project={args.project} --set inputs.datasets_pkl.path=azureml://datastores/output/paths/{args.project}/gold/ --set inputs.trainlog.path=azureml://datastores/output/paths/{args.project}/trainlog --set experiment_name={args.project} --set inputs.label={args.label}'
    print(command)
    list_files = subprocess.run(command.split(' '))
    print("The exit code was: %d" % list_files.returncode)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    
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