import os, argparse
import subprocess


def main(args):
    command = f'az ml job create --file {os.path.dirname(os.path.realpath(__file__))}/../../config/pipeline/train.yaml --web --set tags.project={args.project} --set tags.type={args.type} --set inputs.primary_metric={args.primary_metric} --set inputs.datasets_pkl.path=azureml://datastores/output/paths/{args.project}/gold/ --set inputs.trainlog.path=azureml://datastores/output/paths/{args.project}/trainlog --set experiment_name={args.project} --set inputs.label={args.label} --set inputs.web_hook="{args.web_hook}" --set inputs.next_pipeline={args.next_pipeline}'
    print(command)
    
    list_files = subprocess.run(command.split(' '))
    print("The exit code was: %d" % list_files.returncode)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--type', type=str, required=False)
    parser.add_argument('--primary-metric', type=str, required=False)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--web-hook', type=str, required=False)
    parser.add_argument('--next-pipeline', type=int, required=False)
    
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