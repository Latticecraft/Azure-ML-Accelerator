import os, argparse
import subprocess


def main(args):
    command = f'az ml job create --file {os.path.dirname(os.path.realpath(__file__))}/../../config/pipeline/featurize.yaml --web --set tags.project={args.project} --set tags.type={args.type} --set inputs.input_csv.path=azureml://datastores/input/paths/{args.project}/{args.input} --set inputs.runinfo.path=azureml://datastores/output/paths/{args.project}/runinfo --set inputs.trainlog.path=azureml://datastores/output/paths/{args.project}/trainlog --set experiment_name={args.project} --set inputs.label={args.label} --set inputs.unwanted={args.unwanted} --set inputs.replacements={args.replacements} --set inputs.datatypes={args.datatypes} --set inputs.separator={args.separator} --set inputs.web_hook="{args.web_hook}" --set inputs.next_pipeline={args.next_pipeline}'

    list_files = subprocess.run(command.split(' '))
    print('The exit code was: %d' % list_files.returncode)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--type', type=str, required=False)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--replacements', type=str, required=False)
    parser.add_argument('--datatypes', type=str, required=False)
    parser.add_argument('--separator', type=str, required=False)
    parser.add_argument('--unwanted', type=str, required=False)
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