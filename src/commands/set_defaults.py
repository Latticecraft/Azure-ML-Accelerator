import os, argparse
import subprocess


def main(args):
    command = f'az configure --defaults group={args.group} workspace={args.workspace} location={args.location}'

    list_files = subprocess.run(command.split(' '))
    print('The exit code was: %d' % list_files.returncode)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--group', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    parser.add_argument('--location', type=str, required=True)
    
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