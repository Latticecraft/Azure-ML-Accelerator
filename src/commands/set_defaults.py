import os, argparse
import subprocess


def main(args):
    command = f'az configure --defaults group={args.group} workspace={args.workspace} location={args.location}'

    list_files = subprocess.run(command.split(' '))
    print('The exit code was: %d' % list_files.returncode)


def prompt_input(args, name, prompt):
    if getattr(args, name) is None:
        setattr(args, name, input(prompt))
        if getattr(args, name) is None:
            exit(1)

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--group', type=str, required=False)
    parser.add_argument('--workspace', type=str, required=False)
    parser.add_argument('--location', type=str, required=False)
    
    # parse args
    args = parser.parse_args()

    prompt_input(args, 'group', 'Enter Resource Group: ')
    prompt_input(args, 'workspace', 'Enter Workspace Name: ')
    prompt_input(args, 'location', 'Enter Location (ex. westus3): ')
        
    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)