import os, argparse

from azureml.core import Environment, Run


def main(args):
    run = Run.get_context()
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_DS3_v2', max_nodes=1)
    cpu_cluster = ComputeTarget.create(run.experiment.workspace, 'cpu-cluster', compute_config)
    cpu_cluster.wait_for_completion(show_output=True)


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