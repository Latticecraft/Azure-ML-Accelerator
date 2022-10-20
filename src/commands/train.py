import os, argparse
import subprocess
import yaml

from pathlib import PurePath


def main(args):
    # reason for dynamic generation of yaml is b/c afaik oob input binding doesn't yet
    # support things like array and integer binding to be able to control sweep
    template = {
        '$schema': 'https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json',
        'type': 'pipeline',
        'experiment_name': 'None',
        'compute': 'azureml:cpu-cluster',
        'inputs': {
            'datasets_pkl': {
                'path': 'azureml://datastores/output/paths/placeholder',
                'type': 'uri_folder'
            },
            'project': 'None',
            'sweep_name': 'None',
            'label': 'None',
            'type': 'None',
            'primary_metric': 'None',
            'source': 'None',
            'downsample': False,
            'drop_bools': False,
            'web_hook': 'None',
            'next_pipeline': 0
        },
        'jobs': {
            'sweep_job': {
                'type': 'sweep',
                'trial': 'file:../../config/component/train.yaml',
                'compute': 'azureml:cpu-cluster',
                'sampling_algorithm': 'random',
                'search_space': {
                    'imputer': {
                        'type': 'choice',
                        'values': args.imputers.split(',')
                    },
                    'balancer': {
                        'type': 'choice',
                        'values': args.balancers.split(',')
                    },
                    'max_depth': {
                        'type': 'quniform',
                        'min_value': 5,
                        'max_value': 20,
                        'q': 1
                    },
                    'num_leaves': {
                        'type': 'quniform',
                        'min_value': 10,
                        'max_value': 50,
                        'q': 1
                    },
                    'colsample_bytree': {
                        'type': 'uniform',
                        'min_value': 0.5,
                        'max_value': 1.0,
                    },
                    'subsample': {
                        'type': 'uniform',
                        'min_value': 0.5,
                        'max_value': 1.0
                    },
                    'learning_rate': {
                        'type': 'uniform',
                        'min_value': 0.05,
                        'max_value': 0.2
                    }
                },
                'objective': {
                    'goal': 'maximize',
                    'primary_metric': '${{parent.inputs.primary_metric}}'
                },
                'limits': {
                    'max_total_trials': args.num_trials,
                    'max_concurrent_trials': 5,
                    'timeout': 7200
                },
                'inputs': {
                    'datasets_pkl': '${{parent.inputs.datasets_pkl}}',
                    'label': '${{parent.inputs.label}}',
                    'type': '${{parent.inputs.type}}',
                    'primary_metric': '${{parent.inputs.primary_metric}}'
                },
                'outputs': {
                    'train_artifacts': {}
                }
            },
            'get_best_model_job': {
                'type': 'command',
                'component': 'file:../../config/component/get_best_model.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.inputs.datasets_pkl}}',
                    'sweep_name': '${{parent.inputs.sweep_name}}',
                    'train_artifacts': '${{parent.jobs.sweep_job.outputs.train_artifacts}}',
                    'primary_metric': '${{parent.inputs.primary_metric}}'
                },
                'outputs': {
                    'transformed_data': ''
                }
            },
            'calibrate_job': {
                'type': 'command',
                'component': 'file:../../config/component/calibrate.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.jobs.get_best_model_job.outputs.transformed_data}}',
                    'type': '${{parent.inputs.type}}',
                    'label': '${{parent.inputs.label}}'
                },
                'outputs': {
                    'transformed_data': ''
                }
            },
            'register_model_job': {
                'type': 'command',
                'component': 'file:../../config/component/register_model.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.jobs.calibrate_job.outputs.transformed_data}}',
                    'project': '${{parent.inputs.project}}',
                    'label': '${{parent.inputs.label}}',
                    'primary_metric': '${{parent.inputs.primary_metric}}',
                    'type': '${{parent.inputs.type}}',
                    'source': '${{parent.inputs.source}}',
                    'downsample': '${{parent.inputs.downsample}}',
                    'drop_bools': '${{parent.inputs.drop_bools}}'
                },
                'outputs': {
                    'transformed_data': ''
                }
            },
            'web_hook_job': {
                'type': 'command',
                'component': 'file:../../config/component/web_hook.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.jobs.register_model_job.outputs.transformed_data}}',
                    'url': '${{parent.inputs.web_hook}}',
                    'param1': '${{parent.inputs.next_pipeline}}'
                }
            },
            'log_metrics_job': {
                'type': 'command',
                'component': 'file:../../config/component/log_metrics.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.jobs.calibrate_job.outputs.transformed_data}}',
                    'project': '${{parent.inputs.project}}',
                    'destination_folder': 'trainlog',
                    'downsample': '${{parent.inputs.downsample}}',
                    'drop_bools': '${{parent.inputs.drop_bools}}'
                },
                'outputs': {
                    'transformed_data': ''
                }
            }
        }
    }

    filepath = PurePath(os.path.dirname(os.path.realpath(__file__)), '../../config/pipeline', args.filename)
    print(f'pipeline yaml path: {filepath}')

    if os.path.exists(filepath):
        os.remove(filepath)

    with open(filepath, 'w') as f:
        yaml.SafeDumper.ignore_aliases = lambda *args: True
        yaml_str = yaml.safe_dump(template, sort_keys=False,  default_flow_style=False).replace(r"''", '')
        f.write(yaml_str)
    
    if eval(args.run) == True:
        dataset_path = f'azureml://datastores/output/paths/{args.project}/gold/' if args.variant == '' else f'azureml://datastores/output/paths/{args.project}/gold{args.variant}/'
        cmd = f'az ml job create --file {filepath} --stream --set inputs.project={args.project} --set inputs.source={args.source} --set inputs.type={args.type} --set inputs.primary_metric={args.primary_metric} --set inputs.datasets_pkl.path={dataset_path} --set experiment_name={args.project} --set inputs.label={args.label} --set inputs.downsample={args.downsample} --set inputs.drop_bools={args.drop_bools} --set inputs.web_hook="{args.web_hook}" --set inputs.next_pipeline={args.next_pipeline}'
        print(f'Running command: {cmd}')

        proc = subprocess.run(cmd.split(' '))
        print('The exit code was: %d' % proc.returncode)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--run', type=str, required=True)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
    parser.add_argument('--primary-metric', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--source', type=str, required=False)
    parser.add_argument('--imputers', type=str, required=False, default='mean')
    parser.add_argument('--balancers', type=str, required=False, default='none')
    parser.add_argument('--num-trials', type=int, required=False, default=5)
    parser.add_argument('--force-login', type=str, required=False, default='False')
    parser.add_argument('--downsample', type=str, default='False', required=False)
    parser.add_argument('--drop-bools', type=str, default='False', required=False)
    parser.add_argument('--web-hook', type=str, required=False)
    parser.add_argument('--next-pipeline', type=int, required=False)
    
    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == '__main__':
    # parse args
    args = parse_args()

    # determine variant of gold dataset to use
    variant = ''
    variant = variant + '-downsample' if eval(args.downsample) == True else variant
    variant = variant + '-dropbools' if eval(args.drop_bools) == True else variant
    args.variant = variant

    # run main function
    main(args)