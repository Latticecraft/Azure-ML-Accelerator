import os, argparse
import subprocess
import yaml

from pathlib import PurePath


def main(args):
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
            'source': 'None',
            'primary_metric': 'None'
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
                        'values': ['mean', 'knn']
                    },
                    'balancer': {
                        'type': 'choice',
                        'values': ['none', 'ros', 'smote']
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
                    'goal': 'minimize',
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
                    'sweep_name': '${{parent.inputs.sweep_name}}',
                    'train_artifacts': '${{parent.jobs.sweep_job.outputs.train_artifacts}}',
                    'primary_metric': '${{parent.inputs.primary_metric}}'
                },
                'outputs': {
                    'transformed_data': {}
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
                    'transformed_data': {}
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
                    'source': '${{parent.inputs.source}}'
                },
                'outputs': {
                    'transformed_data': {}
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
                    'destination_folder': 'trainlog'
                },
                'outputs': {
                    'transformed_data': {}
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
        yaml.safe_dump(template, f, sort_keys=False,  default_flow_style=False)
    
    if eval(args.run) == True:
        command = f'az ml job create --file {filepath} --web --set inputs.project={args.project} --set inputs.source={args.source} --set inputs.type={args.type} --set inputs.primary_metric={args.primary_metric} --set inputs.datasets_pkl.path=azureml://datastores/output/paths/{args.project}/gold/ --set experiment_name={args.project} --set inputs.label={args.label} --set inputs.web_hook="{args.web_hook}" --set inputs.next_pipeline={args.next_pipeline}'
        print(f'command: {command}')

        list_files = subprocess.run(command.split(' '))
        print('The exit code was: %d' % list_files.returncode)

        # remove temp file
        os.remove(filepath)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--run', type=str, required=False)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--type', type=str, required=False)
    parser.add_argument('--primary-metric', type=str, required=False)
    parser.add_argument('--source', type=str, required=False)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--num-trials', type=int, required=False, default=5)
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

    # run main function
    main(args)