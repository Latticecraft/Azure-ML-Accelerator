import os, argparse
import subprocess
import yaml

from pathlib import PurePath


def main(args):
    template = {
        '$schema': 'https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json',
        'type': 'pipeline',
        'inputs': {
            'input_csv': {
                'type': 'uri_file',
                'path': 'azureml://datastores/input/paths/placeholder'
            },
            'project': 'None',
            'type': 'None',
            'separator': 'comma',
            'label': 'None',
            'unwanted': 'None',
            'replacements': 'None',
            'datatypes': 'None',
            'remove_low_info': True,
            'remove_high_corr': True,
            'remove_high_nan': True,
            'remove_single_val': True,
            'web_hook': 'None',
            'next_pipeline': 0
        },
        'experiment_name': 'None',
        'compute': 'azureml:cpu-cluster',
        'jobs': {
            'input_job': {
                'type': 'command',
                'component': 'file:../../config/component/input.yaml',
                'inputs': {
                    'environment': '${{parent.inputs.environment}}',
                    'input_csv': '${{parent.inputs.input_csv}}',
                    'separator': '${{parent.inputs.separator}}'
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'set_header_job': {
                'type': 'command',
                'component': 'file:../../config/component/set_header.yaml',
                'inputs': {
                    'input_data': '${{parent.jobs.input_job.outputs.transformed_data}}'
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'format_job': {
                'type': 'command',
                'component': 'file:../../config/component/format.yaml',
                'inputs': {
                    'input_data': '${{parent.jobs.set_header_job.outputs.transformed_data}}',
                    'label': '${{parent.inputs.label}}',
                    'replacements': '${{parent.inputs.replacements}}',
                    'datatypes': '${{parent.inputs.datatypes}}',
                    'type': '${{parent.inputs.type}}'
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'select_nonlabel_job': {
                'type': 'command',
                'component': 'file:../../config/component/select_columns.yaml',
                'inputs': {
                    'input_data': '${{parent.jobs.format_job.outputs.transformed_data}}',
                    'columns': '${{parent.inputs.label}}',
                    'take_complement': True
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'select_unwanted_job': {
                'type': 'command',
                'component': 'file:../../config/component/select_columns.yaml',
                'inputs': {
                    'input_data': '${{parent.jobs.select_nonlabel_job.outputs.transformed_data}}',
                    'columns': '${{parent.inputs.unwanted}}',
                    'take_complement': False
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'select_features_job': {
                'type': 'command',
                'component': 'file:../../config/component/select_columns.yaml',
                'inputs': {
                    'input_data': '${{parent.jobs.select_nonlabel_job.outputs.transformed_data}}',
                    'columns': '${{parent.inputs.unwanted}}',
                    'take_complement': True
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'select_label_job': {
                'type': 'command',
                'component': 'file:../../config/component/select_columns.yaml',
                'inputs': {
                    'input_data': '${{parent.jobs.format_job.outputs.transformed_data}}',
                    'columns': '${{parent.inputs.label}}',
                    'take_complement': False
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'package_job': {
                'type': 'command',
                'component': 'file:../../config/component/package_files.yaml',
                'inputs': {
                    'features_csv': '${{parent.jobs.select_features_job.outputs.transformed_data}}',
                    'label_csv': '${{parent.jobs.select_label_job.outputs.transformed_data}}',
                    'unwanted_csv': '${{parent.jobs.select_unwanted_job.outputs.transformed_data}}'
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'split_dataset_job': {    
                'type': 'command',
                'component': 'file:../../config/component/split_dataset.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.jobs.package_job.outputs.transformed_data}}'
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'downsample_job': {
                'type': 'command',
                'component': 'file:../../config/component/downsample.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.jobs.split_dataset_job.outputs.transformed_data}}'
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'dfs_job': {
                'type': 'command',
                'component': 'file:../../config/component/dfs.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.jobs.downsample_job.outputs.transformed_data}}',
                    'remove_low_info': '${{parent.inputs.remove_low_info}}',
                    'remove_high_corr': '${{parent.inputs.remove_high_corr}}',
                    'remove_high_nan': '${{parent.inputs.remove_high_nan}}',
                    'remove_single_val': '${{parent.inputs.remove_single_val}}'
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'stats_job': {
                'type': 'command',
                'component': 'file:../../config/component/stats.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.jobs.dfs_job.outputs.transformed_data}}'
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'impute_job': {
                'type': 'command',
                'component': 'file:../../config/component/impute.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.jobs.stats_job.outputs.transformed_data}}'
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'balancer_job': {
                'type': 'command',
                'component': 'file:../../config/component/balancer.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.jobs.impute_job.outputs.transformed_data}}',
                    'type': '${{parent.inputs.type}}'
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'register_dataset_job': {
                'type': 'command',
                'component': 'file:../../config/component/register_dataset.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.jobs.balancer_job.outputs.transformed_data}}',
                    'project': '${{parent.inputs.project}}'
                },
                'outputs': {
                    'transformed_data': {}
                }
            },
            'web_hook_job': {
                'type': 'command',
                'component': 'file:../../config/component/web_hook.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.jobs.register_dataset_job.outputs.transformed_data}}',
                    'url': '${{parent.inputs.web_hook}}',
                    'param1': '${{parent.inputs.next_pipeline}}'
                }
            },
            'log_metrics_job': {
                'type': 'command',
                'component': 'file:../../config/component/log_metrics.yaml',
                'inputs': {
                    'datasets_pkl': '${{parent.jobs.balancer_job.outputs.transformed_data}}',
                    'project': '${{parent.inputs.project}}',
                    'destination_folder': 'runinfo'
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
        command = f'az ml job create --file {filepath} --web --set inputs.project={args.project} --set inputs.type={args.type} --set inputs.input_csv.path=azureml://datastores/input/paths/{args.project}/{args.input} --set experiment_name={args.project} --set inputs.label={args.label} --set inputs.unwanted={args.unwanted} --set inputs.replacements={args.replacements} --set inputs.datatypes={args.datatypes} --set inputs.separator={args.separator} --set inputs.web_hook="{args.web_hook}" --set inputs.next_pipeline={args.next_pipeline}'
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
    parser.add_argument('--run', type=str, required=True)
    parser.add_argument('--project', type=str, required=True)
    parser.add_argument('--type', type=str, required=True)
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