import sys, os, argparse
import pickle
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, ADASYN

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from lazy_eval import LazyEval

def main(ctx):
    # read in data
    with open(ctx['args'].datasets_pkl + '/datasets.pkl', 'rb') as f:
        dict_files = pickle.load(f)
        data = LazyEval(dict_files)

    keys = [x for x in dict_files.keys()]
    for key in keys:
        if key.startswith('imputer'):
            arr = key.split('_')
            if arr[5] != 'rus':
                df_x = dict_files['X_train_none']
                cat_indices = [df_x.columns.get_loc(x) for x in df_x.columns if df_x[x].dtype.name == 'category' or df_x[x].dtype.name == 'boolean']

                df_x, df_y = data.get('train', arr[4])

                # apply over-samplers
                balancer = RandomOverSampler()
                balancer.fit_resample(df_x, df_y)
                dict_files[f'balancer____{arr[4]}_ros'] = balancer

                if len(cat_indices) > 0:
                    print('Categorical and/or boolean features found, using SMOTENC')
                    try:
                        balancer = SMOTENC(categorical_features=cat_indices)
                        balancer.fit_resample(df_x, df_y)
                        dict_files[f'balancer____{arr[4]}_smote'] = balancer
                    except:
                        print('Error running SMOTE')
                else:
                    print('Categorical and/or boolean features NOT found, using SMOTE/ADASYN')
                    try:
                        balancer = SMOTE()
                        balancer.fit_resample(df_x, df_y)
                        dict_files[f'balancer____{arr[4]}_smote'] = balancer
                    except:
                        print('Error running SMOTE')

                    try:
                        balancer = ADASYN()
                        balancer.fit_resample(df_x, df_y)
                        dict_files[f'balancer____{arr[4]}_adasyn'] = balancer
                    except:
                        print('Error running ADASYN')

        elif key.startswith('imputer') or key.startswith('outliers'):
            dict_files[key] = dict_files[key]

    # save data to outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(dict_files, f, protocol=pickle.HIGHEST_PROTOCOL)

    copy_tree('outputs', args.transformed_data)


def start(args):
    os.makedirs('outputs', exist_ok=True)
    mlflow.start_run()
    mlflow.autolog()
    run = Run.get_context()
    tags = run.parent.get_tags()
    return {
        'args': args,
        'run': run,
        'tags': tags
    }


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--datasets-pkl', type=str, default='data')
    parser.add_argument('--type', type=str)
    parser.add_argument('--transformed_data', type=str, help='Path of output data')

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == '__main__':
    # parse args
    args = parse_args()
    ctx = start(args)

    # run main function
    main(ctx)
