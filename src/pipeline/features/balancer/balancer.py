import sys, os, argparse
import mlflow
import pickle
import time

from azureml.core import Run
from distutils.dir_util import copy_tree
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, SMOTEN, ADASYN, BorderlineSMOTE, KMeansSMOTE
from imblearn.under_sampling import NearMiss, NeighbourhoodCleaningRule, OneSidedSelection

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
            df_x = dict_files['X_train']
            cat_indices = [df_x.columns.get_loc(x) for x in df_x.columns if df_x[x].dtype.name == 'category' or df_x[x].dtype.name == 'bool']

            df_x, df_y = data.get('train', arr[4])

            # apply over-samplers
            start = time.time()
            balancer = RandomOverSampler()
            balancer.fit_resample(df_x, df_y)
            dict_files[f'balancer____{arr[4]}_ros'] = balancer
            end = time.time()
            print(f'RandomOverSampler elapsed time: {end - start}')

            if len(cat_indices) == len(df_x.columns):
                print('All categorical/boolean features found, using SMOTEN')
                try:
                    start = time.time()
                    balancer = SMOTEN(categorical_features=cat_indices)
                    balancer.fit_resample(df_x, df_y)
                    dict_files[f'balancer____{arr[4]}_smote'] = balancer
                    end = time.time()
                    print(f'SMOTEN elapsed time: {end - start}')
                except:
                    print('Error running SMOTEN')
            elif len(cat_indices) > 0:
                print('Mix of continuous and categorical/boolean features found, using SMOTENC')
                try:
                    start = time.time()
                    balancer = SMOTENC(categorical_features=cat_indices)
                    balancer.fit_resample(df_x, df_y)
                    dict_files[f'balancer____{arr[4]}_smote'] = balancer
                    end = time.time()
                    print(f'SMOTENC elapsed time: {end - start}')
                except:
                    print('Error running SMOTENC')
            else: # all continuous
                print('Categorical and/or boolean features NOT found, using SMOTE/ADASYN')
                try:
                    start = time.time()
                    balancer = SMOTE()
                    balancer.fit_resample(df_x, df_y)
                    dict_files[f'balancer____{arr[4]}_smote'] = balancer
                    end = time.time()
                    print(f'SMOTE elapsed time: {end - start}')
                except:
                    print('Error running SMOTE')

                try:
                    start = time.time()
                    balancer = ADASYN()
                    balancer.fit_resample(df_x, df_y)
                    dict_files[f'balancer____{arr[4]}_adasyn'] = balancer
                    end = time.time()
                    print(f'ADASYN elapsed time: {end - start}')
                except:
                    print('Error running ADASYN')

                try:
                    start = time.time()
                    balancer = BorderlineSMOTE()
                    balancer.fit_resample(df_x, df_y)
                    dict_files[f'balancer____{arr[4]}_borderlinesmote'] = balancer
                    end = time.time()
                    print(f'BorderlineSMOTE elapsed time: {end - start}')
                except:
                    print('Error running BorderlineSMOTE')

                try:
                    start = time.time()
                    balancer = KMeansSMOTE()
                    balancer.fit_resample(df_x, df_y)
                    dict_files[f'balancer____{arr[4]}_kmeanssmote'] = balancer
                    end = time.time()
                    print(f'KMeansSMOTE elapsed time: {end - start}')
                except:
                    print('Error running KMeansSMOTE')

                # undersamplers
                try:
                    start = time.time()
                    balancer = NearMiss(version=1)
                    balancer.fit_resample(df_x, df_y)
                    dict_files[f'balancer____{arr[4]}_nearmiss1'] = balancer
                    end = time.time()
                    print(f'NearMiss elapsed time: {end - start}')
                except:
                    print('Error running NearMiss')

                try:
                    start = time.time()
                    balancer = NearMiss(version=2)
                    balancer.fit_resample(df_x, df_y)
                    dict_files[f'balancer____{arr[4]}_nearmiss2'] = balancer
                    end = time.time()
                    print(f'NearMiss elapsed time: {end - start}')
                except:
                    print('Error running NearMiss')

                try:
                    start = time.time()
                    balancer = NearMiss(version=3)
                    balancer.fit_resample(df_x, df_y)
                    dict_files[f'balancer____{arr[4]}_nearmiss3'] = balancer
                    end = time.time()
                    print(f'NearMiss elapsed time: {end - start}')
                except:
                    print('Error running NearMiss')

                if eval(ctx['args'].enable_slow) == True:
                    try:
                        start = time.time()
                        balancer =  NeighbourhoodCleaningRule()
                        balancer.fit_resample(df_x, df_y)
                        dict_files[f'balancer____{arr[4]}_neighborhoodcleaningrule'] = balancer
                        end = time.time()
                        print(f'NeighbourhoodCleaningRule elapsed time: {end - start}')
                    except:
                        print('Error running NeighbourhoodCleaningRule')
                    
                    try:
                        start = time.time()
                        balancer =  OneSidedSelection()
                        balancer.fit_resample(df_x, df_y)
                        dict_files[f'balancer____{arr[4]}_onesidedselection'] = balancer
                        end = time.time()
                        print(f'OneSidedSelection elapsed time: {end - start}')
                    except:
                        print('Error running OneSidedSelection')
                
        elif key.startswith('imputer') or key.startswith('outliers'):
            dict_files[key] = dict_files[key]

    # save data to outputs
    with open('outputs/datasets.pkl', 'wb') as f:
        pickle.dump(dict_files, f, protocol=pickle.HIGHEST_PROTOCOL)

    copy_tree('outputs', args.transformed_data)


def start(args):
    os.makedirs('outputs', exist_ok=True)
    mlflow.start_run()
    mlflow.autolog(disable=True)
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
    parser.add_argument('--enable-slow', type=str, default='False')

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
