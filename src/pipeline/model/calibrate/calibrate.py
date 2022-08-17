import sys, os, argparse
import joblib
import pickle
import mlflow

from azureml.core import Run
from distutils.dir_util import copy_tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from lazy_eval import LazyEval


def calc_reliability(y_test, yhat, yhat_proba, suffix):    
    # calculate brier score
    brier_score = brier_score_loss(y_test, yhat_proba)
    mlflow.log_metric('brier_score_{}'.format(suffix), brier_score)

    return brier_score


def main(ctx):
    # read in data
    model = joblib.load(ctx['args'].datasets_pkl + '/model.pkl')

    with open(ctx['args'].datasets_pkl + '/datasets.pkl', 'rb') as f:
        dict_files = pickle.load(f)
        data = LazyEval(dict_files)

    # use lazy eval to get transformed data
    X_valid, y_valid = data.get('valid', ctx['args'].imputer, ctx['args'].balancer)
    X_test, y_test = data.get('test', ctx['args'].imputer, ctx['args'].balancer)

    if ctx['args'].type != 'Regression':
        # brier score before calibration
        yhat_proba = [x[1] for x in model.predict_proba(X_test)]
        yhat = [1 if x >= 0.5 else 0 for x in yhat_proba]

        brier_score_before = calc_reliability(y_test, yhat, yhat_proba, 'before')

        # calibrate probabilities
        calib_clf = CalibratedClassifierCV(model, method='isotonic', cv='prefit') 
        calib_clf.fit(X_valid, y_valid[args.label].ravel()) 

        #  brier score after calibration
        yhat_proba = [x[1] for x in calib_clf.predict_proba(X_test)]
        yhat = [1 if x >= 0.5 else 0 for x in yhat_proba]

        brier_score_after = calc_reliability(y_test, yhat, yhat_proba, 'after')

        if brier_score_after < brier_score_before:
            print('Brier score improves with calibration, will use CalibratedClassifierCV')
            model = calib_clf

    else:
        print('Regression model; nothing to do')

    joblib.dump(model, 'outputs/model.pkl')
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
    parser.add_argument('--type', type=str, default='None')
    parser.add_argument('--label', type=str, default='None')
    parser.add_argument('--transformed_data', type=str)

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

