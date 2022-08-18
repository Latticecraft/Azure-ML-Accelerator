import sys, os, argparse
import pickle
import joblib
import json
import mlflow
import lightgbm as lgb

from azureml.core.run import Run
from azureml.interpret import ExplanationClient
from distutils.dir_util import copy_tree
from interpret.ext.blackbox import TabularExplainer
from pathlib import Path
from sklearn.metrics import classification_report, mean_squared_error

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from lazy_eval import LazyEval


def main(ctx):
    # read in data
    with open(ctx['args'].datasets_pkl + '/datasets.pkl', 'rb') as handle:
        dict_files = pickle.load(handle)
        data = LazyEval(dict_files)

    print(f'dict_files.keys(): {dict_files.keys()}')

    # use lazy eval to get transformed data
    X_train, y_train = data.get('train', ctx['args'].imputer, ctx['args'].balancer)
    X_valid, y_valid = data.get('valid', ctx['args'].imputer, ctx['args'].balancer)
    X_test, y_test = data.get('test', ctx['args'].imputer, ctx['args'].balancer)

    # train model and get predictions
    clf = get_untrained_model(ctx)
    fit_model(ctx, clf, X_train, y_train, X_valid, y_valid)
    yhat = get_predictions(ctx, clf, X_test, y_test)
    metrics = log_metrics(ctx, y_test, yhat)

    # explanations
    client = ExplanationClient.from_run(ctx['run'])
    explainer = TabularExplainer(clf, X_train)
    global_explanation = explainer.explain_global(X_test)
    client.upload_model_explanation(global_explanation, comment='global explanation: all features')

    # log important features
    dict_features = { f'feature_rank_{i}': v for i,v in enumerate(global_explanation.get_ranked_global_names()) if i < 20 }
    with open('outputs/features_ranked.json', 'w') as f:
        json.dump(dict_features, f)

    mlflow.log_dict('features_ranked', 'outputs/features_ranked.json')

    # log model
    joblib.dump(clf, 'outputs/model.pkl')

    with open('outputs/datasets.pkl', 'wb') as handle:
        pickle.dump(dict_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path(ctx['args'].train_artifacts) / 'best_run.json', 'w') as f:
        json.dump({
            'runId': ctx['run'].id, 
            'sweepId': ctx['run'].parent.id, 
            'best_score': metrics[ctx['args'].primary_metric], 
            'label': ctx['args'].label,
            'imputer': ctx['args'].imputer,
            'balancer': ctx['args'].balancer
        }, f)

    copy_tree(ctx['args'].train_artifacts, 'outputs')


def get_untrained_model(ctx):
    if ctx['args'].type != 'Regression':
        clf = lgb.LGBMClassifier(num_leaves=int(ctx['args'].num_leaves), 
                max_depth=int(ctx['args'].max_depth), 
                colsample_bytree=ctx['args'].colsample_bytree,
                subsample=ctx['args'].subsample,
                learning_rate=ctx['args'].learning_rate,
                n_estimators=1000,
                n_jobs=4,
                random_state=314,
                force_row_wise=True,
                verbose=2)
    
    else:
        clf = lgb.LGBMRegressor(num_leaves=int(ctx['args'].num_leaves), 
                max_depth=int(ctx['args'].max_depth), 
                colsample_bytree=ctx['args'].colsample_bytree,
                subsample=ctx['args'].subsample,
                learning_rate=ctx['args'].learning_rate,
                n_estimators=1000,
                n_jobs=4,
                random_state=314,
                force_row_wise=True,
                verbose=2)

    return clf


def fit_model(ctx, clf, X_train, y_train, X_valid, y_valid):
    clf.fit(X_train, y_train[ctx['args'].label].ravel(),
            eval_set=[(X_valid, y_valid[ctx['args'].label].ravel())],
            eval_metric='logloss' if ctx['args'].type != 'Regression' else 'l2',
            callbacks=[lgb.early_stopping(10)])


def get_predictions(ctx, clf, X_test, y_test):
    if ctx['args'].type != 'Regression':
        yhat_proba = [x[1] for x in clf.predict_proba(X_test)]
        yhat = [1 if x >= 0.5 else 0 for x in yhat_proba]
    else:
        yhat = [x for x in clf.predict(X_test)]

    return yhat


def log_metrics(ctx, y_test, yhat):
    metrics = {}
    if ctx['args'].type != 'Regression':
        report = classification_report(y_test[ctx['args'].label].ravel(), yhat, output_dict=True)
        for k1,v1 in report.items():
            if isinstance(v1, dict):
                for k2,v2 in v1.items():
                    metrics['{}_{}'.format(k1,k2).replace(' ', '-')] = v2
                    mlflow.log_metric('{}_{}'.format(k1,k2).replace(' ', '-'), v2)
    else:
        rmse = mean_squared_error(y_test[ctx['args'].label].ravel(), yhat, squared=False)
        metrics['neg_root_mean_squared_error'] = -rmse
        mlflow.log_metric('neg_root_mean_squared_error', -rmse)

    return metrics


def start(args):
    os.makedirs('outputs', exist_ok=True)
    mlflow.start_run()
    mlflow.autolog(log_models=False)
    run = Run.get_context()
    tags = run.parent.parent.parent.get_tags()
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
    parser.add_argument('--label', type=str, default='label')
    parser.add_argument('--type', type=str, default='None')
    parser.add_argument('--primary-metric', type=str, default='None')
    parser.add_argument('--imputer', type=str, default='knn')
    parser.add_argument('--balancer', type=str, default='smote')
    parser.add_argument('--num-leaves', type=float, default=5)
    parser.add_argument('--max-depth', type=float, default=5)
    parser.add_argument('--colsample-bytree', type=float, default=0.5)
    parser.add_argument('--subsample', type=float, default=0.5)
    parser.add_argument('--learning-rate', type=float, default=0.15)
    parser.add_argument('--train-artifacts', type=str)

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
