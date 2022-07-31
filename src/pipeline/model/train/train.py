import os, argparse
import pickle
import joblib
import json
import mlflow
import lightgbm as lgb

from azureml.core.run import Run
from azureml.interpret import ExplanationClient
from interpret.ext.blackbox import TabularExplainer
from pathlib import Path
from sklearn.metrics import classification_report, mean_squared_error


def main(ctx):
    # read in data
    with open(ctx['args'].datasets_pkl + '/datasets.pkl', 'rb') as handle:
        dict_files = pickle.load(handle)

    print(f'dict_files.keys(): {dict_files.keys()}')

    X_train = dict_files[f'X_train_{args.imputer}_{args.balancer}']
    y_train = dict_files[f'y_train_{args.imputer}_{args.balancer}']
    X_valid = dict_files[f'X_valid_{args.imputer}_{args.balancer}']
    y_valid = dict_files[f'y_valid_{args.imputer}_{args.balancer}']
    X_test = dict_files[f'X_test_{args.imputer}_{args.balancer}']
    y_test = dict_files[f'y_test_{args.imputer}_{args.balancer}']
    
    # train model
    metrics = {}
    if ctx['args'].type != 'Regression':
        clf = lgb.LGBMClassifier(num_leaves=int(args.num_leaves), 
                            max_depth=int(args.max_depth), 
                            colsample_bytree=args.colsample_bytree,
                            subsample=args.subsample,
                            learning_rate=args.learning_rate,
                            n_estimators=1000,
                            n_jobs=4,
                            random_state=314,
                            force_row_wise=True,
                            verbose=2)

        clf.fit(X_train, y_train[args.label].ravel(),
            eval_set=[(X_valid, y_valid[args.label].ravel())],
            eval_metric='logloss',
            callbacks=[lgb.early_stopping(10)])

        yhat_proba = [x[1] for x in clf.predict_proba(X_test)]
        yhat = [1 if x >= 0.5 else 0 for x in yhat_proba]

        # log metrics
        report = classification_report(y_test[args.label].ravel(), yhat, output_dict=True)
        for k1,v1 in report.items():
            if isinstance(v1, dict):
                for k2,v2 in v1.items():
                    metrics['{}_{}'.format(k1,k2).replace(' ', '-')] = v2
                    mlflow.log_metric('{}_{}'.format(k1,k2).replace(' ', '-'), v2)
    else:
        clf = lgb.LGBMRegressor(num_leaves=int(args.num_leaves), 
                            max_depth=int(args.max_depth), 
                            colsample_bytree=args.colsample_bytree,
                            subsample=args.subsample,
                            learning_rate=args.learning_rate,
                            n_estimators=1000,
                            n_jobs=4,
                            random_state=314,
                            force_row_wise=True,
                            verbose=2)

        clf.fit(X_train, y_train[args.label].ravel(),
            eval_set=[(X_valid, y_valid[args.label].ravel())],
            eval_metric='l2',
            callbacks=[lgb.early_stopping(10)])

        yhat = [x for x in clf.predict(X_test)]

        rmse = mean_squared_error(y_test[args.label].ravel(), yhat, squared=False)
        metrics['root_mean_squared_error'] = rmse
        mlflow.log_metric('root_mean_squared_error', rmse)

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

    dict_files = {
        'X_train': X_train,
        'y_train': y_train,
        'X_valid': X_valid,
        'y_valid': y_valid,
        'X_test': X_test,
        'y_test': y_test
    }

    with open('outputs/datasets.pkl', 'wb') as handle:
        pickle.dump(dict_files, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(Path(ctx['args'].train_artifacts) / 'best_run.json', 'w') as f:
        json.dump({'runId': ctx['run'].id, 'sweepId': ctx['run'].parent.id, 'best_score': metrics[ctx['args'].primary_metric], 'label': ctx['args'].label}, f)


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
