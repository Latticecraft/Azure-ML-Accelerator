# Azure-ML-Accelerator

## Intro

This accelerator can be used to start a new machine learning project for Azure ML Service using a known good configuration and with the core ML tasks generalized so as to be as reusable as possible.  Functionality will continue to be added along with more advanced scenarios.  The intention is to have an architectural reference that can be adapted to a variety of projects.

This accelerator is **not** meant to be a one-sized-fits-all solution to all problems, and assumes that the input dataset has been engineered to a point similar to datasets that might be found on Kaggle.

## Quickstart

1. Setup local environment using: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public

2. Create a new resource group, and from the command line cd to the project root.  Run:
~~~
python src/commands/setup_rg.py --resource-group <RG name>
~~~

3. Set workspace defaults:
~~~
python src/commands/set_defaults.py
~~~

4. Upload dataset to a new folder in the input container in the blob storage with ltcft prefix.  The name of the folder will be the name of the project and must be used in subsequent steps.

5. Run Featurization pipeline:
~~~
python src/commands/featurize.py --project [Folder name]  --input [File name] --separator [comma|semicolon] --label [Column name of label] --type [Binary|Regression] --replacements [URL encode of Replacements JSON, optional] --datatypes [URL encode of DataTypes JSON, optional] --filename featurize-gen.yaml --run True
~~~

6. Run Train pipeline:
~~~
python src/commands/train.py --project [Folder name] --label [Column name of label] --type [Binary|Regression] --primary-metric [weighted-avg_f1-score|root_mean_squared_error]  --source [Source, optional] --filename train-gen.yaml --run True
~~~

7. Optionally, DevOps release pipelines are in config/devops folder which runs above commands.
