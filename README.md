# Azure-ML-Accelerator (CLI V2)

## Intro

First off, this accelerator is **not** affiliated with Microsoft and is more of an effort to generalize the activities found in most machine learning pipelines and package them up into a clean set of reusable pipelines that can then be adapted to many types of projects.  Functionality will continue to be added along with more advanced scenarios.  Please see http://latticecraft.ai for more information about current and planned features.

Furthermore, this accelerator is **not** meant to be a one-sized-fits-all solution to all problems, and assumes that the input dataset has been engineered to a point similar to datasets that might be found on Kaggle.

## Quickstart

1. Setup local environment using: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public
2. Create a new resource group, and from the command line cd to the project root.  Run:
~~~
python src/commands/setup_rg.py --resource-group [RG name]
~~~
3. Set workspace defaults:
~~~
python src/commands/set_defaults.py
~~~
4. To register a new remote environment run:
~~~
python src/commands/setup_env.py
~~~
5. Upload the input dataset to a new folder in the input container in the blob storage with ltcft prefix.  The name of the folder will be the name of the project and must be used in subsequent steps.
6. Run Featurization pipeline:
~~~
python src/commands/featurize.py --project [Folder name]  --input [File name] --separator [comma|semicolon] --label [Column name of label] --type [Binary|Regression] --replacements [URL encode of Replacements JSON, optional] --datatypes [URL encode of DataTypes JSON, optional] --filename featurize-gen.yaml --run True
~~~
7. Run Train pipeline:
~~~
python src/commands/train.py --project [Folder name] --label [Column name of label] --type [Binary|Regression] --primary-metric [weighted-avg_f1-score|root_mean_squared_error]  --source [Source, optional] --filename train-gen.yaml --run True
~~~
8. Optionally, DevOps release pipelines are in config/devops folder which runs above commands.
