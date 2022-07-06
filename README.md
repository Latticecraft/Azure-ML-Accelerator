# Azure-ML-Accelerator (CLI V2)

## Few notes...

- This accelerator is **not** affiliated with Microsoft.

- This accelerator is **not** meant to be a one-size-fits-all solution to all problems, and assumes that the input dataset has been engineered to a point similar to datasets that might be found on Kaggle.

- If using provided ARM template, it is generally advisable that you first audit the security settings of created resources to ensure they conform to your standards prior to use.

## Intro

The goal of this project is to generalize the activities found in most machine learning pipelines and package them up into a clean set of reusable pipelines that can then be adapted to many types of projects.

Functionality will continue to be added along with more advanced scenarios including the automated HTML5 dashboard.  Please see http://latticecraft.ai for more information about current and planned features.

## Quickstart

### CLI

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

### DevOps

DevOps release pipelines are stored in config/devops which will run above commands.

1. In Releases section, import Featurize.json and Train.json

2. In Library section, create a new variable group with the following variables (values included here as an example):
![VariableGroupExample](https://user-images.githubusercontent.com/1169037/177462462-802632ac-1b41-4721-9598-f61f42899f0a.png)

## Folder structure

> config

>> component

YAML definitions for each pipeline step.

>> devops

DevOps release pipeline exports for each pipeline.

>> environment

Conda environment definition which is used for local/remote pipeline runs.

>> pipeline

Pipeline YAML definitions.

>> resourcegroup

ARM template which can be used to provision PaaS components.

> src

>> commands

Python wrappers for common tasks such as running pipelines and setting up environment.

>> pipeline

Python code for each pipeline step.

