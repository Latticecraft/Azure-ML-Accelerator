# Azure-ML-Accelerator (CLI V2)

## Intro

The goal of this project is to generalize the activities found in most machine learning pipelines and package them up into a clean set of reusable pipelines that can then be adapted to many types of projects.

Functionality will continue to be added along with more advanced scenarios including the automated HTML5 dashboard.  Please see http://latticecraft.ai for more information about current and planned features.

## Quickstart

### CLI

1. Setup local environment using: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?tabs=public

2. Login to Azure CLI
~~~
az login
~~~
3. Copy/paste the following snippet into a shell (taking into consideration parameters listed):
- RG_NAME: Change to your desired Resource Group name
- vmSize: Defaults to STANDARD_DS3_V2, but this may need to be changed depending on allowed SKUs.  If deployment fails it's likely due to this.
~~~
RG_NAME="ReplaceWithNewName"
az group create --name ${RG_NAME}
ROLE_ID=$(az role definition list --query "[?roleName=='Contributor'].name | [0]" |  tr -d '"')
az deployment group create --resource-group ${RG_NAME} --template-uri https://raw.githubusercontent.com/Latticecraft/ML-Builder/main/config/azuredeploy.json --parameters roleDefinitionId=${ROLE_ID} imageLabel="3" vmSize="STANDARD_DS3_V2"
~~~
4. The above command will create the ML-Builder ACI container in the specified resource group which will start an Airflow DAG and fully provision the Azure ML Service environment with a sample dataset as well as run the Featurization and Train pipelines.  This process generally takes around 30 minutes.  Once completed you can clone this repository and customize as needed with a new dataset.

## Workflow

![Workflow](http://www.latticecraft.ai/images/workflow2.svg)

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

