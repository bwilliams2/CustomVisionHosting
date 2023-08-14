# Hosting Custom Vision Models in AzureML Using Nvidia Triton Inference Server

## Introduction
This repository is a guide for hosting a custom vision model in AzureML using Nvidia Triton Inference Server. The model is trained using the [Custom Vision Service](https://www.customvision.ai/). The model is exported as an ONNX model and hosted in AzureML using Nvidia Triton Inference Server deployed on an online endpoint. This repository has been adapted from configurations and workflows linked in the [Additional Resources](#additional-resources) section.

## Pre-requisites

- [Azure Account](https://azure.microsoft.com/en-us/free/)
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest)
- [Azure CLI ml extension](https://docs.microsoft.com/en-us/azure/machine-learning/reference-azure-machine-learning-cli)
- [Python $\geq$ 3.9](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/get-docker/)

## Setup

It is assumed that both an AzureML workspace and a Custom Vision Service have been created and are available for use. If not, follow the steps below to create both.

- [Create AzureML Workspace](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources?view=azureml-api-2)
- [Create Custom Vision Service](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources?view=azureml-api-2)

### Azure CLI

Login to Azure using the Azure CLI. To login using your browser, run the following command:

```bash
az login
```

For more information on logging in to Azure using the Azure CLI, see [here](https://docs.microsoft.com/en-us/cli/azure/authenticate-azure-cli?view=azure-cli-latest).

### Environment Variables

The following environment variables are required to run the `deploy.sh` script and are read from `.env` file in the top-level directory. The values for these variables can be found in the Azure Portal.

#### AzureML Variables
- `AZUREML_RESOURCE_GROUP`
- `AZUREML_WORKSPACE`
- `AZURE_LOCATION`

#### Custom Vision Resource Variables
- `VISION_TRAINING_KEY`
- `VISION_TRAINING_ENDPOINT`
- `VISION_PREDICTION_KEY`
- `VISION_PREDICTION_ENDPOINT`
- `VISION_PREDICTION_RESOURCE_ID`

#### Custom Vision Project Variables
- `VISION_PROJECT_NAME`
- `VISION_PUBLISH_ITERATION_NAME`

#### Use With Existing Project and Iteration (Optional)
- `VISION_ITERATION_ID`
- `VISION_PROJECT_ID`

If `VISION_PROJECT_NAME` is provided and resolves to a current project, the script will assign the project ID and attempt to find any provided iteration ID using the Custom Vision Service API.

## Steps

The `deploy.sh` script contains the steps to setup, train, and deploy the Custom Vision model to the AzureML endpoint. It can be run using the following command:

```bash
./deploy.sh
```

During execution the script will:

1. Create Custom Vision project, train model, and publish iteration
2. Export model as ONNX
3. Build Docker image for Triton Inference Server
4. Create AzureML endpoint and deploy Triton Inference Server with Custom Vision model
5. Test endpoint with sample image

## Additional Resources
- [Custom Vision Service Quickstart](https://learn.microsoft.com/en-us/azure/ai-services/custom-vision-service/quickstarts/object-detection?tabs=linux%2Cvisual-studio&pivots=programming-language-python)

- [AzureML Examples - Triton Custom Container](https://github.com/Azure/azureml-examples/tree/main/cli/endpoints/online/custom-container/triton/single-model)

- [AzureML Examples - Triton Managed Deployment](https://github.com/Azure/azureml-examples/tree/main/cli/endpoints/online/triton/single-model)