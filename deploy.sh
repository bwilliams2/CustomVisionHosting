#/bin/bash

set -e

pip install gevent requests pillow tritonclient[all] azure-cognitiveservices-vision-customvision python-dotenv click onnx numpy

# <set_variables>
export $(grep -v '^#' .env | xargs)
# </set_variables>

# Set default resource group, workspace, and location
az configure --defaults group=$AZUREML_RESOURCE_GROUP workspace=$AZUREML_WORKSPACE location=$AZURE_LOCATION

# Train project
python customvision/training.py

export $(grep -v '^#' .env | xargs)

# Export model
python customvision/export.py

# export ENDPOINT_NAME=endpt-moe-`echo $RANDOM`
export ACR_NAME=$(az ml workspace show --query container_registry -o tsv | cut -d'/' -f9-)

echo "Building Docker image"
# <set_base_path_and_copy_assets>
export BASE_PATH="triton"
cp $BASE_PATH/triton-cc-deployment.yml $BASE_PATH/deployment.yaml
cp $BASE_PATH/triton-cc-endpoint.yml $BASE_PATH/endpoint.yaml
sed -i "s/{{acr_name}}/$ACR_NAME/g;\
        s/{{ENDPOINT_NAME}}/$ENDPOINT_NAME/g;" $BASE_PATH/deployment.yaml
sed -i "s/{{MODEL_NAME}}/$MODEL_NAME/g;" $BASE_PATH/deployment.yaml
sed -i "s/{{ENDPOINT_NAME}}/$ENDPOINT_NAME/g;" $BASE_PATH/endpoint.yaml
# </set_base_path_and_copy_assets>

# <login_to_acr>
az acr login -n $ACR_NAME
# </login_to_acr> 

# <build_with_acr>
az acr build -t azureml-examples/triton-cc:latest -r $ACR_NAME -f $BASE_PATH/triton-cc.dockerfile $BASE_PATH
# </build_with_acr>

echo "Creating endpoint"
# <create_endpoint>
az ml online-endpoint create -f $BASE_PATH/endpoint.yaml
# </create_endpoint>

echo "Creating deployment"
# <create_deployment>
az ml online-deployment create --endpoint-name $ENDPOINT_NAME -f $BASE_PATH/deployment.yaml --all-traffic
# </create_deployment> 

# Check if deployment was successful
echo "Checking status of deployment"
deploy_status=`az ml online-deployment show --name triton-cc-deployment --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
echo $deploy_status
attempts=1
while [[ $deploy_status != "Succeeded" ]]
do
  sleep 5
  deploy_status=`az ml online-deployment show --name triton-cc-deployment --endpoint $ENDPOINT_NAME --query "provisioning_state" -o tsv`
  echo $deploy_status
  if [[ $attempts -eq 10 ]]
  then
    echo "Deployment failed"
    exit 1
  fi
  attempts=$((attempts+1))
done

echo "Testing deployment"
# Get scoring url
echo "Getting scoring url..."
SCORING_URL=$(az ml online-endpoint show -n $ENDPOINT_NAME --query scoring_uri -o tsv)
echo "Scoring url is $SCORING_URL"

# Get auth token
echo "Getting auth token..."
AUTH_TOKEN=$(az ml online-endpoint get-credentials -n $ENDPOINT_NAME --query accessToken -o tsv)

# <test_online_endpoint>
python $BASE_PATH/scoring/triton_scoring.py --base_url $SCORING_URL --token $AUTH_TOKEN --image_filename ./data/test/test_image.jpg --model_name $MODEL_NAME --labels $BASE_PATH/scoring/labels.txt
# </test_online_endpoint>

# Command below will compare performance of Triton server on AzureML to that of Azure Custom Vision predictions
# python $BASE_PATH/scoring/scoring_evaluation.py --base_url $SCORING_URL --token $AUTH_TOKEN --image_filename ./data/test/test_image.jpg --model_name $MODEL_NAME --labels $BASE_PATH/scoring/labels.txt

# <delete_online_endpoint>
# az ml online-endpoint delete -y -n $ENDPOINT_NAME
# </delete_online_endpoint>
