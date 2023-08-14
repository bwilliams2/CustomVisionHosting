#%%
import os

from dotenv import load_dotenv
from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from msrest.authentication import ApiKeyCredentials

load_dotenv("../.env")
publish_iteration_name = os.environ["VISION_ITERATION_ID"]
base_image_location = os.path.join(os.path.dirname(__file__), "data")

ENDPOINT = os.environ["VISION_TRAINING_ENDPOINT"]
PREDICTION_ENDPOINT = os.environ["VISION_PREDICTION_ENDPOINT"]
PROJECT_ID = os.environ["VISION_PROJECT_ID"]
training_key = os.environ["VISION_TRAINING_KEY"]
prediction_key = os.environ["VISION_PREDICTION_KEY"]
prediction_resource_id = os.environ["VISION_PREDICTION_RESOURCE_ID"]


prediction_credentials = ApiKeyCredentials(
    in_headers={"Prediction-key": prediction_key}
)


predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, prediction_credentials)

#%%

# Now there is a trained endpoint that can be used to make a prediction

# Open the sample image and get back the prediction results.
with open(
    os.path.join(base_image_location, "test", "test_image.jpg"), mode="rb"
) as test_data:
    results = predictor.detect_image(PROJECT_ID, publish_iteration_name, test_data)

# Display the results.
for prediction in results.predictions:
    print(
        "\t"
        + prediction.tag_name
        + ": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f}, bbox.width = {3:.2f}, bbox.height = {4:.2f}".format(
            prediction.probability * 100,
            prediction.bounding_box.left,
            prediction.bounding_box.top,
            prediction.bounding_box.width,
            prediction.bounding_box.height,
        )
    )
