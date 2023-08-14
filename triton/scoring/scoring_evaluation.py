from triton_scoring import TritonObjectDetection
import click
import datetime
import numpy as np
import os
from dotenv import load_dotenv
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from pathlib import Path
env_file = Path(__file__).parent.parent.parent.joinpath(".env")
load_dotenv(env_file, override=True)


ENDPOINT = os.environ["VISION_TRAINING_ENDPOINT"]
PREDICTION_ENDPOINT = os.environ["VISION_PREDICTION_ENDPOINT"]
PROJECT_ID = os.environ["VISION_PROJECT_ID"]
training_key = os.environ["VISION_TRAINING_KEY"]
prediction_key = os.environ["VISION_PREDICTION_KEY"]
prediction_resource_id = os.environ["VISION_PREDICTION_RESOURCE_ID"]
publish_iteration_name = os.environ["VISION_PUBLISH_ITERATION_NAME"]


prediction_credentials = ApiKeyCredentials(
    in_headers={"Prediction-key": prediction_key}
)


predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, prediction_credentials)


@click.command()
@click.option("--base_url", help="scoring url")
@click.option("--token", help="token for authorization")
@click.option("--image_filename", help="image")
@click.option("--model_name", help="model name")
@click.option("--labels", help="model name")
def main(base_url, token, image_filename, model_name, labels): 
    N_THREAD = 1
    
    for N_REQ in [2, 10, 100]:
        print(
            f"TIME FOR {N_REQ} REQUESTS OVER {N_THREAD} THREADS (stats exclude the first request):"
        )

        def make_triton_request():
            tic = datetime.datetime.now()
            triton = TritonObjectDetection(labels)
            outputs = triton.predict_image(image_filename, model_name, base_url, token)
            toc = datetime.datetime.now()
            return (toc - tic).total_seconds()
        
        def make_cv_request():
            tic = datetime.datetime.now()
            with open(image_filename, mode="rb") as test_data:
                results = predictor.detect_image(PROJECT_ID, publish_iteration_name, test_data)
            toc = datetime.datetime.now()
            return (toc - tic).total_seconds()

        print("Triton hosted model:")
        total_tic = datetime.datetime.now()
        times = list([make_triton_request() for _ in range(N_REQ)])
        total_toc = datetime.datetime.now()
        print(f"mean: {np.mean(times[1:])}, std: {np.std(times[1:])}")
        print(times)
        print(f"total time: {total_toc - total_tic}")

        print()
        print("Custom Vision prediction:")
        total_tic = datetime.datetime.now()
        times = list([make_cv_request() for _ in range(N_REQ)])
        total_toc = datetime.datetime.now()
        print(f"mean: {np.mean(times[1:])}, std: {np.std(times[1:])}")
        print(times)
        print(f"total time: {total_toc - total_tic}")
        print("")

    
if __name__ == "__main__":
    main()
    

