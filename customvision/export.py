# %%
import os
import shutil
import requests
import time
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateBatch,
    ImageFileCreateEntry,
    Region,
)
from msrest.authentication import ApiKeyCredentials


load_dotenv(Path(__file__).parent.parent.joinpath(".env"))
ENDPOINT = os.environ["VISION_TRAINING_ENDPOINT"]
training_key = os.environ["VISION_TRAINING_KEY"]
prediction_key = os.environ["VISION_PREDICTION_KEY"]
prediction_resource_id = os.environ["VISION_PREDICTION_RESOURCE_ID"]
project_id = os.environ["VISION_PROJECT_ID"]
iteration_id = os.environ["VISION_ITERATION_ID"]

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)
platform = "ONNX"
flavor = "ONNX16"

try:
    export = trainer.export_iteration(
        project_id, iteration_id, platform, flavor, raw=False
    )
except Exception as e:
    export = None
    pass

if export is not None:
    while export.status == "Exporting":
        exports = trainer.get_exports(project_id, iteration_id)
        for e in exports:
            if e.platform == export.platform and e.flavor == export.flavor:
                export = e
                break
        print("Waiting 10 seconds...")
        time.sleep(10)

# %%
exports = trainer.get_exports(project_id, iteration_id)
# Locate the export for this iteration and check its status
for e in exports:
    if e.platform == platform:
        export = e
        break
print("Export status is: ", export.status)

if export.status == "Done":
    # Success, now we can download it
    export_file = requests.get(export.download_uri)
    with open("export.zip", "wb") as file:
        file.write(export_file.content)


with zipfile.ZipFile("export.zip", "r") as zip_ref:
    zip_ref.extractall("export")

model_name = os.getenv("MODEL_NAME", "cv_model")
model_version = os.getenv("MODEL_VERSION", "1")

triton_path = Path(__file__).parent.parent.joinpath("triton")
model_path = triton_path.joinpath(f"models/{model_name}/{model_version}")
model_path.mkdir(parents=True, exist_ok=True)
scoring_path = triton_path.joinpath("scoring")

shutil.copyfile("export/model.onnx", str(model_path.joinpath("model.onnx")))
shutil.copyfile("export/python/object_detection.py", str(scoring_path.joinpath("object_detection.py")))
shutil.copyfile("export/labels.txt", str(scoring_path.joinpath("labels.txt")))