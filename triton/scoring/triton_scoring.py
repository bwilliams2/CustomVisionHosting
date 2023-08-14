# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import tritonclient.http as httpclient
import gevent
import os
import sys
import click
import math
import onnx
import numpy as np
from PIL import Image, ImageDraw
from object_detection import ObjectDetection
import tritonclient.http as httpclient
import tempfile

MODEL_FILENAME = 'model.onnx'
LABELS_FILENAME = 'labels.txt'


class TritonObjectDetection(ObjectDetection):
    """Object Detection class for ONNX Runtime"""
    def __init__(self, labels_file):
        with open(labels_file) as f:
            labels = f.readlines()
        labels=[x.strip() for x in labels]
        super(TritonObjectDetection, self).__init__(labels)

    def preprocess(self, image, new_width, new_height):
        image = image.convert("RGB") if image.mode != "RGB" else image
        image = self._update_orientation(image)

        new_width = 32 * math.ceil(new_width / 32)
        new_height = 32 * math.ceil(new_height / 32)
        image = image.resize((new_width, new_height))
        return image

    def predict_image(self, image, model_name, scoring_uri, aml_token, model_version="1", verbose=False):
        url = scoring_uri[8:]
        with httpclient.InferenceServerClient(
            url=url,
            ssl=True, 
            ssl_context_factory=gevent.ssl._create_default_https_context,
        ) as client:
            headers = {}
            headers["Authorization"] = f"Bearer {aml_token}"

            health_ctx = client.is_server_ready(headers=headers)
            if verbose:
                print("Is server ready -{}".format(health_ctx))
        
            # Check status of model
            status_ctx = client.is_model_ready(model_name, model_version, headers)
            if verbose:
                print("Is model ready - {}".format(status_ctx))

            metadata=client.get_model_metadata(model_name, "1", headers=headers)
            dtype = metadata["inputs"][0]["datatype"]
            shape = metadata["inputs"][0]["shape"]
            name=metadata["inputs"][0]["name"]
            
            input = [httpclient.InferInput(name, [1, *shape[1:]], dtype)]
            image = Image.open(image)
            self.DEFAULT_INPUT_SIZE = shape[-2] * shape[-1]
            inputs = self.preprocess(image, shape[-2], shape[-1])
            inputs = np.array(inputs, dtype=np.float32)[np.newaxis,:,:,(2,1,0)] # RGB -> BGR
            inputs = np.ascontiguousarray(np.rollaxis(inputs, 3, 1))
            input[0].set_data_from_numpy(inputs)
            resp = client.infer(model_name, input, headers=headers)
            res_json = resp.get_response()
            output_name = res_json["outputs"][0]["name"]
            outputs = resp.as_numpy(output_name)
            prediction_outputs = np.squeeze(outputs).transpose((1,2,0)).astype(np.float32)
        return self.postprocess(prediction_outputs)

@click.command()
@click.option("--base_url", help="scoring url")
@click.option("--token", help="token for authorization")
@click.option("--image_filename", help="image")
@click.option("--model_name", help="model name")
@click.option("--labels", help="labels file location")
@click.option("--verbose", help="verbose", default=False)
def main(base_url, token, image_filename, model_name, labels, verbose): 
    triton = TritonObjectDetection(labels)
    outputs = triton.predict_image(image_filename, model_name, base_url, token, verbose=verbose)
    if verbose:
        print(outputs)

if __name__ == "__main__":
    main()
    
