import json
import shutil

import PIL
import bentoml
import logging
from ultralytics import YOLO
from bentoml.io import Image
from bentoml.io import JSON
import os
from metrics import confidence_histogram

# Create a stream handler
ch = logging.StreamHandler()

# Set a format for the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Get the BentoML logger
bentoml_logger = logging.getLogger("bentoml")

# Add the handler to the BentoML logger
bentoml_logger.addHandler(ch)
bentoml_logger.setLevel(logging.DEBUG)


class YOLOv8Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model = YOLO("yolov8_custom.pt")
        self.model.cpu()

    @bentoml.Runnable.method(batchable=False)
    def inference(self, input_img):
        results = self.model(input_img)[0]
        if len(results) == 0:
            bentoml_logger.error("Error while processing")
            return {"status": "failed"}
        response = json.loads(results[0].tojson())
        confidence_histogram.observe(response[0]["confidence"])
        return response

    @bentoml.Runnable.method(batchable=False)
    def render(self, input_img):
        result = self.model(input_img, save=True, project=os.getcwd())
        return PIL.Image.open(os.path.join(result[0].save_dir, result[0].path))


yolo_v8_runner = bentoml.Runner(YOLOv8Runnable)

svc = bentoml.Service("yolo_v8", runners=[yolo_v8_runner])


@svc.api(input=Image(), output=JSON())
async def invocation(input_img):
    ret = await yolo_v8_runner.inference.async_run([input_img])
    return ret


@svc.api(input=Image(), output=Image())
async def render(input_img):
    ret = await yolo_v8_runner.render.async_run([input_img])
    return ret
