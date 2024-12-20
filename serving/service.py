import json
import shutil

import PIL
import bentoml
from ultralytics import YOLO
from bentoml.io import Image
from bentoml.io import JSON
import os


class YOLOv8Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model = YOLO("yolov8_custom.pt")
        self.model.cpu()

    @bentoml.Runnable.method(batchable=False)
    def inference(self, input_img):
        results = self.model(input_img)[0]

        return json.loads(results[0].tojson())

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
