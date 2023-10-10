import PIL
import bentoml
from ultralytics import YOLO
from bentoml.io import Image
from bentoml.io import PandasDataFrame


class YOLOv8Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        # TODO: How can we pass in the path here?
        self.model = YOLO("yolov8_custom.pt")

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def inference(self, input_imgs):
        # Return predictions only
        results = self.model(input_imgs)
        return results.pandas().xyxy

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def render(self, input_imgs):
        # Return images with boxes and labels
        results = self.model(input_imgs)[0]

        im_array = results.plot()  # plot a BGR numpy array of predictions
        im = PIL.Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save('results.jpg')
        return im



yolo_v8_runner = bentoml.Runner(YOLOv8Runnable, max_batch_size=30)

svc = bentoml.Service("yolo_v8", runners=[yolo_v8_runner])


@svc.api(input=Image(), output=PandasDataFrame())
async def invocation(input_img):
    batch_ret = await yolo_v8_runner.inference.async_run([input_img])
    return batch_ret[0]


@svc.api(input=Image(), output=Image())
async def render(input_img):
    batch_ret = await yolo_v8_runner.render.async_run([input_img])
    return batch_ret[0]
