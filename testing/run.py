import glob
import cv2
from ultralytics import YOLO

# model = YOLO("yolov8n.pt") # <-- default model
# for file in glob.glob("/Users/benjamintan/workspace/object-detection-project/testing/sample/**.png"):
#     result = model(cv2.imread(file))
#     res_plotted = result[0].plot()
#     cv2.imshow("result", res_plotted)
#     cv2.waitKey(0)


model = YOLO("/Users/benjamintan/workspace/object-detection-project/testing/weights/best.pt")
for file in glob.glob("/Users/benjamintan/workspace/object-detection-project/testing/sample/**.png"):
    result = model(cv2.imread(file))
    res_plotted = result[0].plot()
    cv2.imshow("result", res_plotted)
    cv2.waitKey(0)
