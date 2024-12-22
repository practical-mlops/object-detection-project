from Dataset import load_dataset
from deepchecks.vision.checks import ImageDatasetDrift

train_dataset = load_dataset(train=True, object_type="VisionData")
test_dataset = load_dataset(train=False, object_type="VisionData")


check_result = ImageDatasetDrift().run(train_dataset, test_dataset)
check_result.save_as_html("deepcheck_vision_drift_check.html")
print(check_result.value)
