
from kfp.components import OutputPath


def validate_model(project_path: str,
                   yolo_model_name: str,
                   mlpipeline_metrics_path: OutputPath("Metrics")):
    from ultralytics import YOLO
    import json
    import os

    weights_path = os.path.join(project_path, yolo_model_name, "weights", "best.pt")
    print(f"Loading weights at: {weights_path}")

    model = YOLO(weights_path)

    metrics = model.val()
    metrics_dict = {
        "metrics": [
            {
                "name": "map50-95",
                "numberValue": metrics.box.map,
                "format": "RAW",
            },
            {
                "name": "map50",
                "numberValue": metrics.box.map50,
                "format": "RAW",
            },
            {
                "name": "map75",
                "numberValue": metrics.box.map75,
                "format": "RAW",
            }]
    }

    with open(mlpipeline_metrics_path, "w") as f:
        json.dump(metrics_dict, f)