
from kfp.components import OutputPath


def train_model(epochs: int,
                batch: int,
                yolo_model_name: str,
                data_yaml_path: OutputPath(str),
                ):

    import os
    import yaml

    data = {
        'path': '/mnt/pipeline/DATASET/DATA',
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'id_card'
        }
    }

    data_yaml_full_path = os.path.join(data_yaml_path, "data.yaml")
    from pathlib import Path
    Path(data_yaml_path).mkdir(parents=True, exist_ok=True)

    try:
        with open(data_yaml_full_path, 'w') as file:
            yaml.dump(data, file)
        print("YAML file has been written successfully.")
    except Exception as e:
        print(f"Error writing YAML file: {e}")

    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')

    results = model.train(
        data=data_yaml_full_path,
        imgsz=640,
        epochs=epochs,
        batch=batch,
        project="/mnt/pipeline",
        name=yolo_model_name,
   )


