
from kfp.components import InputPath, OutputPath


def train_model(epochs: int,
                batch: int,
                source_bucket: str,
                yolo_model_name: str,
                x_train_file: InputPath(str),
                y_train_file: InputPath(str),
                x_test_file: InputPath(str),
                y_test_file: InputPath(str),
                x_val_file: InputPath(str),
                y_val_file: InputPath(str),
                data_yaml_path: OutputPath(str),
                ):
    from minio import Minio
    from minio.error import S3Error
    from tqdm import tqdm
    import os
    import yaml

    def download_from_minio(source_bucket, source_object, minio_client, download_path):
        try:
            # Download the file from MinIO
            minio_client.fget_object(source_bucket, source_object, download_path)
        except S3Error as err:
            print(f"Error downloading {source_object}: {err}")

    minio_client = Minio(
        'minio-service.kubeflow:9000',
        access_key='minio',
        secret_key='minio123',
        secure=False
    )

    # Create local directories.
    for splits in ["train", "test", "val"]:
        for x in ["images", "labels"]:
            os.makedirs(f"/dataset/{splits}/{x}", exist_ok=True)

    Xs = [x_train_file, x_test_file, x_val_file]
    Ys = [y_train_file, y_test_file, y_val_file]

    for i, splits in enumerate(["train", "test", "val"]):
        # Download image
        with open(Xs[i], "r") as f:
            for source_object in tqdm(f.readlines()):
                source_object = source_object.strip()
                download_path = f"/dataset/{splits}/images/{os.path.basename(source_object)}"
                download_from_minio(source_bucket, source_object, minio_client, download_path)

        # Download label
        with open(Ys[i], "r") as f:
            for source_object in f.readlines():
                source_object = source_object.strip()
                download_path = f"/dataset/{splits}/labels/{os.path.basename(source_object)}"
                download_from_minio(source_bucket, source_object, minio_client, download_path)

    data = {
        'path': '/dataset/',
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
    # from ultralytics import settings
    #
    # # Update a setting
    # settings.update({'mlflow': True})
    #
    # # Reset settings to default values
    # settings.reset()

    model = YOLO('yolov8n.pt')

    results = model.train(
        data=data_yaml_full_path,
        imgsz=640,
        epochs=epochs,
        batch=batch,
        project="/mnt/pipeline",
        name=yolo_model_name,
   )


