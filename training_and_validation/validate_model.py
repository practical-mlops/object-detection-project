
from kfp.components import OutputPath, InputPath


def validate_model(data_yaml_path: InputPath(str),
                   yolo_model_name: str,
                   source_bucket: str,
                   x_val_file: InputPath(str),
                   y_val_file: InputPath(str),
                   mlpipeline_metrics_path: OutputPath("Metrics")):
    import os
    from minio import S3Error, Minio
    from tqdm import tqdm

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
    for x in ["images", "labels"]:
        os.makedirs(f"/dataset/val/{x}", exist_ok=True)

    X = x_val_file
    Y = y_val_file

    with open(X, "r") as f:
        for source_object in tqdm(f.readlines()):
            source_object = source_object.strip()
            download_path = f"/dataset/val/images/{os.path.basename(source_object)}"
            download_from_minio(source_bucket, source_object, minio_client, download_path)

    # Download label
    with open(Y, "r") as f:
        for source_object in f.readlines():
            source_object = source_object.strip()
            download_path = f"/dataset/val/labels/{os.path.basename(source_object)}"
            download_from_minio(source_bucket, source_object, minio_client, download_path)

    from ultralytics import YOLO
    import json
    import os

    weights_path = os.path.join("mnt", "pipeline", yolo_model_name, "weights", "best.pt")
    print(f"Loading weights at: {weights_path}")

    model = YOLO(weights_path)

    metrics = model.val(
        data=os.path.join(data_yaml_path, "data.yaml")
    )
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
