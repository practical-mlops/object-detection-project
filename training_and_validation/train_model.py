from kfp.components import InputPath, OutputPath


def train_model(epochs: int,
                batch: int,
                source_bucket: str,
                x_train_file: InputPath(str),
                y_train_file: InputPath(str),
                x_test_file: InputPath(str),
                y_test_file: InputPath(str),
                x_val_file: InputPath(str),
                y_val_file: InputPath(str),
                project_path: OutputPath(str),
                yolo_model_name: str = "yolov8n_custom.pt"):
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
                print(download_path)

        # Download label
        with open(Ys[i], "r") as f:
            for source_object in f.readlines():
                source_object = source_object.strip()
                download_path = f"/dataset/{splits}/labels/{os.path.basename(source_object)}"
                download_from_minio(source_bucket, source_object, minio_client, download_path)
                print(download_path)

    data = {
        'path': '/dataset/',
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {
            0: 'id_card'
        }
    }

    file_path = 'custom_data.yaml'
    try:
        with open(file_path, 'w') as file:
            yaml.dump(data, file)
        print("YAML file has been written successfully.")
    except Exception as e:
        print(f"Error writing YAML file: {e}")

    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    results = model.train(
        data='custom_data.yaml',
        imgsz=640,
        epochs=epochs,
        batch=batch,
        project=project_path,
        name=yolo_model_name,
    )

