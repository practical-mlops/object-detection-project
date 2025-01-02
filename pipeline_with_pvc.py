from kfp import dsl
from kfp import kubernetes
from kfp.dsl import Input, Output, Model, Artifact, Metrics


@dsl.component(
    packages_to_install=["requests", "boto3", "tqdm"],
    base_image="python:3.11"
)
def download_dataset():
    import requests
    import tarfile
    from tqdm import tqdm

    # MINI Dataset
    url = "https://manning.box.com/shared/static/coiv3n2t5t0v42xgfhlsfi8bhvd7b441.gz"
    downloaded_file = "DATASET.gz"

    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    progress_bar = tqdm(total=file_size, unit="B", unit_scale=True)

    with open(downloaded_file, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            progress_bar.update(len(chunk))
            file.write(chunk)

    # Extract to the output directory
    # NOTE: Not needed anymore
    # extraction_path = os.path.join(output_dataset.path, output_dir)
    # os.makedirs(extraction_path, exist_ok=True)

    with tarfile.open(downloaded_file, 'r:gz') as tar:
        tar.extractall("/data/DATASET")


@dsl.component(
    packages_to_install=["scikit-learn"],
    base_image="python:3.11"
)
def split_dataset(random_state: int):
    import os
    import glob
    import shutil
    from sklearn.model_selection import train_test_split

    BASE_PATH = "MINIDATA"

    images = list(glob.glob(os.path.join("/data/DATASET", BASE_PATH, "images", "**")))
    labels = list(glob.glob(os.path.join("/data/DATASET", BASE_PATH, "labels", "**")))

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=1 - train_ratio, random_state=random_state
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test,
        test_size=test_ratio / (test_ratio + validation_ratio),
        random_state=random_state
    )

    # Create directories for splits
    for split in ["train", "test", "val"]:
        for category in ["images", "labels"]:
            os.makedirs(os.path.join("/data", split, category), exist_ok=True)

    def move_files(files, split, category):
        for src in files:
            dest = os.path.join("/data", split, category, os.path.basename(src))
            shutil.copy2(src, dest)

    # Move files to respective splits
    move_files(x_train, "train", "images")
    move_files(y_train, "train", "labels")
    move_files(x_val, "val", "images")
    move_files(y_val, "val", "labels")
    move_files(x_test, "test", "images")
    move_files(y_test, "test", "labels")


TRAINING_SCRIPT = '''
import os
import yaml
import shutil
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--batch', type=int, required=True, help='Batch size')
    parser.add_argument('--model-name', required=True, help='Name of the model')
    parser.add_argument('--model-output', required=True, help='Path to save the model')
    parser.add_argument('--data-yaml', required=True, help='Path to save data.yaml')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define data configuration using absolute paths
    data = {
        'train': '/data/train/images',
        'val': '/data/val/images',
        'test': '/data/test/images',
        'nc': 1,
        'names': {
            0: 'id_card'
        }
    }

    # Write data.yaml
    data_yaml_path = os.path.join(args.data_yaml, "data.yaml")
    os.makedirs(os.path.dirname(data_yaml_path), exist_ok=True)

    print(f"Writing data configuration to {data_yaml_path}")
    print("Data YAML contents:")
    print(yaml.dump(data))

    with open(data_yaml_path, 'w') as file:
        yaml.dump(data, file)

    # Train model
    model = YOLO('yolov8n.pt')
    results = model.train(
        data=data_yaml_path,
        imgsz=640,
        epochs=args.epochs,
        batch=args.batch,
        project=os.path.dirname(args.model_output),
        name=args.model_name
    )

    # Save best model
    best_model_source = os.path.join(os.path.dirname(args.model_output), args.model_name, "weights", "best.pt")
    best_model_dest = os.path.join(args.model_output, "best.pt")
    os.makedirs(os.path.dirname(best_model_dest), exist_ok=True)
    shutil.copy2(best_model_source, best_model_dest)

if __name__ == "__main__":
    main()
'''


@dsl.container_component
def train_model(
        epochs: int,
        batch: int,
        yolo_model_name: str,
        model_output: dsl.Output[dsl.Model],
        data_yaml: dsl.Output[dsl.Artifact]
):
    return dsl.ContainerSpec(
        image='python:3.11-slim',
        command=['bash', '-c'],
        args=[
            f'''
            # Install system dependencies
            apt-get update && \
            apt-get install -y --no-install-recommends \
                libgl1-mesa-glx \
                libglib2.0-0 \
                && rm -rf /var/lib/apt/lists/* && \
            
            # Install Python packages
            pip install --no-cache-dir \
                ultralytics \
                torch \
                opencv-python-headless==4.8.1.78 \
                minio \
                tqdm \
                pyyaml && \
            
            # Write training script
            cat << 'EOF' > /train.py
{TRAINING_SCRIPT}
EOF

            # Execute the training script
            python3 /train.py \
                --epochs "$0" \
                --batch "$1" \
                --model-name "$2" \
                --model-output "$3" \
                --data-yaml "$4"
            ''',
            epochs,
            batch,
            yolo_model_name,
            model_output.path,
            data_yaml.path
        ]
    )


@dsl.component(
    base_image="ultralytics/ultralytics:8.3.56-cpu",
    packages_to_install=["minio", "tqdm"],
)
def validate_model(
        data_yaml: Input[Artifact],
        model: Input[Model],
        metrics: Output[Metrics]
):
    from ultralytics import YOLO
    import os

    # Print paths for debugging
    print(f"Data YAML path: {data_yaml.path}")
    print(f"Model path: {model.path}")

    # Load the trained model
    model_path = os.path.join(model.path, "best.pt")
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Run validation using the existing data.yaml
    validation_results = model.val(
        data=os.path.join(data_yaml.path, "data.yaml"),
        imgsz=640,
        batch=1,
        verbose=True
    )

    # Log metrics
    metrics.log_metric("map50-95", validation_results.box.map)
    metrics.log_metric("map50", validation_results.box.map50)
    metrics.log_metric("map75", validation_results.box.map75)


@dsl.pipeline(
    name="YOLO Object Detection Pipeline",
    description="YOLO Object Detection Pipeline"
)
def pipeline(
        epochs: int = 1,
        batch: int = 8,
        random_state: int = 42,
        yolo_model_name: str = "yolov8n_custom"
):
    # Create PVC
    pvc_name = 'yolo-pipeline-pvc'

    pvc = kubernetes.CreatePVC(
        pvc_name=pvc_name,
        access_modes=['ReadWriteOnce'],
        size='1Gi',
        storage_class_name='local-path',
    )
    # Download dataset
    download_op = download_dataset()

    # Split dataset
    split_op = split_dataset(
        random_state=random_state,
    ).after(download_op)

    # Train model using split data
    train_op = train_model(
        epochs=epochs,
        batch=batch,
        yolo_model_name=yolo_model_name,
    ).after(split_op)

    # Validate model
    validate_op = validate_model(
        data_yaml=train_op.outputs["data_yaml"],
        model=train_op.outputs["model_output"],
    ).after(train_op).set_caching_options(False)


    for op in [download_op, split_op, train_op, validate_op]:
        kubernetes.mount_pvc(
            op,
            pvc_name=pvc_name,
            mount_path='/data'
        )

    kubernetes.DeletePVC(pvc_name=pvc_name).after(validate_op)


if __name__ == '__main__':
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='training_and_validation_pipeline_with_pvc.yaml'
    )
