from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Artifact, Metrics


@dsl.component(
    packages_to_install=["requests", "boto3", "tqdm"],
    base_image="python:3.11"
)
def download_dataset(output_dataset: Output[Dataset], output_dir: str = "DATASET"):
    import os
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
    extraction_path = os.path.join(output_dataset.path, output_dir)
    os.makedirs(extraction_path, exist_ok=True)

    with tarfile.open(downloaded_file, 'r:gz') as tar:
        tar.extractall(extraction_path)

@dsl.component(
    packages_to_install=["scikit-learn"],
    base_image="python:3.11"
)
def split_dataset(
        random_state: int,
        input_dataset: Input[Dataset],
        train_dataset: Output[Dataset],  # Changed output names
        validation_dataset: Output[Dataset],
        test_dataset: Output[Dataset]
):
    import os
    import glob
    import shutil
    from sklearn.model_selection import train_test_split

    BASE_PATH = "MINIDATA"
    # BASE_PATH = "DATA"

    # Adjust paths to use input_dataset.path
    images = list(glob.glob(os.path.join(input_dataset.path, "DATASET", BASE_PATH, "images", "**")))
    labels = list(glob.glob(os.path.join(input_dataset.path, "DATASET", BASE_PATH, "labels", "**")))

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=1 - train_ratio,
        random_state=random_state
    )

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(
        x_test,
        y_test,
        test_size=test_ratio / (test_ratio + validation_ratio),
        random_state=random_state
    )

    # Create output directories for each split
    for dataset_output, x_files, y_files in [
        (train_dataset, x_train, y_train),
        (validation_dataset, x_val, y_val),
        (test_dataset, x_test, y_test)
    ]:
        os.makedirs(os.path.join(dataset_output.path, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_output.path, "labels"), exist_ok=True)

        # Move files
        for src in x_files:
            dest = os.path.join(dataset_output.path, "images", os.path.basename(src))
            shutil.copy2(src, dest)

        for src in y_files:
            dest = os.path.join(dataset_output.path, "labels", os.path.basename(src))
            shutil.copy2(src, dest)

TRAINING_SCRIPT = '''
import os
import yaml
import shutil
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--train-path', required=True, help='Path to training dataset')
    parser.add_argument('--val-path', required=True, help='Path to validation dataset')
    parser.add_argument('--test-path', required=True, help='Path to test dataset')
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
        'train': os.path.join(args.train_path, "images"),
        'val': os.path.join(args.val_path, "images"),
        'test': os.path.join(args.test_path, "images"),
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
        train_dataset: dsl.Input[dsl.Dataset],
        validation_dataset: dsl.Input[dsl.Dataset],
        test_dataset: dsl.Input[dsl.Dataset],
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
                --train-path "$0" \
                --val-path "$1" \
                --test-path "$2" \
                --epochs "$3" \
                --batch "$4" \
                --model-name "$5" \
                --model-output "$6" \
                --data-yaml "$7"
            ''',
            train_dataset.path,
            validation_dataset.path,
            test_dataset.path,
            epochs,
            batch,
            yolo_model_name,
            model_output.path,
            data_yaml.path
        ]
    )


@dsl.component(
    base_image="ultralytics/ultralytics:8.0.194-cpu",
    packages_to_install=["minio", "tqdm"]
)
def validate_model(
        data_yaml: Input[Artifact],
        model: Input[Model],
        validation_dataset: Input[Dataset],
        metrics: Output[Metrics]
):
    from ultralytics import YOLO
    import os

    # Print paths for debugging
    print(f"Data YAML path: {data_yaml.path}")
    print(f"Model path: {model.path}")
    print(f"Validation dataset path: {validation_dataset.path}")

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
    # Download dataset
    download_op = download_dataset()

    # Split dataset
    split_op = split_dataset(
        random_state=random_state,
        input_dataset=download_op.outputs["output_dataset"]
    )

    # Train model using split data
    train_op = train_model(
        epochs=epochs,
        batch=batch,
        yolo_model_name=yolo_model_name,
        train_dataset=split_op.outputs["train_dataset"],
        validation_dataset=split_op.outputs["validation_dataset"],
        test_dataset=split_op.outputs["test_dataset"]
    )

    # Validate model
    validate_op = validate_model(
        data_yaml=train_op.outputs["data_yaml"],
        model=train_op.outputs["model_output"],
        validation_dataset=split_op.outputs["validation_dataset"],

    )


if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='training_and_validation_pipeline.yaml'
    )
