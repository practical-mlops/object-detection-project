import kfp
import kfp.components as comp
import kfp.dsl as dsl

from data_preparation.download_dataset import download_dataset
from data_preparation.split_dataset import split_dataset
from training_and_validation.train_model import train_model
from training_and_validation.validate_model import validate_model

download_task = comp.create_component_from_func(download_dataset,
                                                packages_to_install=["requests", "boto3", "tqdm"])

split_dataset_task = comp.create_component_from_func(split_dataset,
                                                     packages_to_install=["minio", "scikit-learn"])

train_model_task = comp.create_component_from_func(train_model,
                                                   base_image="ultralytics/ultralytics:latest",
                                                   packages_to_install=["minio", "tqdm", "pyyaml"])

validate_model_task = comp.create_component_from_func(validate_model,
                                                      base_image="ultralytics/ultralytics:latest")


@dsl.pipeline(name="YOLO Object Detection Pipeline", description="YOLO Object Detection Pipeline")
def pipeline(epochs: int = 1,
             batch: int = 8,
             source_bucket="dataset",
             random_state=42,
             yolo_model_name="yolov8n_custom.pt"):

    download_op = download_task(bucket_name=source_bucket)
    split_dataset_op = split_dataset_task(bucket_name=source_bucket,
                                          random_state=random_state).after(download_op)

    train_model_op = train_model_task(
        epochs=epochs,
        batch=batch,
        source_bucket="dataset",
        x_train=split_dataset_op.outputs['x_train'],
        y_train=split_dataset_op.outputs['y_train'],
        x_test=split_dataset_op.outputs['x_test'],
        y_test=split_dataset_op.outputs['y_test'],
        x_val=split_dataset_op.outputs['x_val'],
        y_val=split_dataset_op.outputs['y_val'],
    )

    validate_model_op = validate_model_task(
        project_path=train_model_op.outputs['project'],
        yolo_model_name=yolo_model_name,
    )


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='training_and_validation_pipeline.yaml',
    )
