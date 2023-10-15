import kfp
import kfp.components as comp
import kfp.dsl as dsl
from kfp.onprem import mount_pvc

from data_preparation.download_dataset import download_dataset
from data_preparation.split_dataset import split_dataset
from training_and_validation.train_model import train_model
from training_and_validation.validate_model import validate_model

download_task = comp.create_component_from_func(
    download_dataset,
    packages_to_install=["requests", "boto3", "tqdm"])

split_dataset_task = comp.create_component_from_func(
    split_dataset,
    packages_to_install=["minio", "scikit-learn", "tqdm"])

train_model_task = comp.create_component_from_func(
    train_model,
    base_image="ultralytics/ultralytics:8.0.194-cpu",
    packages_to_install=["minio", "tqdm", "pyyaml"])

validate_model_task = comp.create_component_from_func(
    validate_model,
    base_image="ultralytics/ultralytics:8.0.194-cpu",
    packages_to_install=["minio", "tqdm"])


@dsl.pipeline(name="YOLO Object Detection Pipeline", description="YOLO Object Detection Pipeline")
def pipeline(epochs: int = 1,
             batch: int = 8,
             random_state: int = 42,
             yolo_model_name: str = "yolov8n_custom"):

    volume_op = dsl.VolumeOp(
        name="Create PVC",
        resource_name="pipeline-pvc",
        modes=dsl.VOLUME_MODE_RWO,
        size="20Gi",
    )

    download_op = download_task().apply(mount_pvc(
        volume_op.outputs["name"], 'local-storage', '/mnt/pipeline'))

    # split_dataset_op = split_dataset_task(bucket_name=source_bucket,
    #                                       random_state=random_state).after(download_op)

    split_dataset_op = split_dataset_task(
        random_state=random_state
    ).apply(mount_pvc(
        volume_op.outputs["name"], 'local-storage', '/mnt/pipeline')
    ).after(download_op)

    train_model_op = train_model_task(
        epochs=epochs,
        batch=batch,
        yolo_model_name=yolo_model_name
    ).apply(mount_pvc(
            volume_op.outputs["name"], 'local-storage', '/mnt/pipeline')
    ).after(split_dataset_op)

    validate_model_op = validate_model_task(
        data_yaml=train_model_op.outputs['data_yaml'],
        yolo_model_name=yolo_model_name,
    ).apply(
        mount_pvc(
            volume_op.outputs["name"], 'local-storage', '/mnt/pipeline'))


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='training_and_validation_pipeline.yaml',
    )
