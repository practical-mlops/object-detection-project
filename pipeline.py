import kfp
import kfp.components as comp
import kfp.dsl as dsl

from data_preparation.download_dataset import download_dataset
from data_preparation.split_dataset import split_dataset
from train_model import train_model

download_task = comp.create_component_from_func(download_dataset,
                                                packages_to_install=["requests", "boto3", "tqdm"])

split_dataset_task = comp.create_component_from_func(split_dataset,
                                                     packages_to_install=["minio", "scikit-learn"])

train_model_task = comp.create_component_from_func(train_model,
                                                   base_image="ultralytics/ultralytics:latest",
                                                   packages_to_install=["minio", "tqdm", "pyyaml"])


@dsl.pipeline(name="Convert to COCO pipeline")
def pipeline(epochs: int = 1, batch: int = 8, random_state=42):
    # TODO: Turn this back on later
    # download_op = download_task(bucket_name="dataset")
    split_dataset_op = split_dataset_task(bucket_name="dataset", random_state=random_state)

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


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='pipeline.yaml')
