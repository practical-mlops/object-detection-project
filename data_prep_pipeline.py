import kfp
import kfp.components as comp
import kfp.dsl as dsl

from data_preparation.download_dataset import download_dataset
from data_preparation.output_file_contents import output_file_contents
from data_preparation.split_dataset import split_dataset

download_task = comp.create_component_from_func(download_dataset,
                                                packages_to_install=["requests", "boto3", "tqdm"])

split_dataset_task = comp.create_component_from_func(split_dataset,
                                                     packages_to_install=["minio", "scikit-learn"])

output_file_contents_task = comp.create_component_from_func(output_file_contents)


@dsl.pipeline(name="data_preparation_pipeline")
def pipeline(random_state=42):
    download_op = download_task(bucket_name="dataset")
    split_dataset_op = split_dataset_task(bucket_name="dataset", random_state=random_state).after(download_op)
    _ = output_file_contents_task(
        split_dataset_op.outputs['x_val']
    )
    _ = output_file_contents_task(
        split_dataset_op.outputs['y_val']
    )


if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='dataprep_pipeline.yaml')
