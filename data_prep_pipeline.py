from kfp import dsl
from kfp.dsl import Input, Output, Dataset


@dsl.component(
    packages_to_install=["requests", "boto3", "tqdm"],
    base_image="python:3.11"
)
def download_dataset(output_dataset: Output[Dataset], output_dir: str = "DATASET"):
    import os
    import requests
    import tarfile
    from tqdm import tqdm

    # FULL Dataset
    url = "https://manning.box.com/shared/static/34dbdkmhahuafcxh0yhiqaf05rqnzjq9.gz"
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



@dsl.component
def output_file_contents(dataset: Input[Dataset]):
    import os

    def list_files(start_path):
        for root, dirs, files in os.walk(start_path):
            level = root.replace(start_path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print(f'{indent}{os.path.basename(root)}/')
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                print(f'{sub_indent}{f}')

    print(f"Contents of {dataset.path}:")
    list_files(dataset.path)


@dsl.pipeline(
    name="data_preparation_pipeline",
    description="Pipeline for preparing and splitting dataset"
)
def pipeline(random_state: int = 42):
    # Download the dataset
    download_op = download_dataset()

    # Split the dataset
    split_op = split_dataset(
        random_state=random_state,
        input_dataset=download_op.outputs["output_dataset"]
    )

    # Output the contents of both validation sets
    output_file_contents(dataset=split_op.outputs["train_dataset"])


if __name__ == '__main__':
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='dataprep_pipeline.yaml'
    )
