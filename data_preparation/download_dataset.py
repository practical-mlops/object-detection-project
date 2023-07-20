from kfp.components import OutputPath


def download_dataset(bucket_name: str, output_file: OutputPath(str)):
    import boto3
    import os
    import requests
    import tarfile
    from tqdm import tqdm

    url = "https://manning.box.com/shared/static/34dbdkmhahuafcxh0yhiqaf05rqnzjq9.gz"

    output_dir = "DATASET"
    downloaded_file = "DATASET.gz"

    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    progress_bar = tqdm(total=file_size, unit="B", unit_scale=True)

    with open(downloaded_file, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            # Update the progress bar with the size of the downloaded chunk
            progress_bar.update(len(chunk))
            file.write(chunk)

    # Open the tar archive
    with tarfile.open(downloaded_file, 'r:gz') as tar:
        # Extract all files from the archive
        tar.extractall(output_dir)

    minio_client = boto3.client(
        's3',
        endpoint_url='http://minio-service.kubeflow:9000',
        aws_access_key_id='minio',
        aws_secret_access_key='minio123'
    )

    try:
        minio_client.create_bucket(Bucket=bucket_name)
    except Exception as e:
        # Bucket already created.
        pass

    for f in ["images", "labels"]:
        local_dir_path = os.path.join(output_dir, "DATA", f)
        files = os.listdir(local_dir_path)
        for file in files:
            local_path = os.path.join(local_dir_path, file)
            s3_path = os.path.join(bucket_name, f, file)
            minio_client.upload_file(local_path, bucket_name, s3_path)

    # Write the output file path to the output_file
    with open(output_file, 'w') as file:
        file.write(os.path.join(bucket_name))
