def download_dataset(output_dir: str = "DATASET"):
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
            # Update the progress bar with the size of the downloaded chunk
            progress_bar.update(len(chunk))
            file.write(chunk)

    # Open the tar archive
    with tarfile.open(downloaded_file, 'r:gz') as tar:
        # Extract all files from the archive
        tar.extractall(os.path.join("/", "mnt", "pipeline", output_dir))
