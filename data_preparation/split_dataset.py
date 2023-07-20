from kfp.components import OutputPath


def split_dataset(bucket_name: str,
                  random_state: int,
                  x_train_file: OutputPath(str),
                  y_train_file: OutputPath(str),
                  x_test_file: OutputPath(str),
                  y_test_file: OutputPath(str),
                  x_val_file: OutputPath(str),
                  y_val_file: OutputPath(str)):

    from minio import Minio
    from minio.error import S3Error

    def list_objects_with_prefix(minio_client, bucket_name, prefix):
        try:
            objects = minio_client.list_objects(bucket_name, prefix=prefix, recursive=True)
            return [obj.object_name for obj in objects]
        except S3Error as err:
            print(f"Error listing objects with prefix '{prefix}' in bucket '{bucket_name}': {err}")

    minio_client = Minio(
        'minio-service.kubeflow:9000',
        access_key='minio',
        secret_key='minio123',
        secure=False
    )

    images = list_objects_with_prefix(minio_client, bucket_name, prefix=f"{bucket_name}/images")
    labels = list_objects_with_prefix(minio_client, bucket_name, prefix=f"{bucket_name}/labels")

    from sklearn.model_selection import train_test_split

    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(images, labels,
                                                        test_size=1 - train_ratio,
                                                        random_state=random_state)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                    test_size=test_ratio / (test_ratio + validation_ratio),
                                                    random_state=random_state)

    with open(x_train_file, "w") as f:
        f.writelines(line + '\n' for line in x_train)

    with open(y_train_file, "w") as f:
        f.writelines(line + '\n' for line in y_train)

    with open(x_test_file, "w") as f:
        f.writelines(line + '\n' for line in x_test)

    with open(y_test_file, "w") as f:
        f.writelines(line + '\n' for line in y_test)

    with open(x_val_file, "w") as f:
        f.writelines(line + '\n' for line in x_val)

    with open(y_val_file, "w") as f:
        f.writelines(line + '\n' for line in y_val)

    print(len(x_train))
    print(len(x_val))
    print(len(x_test))
