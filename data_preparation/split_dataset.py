from typing import NamedTuple

def split_dataset(random_state: int) -> NamedTuple('Outputs', [('x_val', str), ('y_val', str)]):
    """
    Split dataset into train, validation, and test sets.


    Args:
        random_state: Random seed for reproducibility

    Returns:
        NamedTuple containing:
            x_val (str): Path to validation images directory
            y_val (str): Path to validation labels directory
    """
    from typing import NamedTuple
    import os
    import glob
    import shutil

    images = list(glob.glob(os.path.join("/", "mnt", "pipeline", "DATASET", "DATA", "images", "**")))
    labels = list(glob.glob(os.path.join("/", "mnt", "pipeline", "DATASET", "DATA", "labels", "**")))

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

    for splits in ["train", "test", "val"]:
        for x in ["images", "labels"]:
            os.makedirs(os.path.join("/", "mnt", "pipeline", "DATASET", "DATA", splits, x))

    def move_files(objects, split, category):
        for source_object in objects:
            src = source_object.strip()
            dest = os.path.join("/", "mnt", "pipeline", "DATASET", "DATA", split, category,
                                os.path.basename(source_object))
            shutil.move(src, dest)

    move_files(x_train, "train", "images")
    move_files(x_test, "test", "images")
    move_files(x_val, "val", "images")
    move_files(y_train, "train", "labels")
    move_files(y_test, "test", "labels")
    move_files(y_val, "val", "labels")

    # Define output paths
    x_val_path = os.path.join("/", "mnt", "pipeline", "DATASET", "DATA", "val", "images")
    y_val_path = os.path.join("/", "mnt", "pipeline", "DATASET", "DATA", "val", "labels")

    output = NamedTuple('Outputs', [('x_val', str), ('y_val', str)])
    return output(x_val_path, y_val_path)