from torchvision.datasets import VisionDataset
import typing as t
from pathlib import Path
import torch
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import DataLoader
from typing_extensions import Literal
from deepchecks.vision.vision_data import BatchOutputFormat, VisionData
import warnings
from ultralytics import YOLO

LABEL_MAP = {0: "id_card"}
from deepchecks.vision.utils.test_utils import get_data_loader_sequential, hash_image
from deepchecks.vision.vision_data import BatchOutputFormat, VisionData


def deepchecks_collate(model) -> t.Callable:
    """Process batch to deepchecks format.

    Parameters
    ----------
    model
        model to predict with
    Returns
    -------
    BatchOutputFormat
        batch of data in deepchecks format
    """

    def _process_batch_to_deepchecks_format(data) -> BatchOutputFormat:
        raw_images = [x[0] for x in data]
        images = [np.array(x) for x in raw_images]

        def move_class(tensor):
            return (
                torch.index_select(
                    tensor, 1, torch.LongTensor([4, 0, 1, 2, 3]).to(tensor.device)
                )
                if len(tensor) > 0
                else tensor
            )

        labels = [move_class(x[1]) for x in data]

        predictions = []
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)
            raw_predictions: "yolov8.models.common.Detections" = model(
                raw_images
            )  # noqa: F821

            # yolo Detections objects have List[torch.Tensor] xyxy output in .pred
            for result in raw_predictions:
                pred_modified = torch.clone(result.boxes.data)
                pred_modified[:, 2] = (
                    pred_modified[:, 2] - pred_modified[:, 0]
                )  # w = x_right - x_left
                pred_modified[:, 3] = (
                    pred_modified[:, 3] - pred_modified[:, 1]
                )  # h = y_bottom - y_top
                predictions.append(pred_modified)
        return BatchOutputFormat(images=images, labels=labels, predictions=predictions)

    return _process_batch_to_deepchecks_format


def _batch_collate(batch):
    imgs, labels = zip(*batch)
    return list(imgs), list(labels)


def get_image_and_label(image_file, label_file, transforms=None):
    """Get image and label in correct format for models from file paths."""
    opencv_image = cv2.imread(str(image_file))
    pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    if label_file is not None and label_file.exists():
        img_labels = [
            l.split() for l in label_file.open("r").read().strip().splitlines()
        ]
        img_labels = np.array(img_labels, dtype=np.float32)
    else:
        img_labels = np.zeros((0, 5), dtype=np.float32)
    # Transform x,y,w,h in yolo format (x, y are of the image center, and coordinates are normalized) to standard
    # x,y,w,h format, where x,y are of the top left corner of the bounding box and coordinates are absolute.
    bboxes = []
    for label in img_labels:
        x, y, w, h = label[1:]
        # Note: probably the normalization loses some accuracy in the coordinates as it truncates the number,
        # leading in some cases to `y - h / 2` or `x - w / 2` to be negative
        bboxes.append(
            np.array(
                [
                    max((x - w / 2) * pil_image.width, 0),
                    max((y - h / 2) * pil_image.height, 0),
                    w * pil_image.width,
                    h * pil_image.height,
                    label[0],
                ]
            )
        )
    if transforms is not None:
        # Albumentations accepts images as numpy and bboxes in defined format + class at the end
        transformed = transforms(image=np.array(pil_image), bboxes=bboxes)
        pil_image = Image.fromarray(transformed["image"])
        bboxes = transformed["bboxes"]
    return pil_image, bboxes


class IdCardDataset(VisionDataset):
    """An instance of PyTorch VisionData the represents the id card dataset.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset.
    name : str
        Name of the dataset.
    train : bool
        if `True` train dataset, otherwise test dataset
    transform : Callable, optional
        A function/transforms that takes in an image and a label and returns the
        transformed versions of both.
        E.g, ``transforms.Rotate``
    target_transform : Callable, optional
        A function/transform that takes in the target and transforms it.
    transforms : Callable, optional
        A function/transform that takes in an PIL image and returns a transformed version.
        E.g, transforms.RandomCrop
    """

    def __init__(
        self,
        root: str,
        name: str,
        train: bool = True,
        transform: t.Optional[t.Callable] = None,
        target_transform: t.Optional[t.Callable] = None,
        transforms: t.Optional[t.Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.train = train
        self.root = Path(root).absolute()
        self.train_images_dir = self.root / "unmodified_brightness" / "images"
        self.train_labels_dir = self.root / "unmodified_brightness" / "labels"

        self.test_images_dir = Path(root) / "modified_brightness" / "images"
        self.test_labels_dir = Path(root) / "modified_brightness" / "labels"

        if self.train is True:
            train_images: t.List[Path] = sorted(self.train_images_dir.glob("./*.tif"))
            train_labels: t.List[t.Optional[Path]] = []
            print(self.train_images_dir)
            print(self.train_images_dir.exists())
            for image in train_images:
                label = self.train_labels_dir / f"{image.stem}.txt"
                train_labels.append(label if label.exists() else None)

            assert (
                len(train_images) != 0
            ), "Did not find folder with images or it was empty"
            assert not all(
                l is None for l in train_labels
            ), "Did not find folder with labels or it was empty"
            self.images = train_images
            self.labels = train_labels
        else:
            test_images: t.List[Path] = sorted(self.test_images_dir.glob("./*.tif"))
            test_labels: t.List[t.Optional[Path]] = []

            for image in test_images:
                label = self.test_labels_dir / f"{image.stem}.txt"
                test_labels.append(label if label.exists() else None)

            assert (
                len(test_images) != 0
            ), "Did not find folder with images or it was empty"
            assert not all(
                l is None for l in test_labels
            ), "Did not find folder with labels or it was empty"
            self.images = test_images
            self.labels = test_labels

    def __getitem__(self, idx: int) -> t.Tuple[Image.Image, torch.Tensor]:
        """Get the image and label at the given index."""
        # open image using cv2, since opening with Pillow give slightly different results based on Pillow version
        img, bboxes = get_image_and_label(
            self.images[idx], self.labels[idx], self.transforms
        )

        # Return tensor of bboxes
        if bboxes:
            bboxes = torch.stack([torch.tensor(x) for x in bboxes])
        else:
            bboxes = torch.tensor([])
        return img, bboxes

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.images)


def load_dataset(
    train: bool = True,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = True,
    object_type: Literal["VisionData", "DataLoader"] = "DataLoader",
    n_samples: t.Optional[int] = None,
    device: t.Union[str, torch.device] = "cpu",
) -> t.Union[DataLoader, VisionData]:
    """Get the id card dataset and return a dataloader.

    Parameters
    ----------
    train : bool, default: True
        if `True` train dataset, otherwise test dataset
    batch_size : int, default: 32
        Batch size for the dataloader.
    num_workers : int, default: 0
        Number of workers for the dataloader.
    shuffle : bool, default: False
        Whether to shuffle the dataset.
    pin_memory : bool, default: True
        If ``True``, the data loader will copy Tensors
        into CUDA pinned memory before returning them.
    object_type : Literal['Dataset', 'DataLoader'], default: 'DataLoader'
        type of the return value. If 'Dataset', :obj:`deepchecks.vision.VisionData`
        will be returned, otherwise :obj:`torch.utils.data.DataLoader`
    n_samples : int, optional
        Only relevant for loading a VisionData. Number of samples to load. Return the first n_samples if shuffle
        is False otherwise selects n_samples at random. If None, returns all samples.
    device : t.Union[str, torch.device], default : 'cpu'
        device to use in tensor calculations

    Returns
    -------
    Union[DataLoader, VisionData]
        A DataLoader or VisionData instance representing our id card dataset
    """
    dataset = IdCardDataset(
        root=str("DATA"),
        name="ID Card Dataset",
        train=train,
    )
    print("dataset_loaded")

    if object_type == "DataLoader":
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_batch_collate,
            pin_memory=pin_memory,
            generator=torch.Generator(),
        )
    elif object_type == "VisionData":
        model = YOLO("../serving/yolov8_custom.pt")
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=deepchecks_collate(model),
            pin_memory=pin_memory,
            generator=torch.Generator(),
        )
        dataloader = get_data_loader_sequential(
            dataloader, shuffle=shuffle, n_samples=n_samples
        )
        return VisionData(
            batch_loader=dataloader,
            label_map=LABEL_MAP,
            task_type="object_detection",
            reshuffle_data=False,
        )
    else:
        raise TypeError(f"Unknown value of object_type - {object_type}")
