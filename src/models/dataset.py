"""Module providing PinesDataset"""

from dataclasses import dataclass
import os
from typing import Callable
from pathlib import Path

import pandas as pd
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

from utils import LOGGER
from utils.data import (
    get_cropped_images_and_targets_from_df,
)
from utils.files_manipulator import read_image_from_name
from utils.merger import PathVerifier


class PinesDataset(datasets.VisionDataset):
    """
    Class for pines flowers dataset inheriting from VisionDataset base
    class compatible with torchvision that's why ``__len__`` and
    ``__getitem`` methods are overriden.

    Args:
        root (string): Root directory of dataset.
        scale (float, optional): A number between 0 and 1, proportion to load.
        transforms (callable, optional): A function/transforms that takes in
            an image and a label and returns the transformed versions of both.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    .. note::

        :attr:`transforms` and the combination of :attr:`transform` and :attr:`target_transform` are mutually exclusive.
    """

    def __init__(
        self,
        root: str,
        scale: float = 1.0,
        transforms: Callable | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        labelfile_path = os.path.join(root, "dataset.csv")
        PathVerifier(root, labelfile_path).verify_paths_exist()
        self.df_labels: pd.DataFrame = pd.read_csv(labelfile_path)
        self.img_names = pd.unique(self.df_labels.Label)
        self.images, self.targets = self._fill_images_targets(scale)
        LOGGER.info("%s images and targets loaded", self.__len__())

    def _fill_images_targets(
        self, scale: float
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """
        Preload images and targets associated to have a fixed number of
        images and targets at each batch.
        """
        images, targets = [], []
        img_names_scaled = self.img_names[: round(scale * len(self.img_names))]
        for img_name in img_names_scaled:
            big_image = read_image_from_name(self.root, img_name)
            current_df = self.df_labels.loc[
                self.df_labels.Label == img_name,
                ["X1", "Y1", "X2", "Y2"],
            ]
            (
                cropped_images,
                targets_of_cropped_images,
            ) = get_cropped_images_and_targets_from_df(
                img=big_image, size=320, df=current_df
            )
            images.append(torch.stack(cropped_images))
            targets += targets_of_cropped_images
        images_stack = torch.cat(images)
        assert len(targets) == images_stack.shape[0]
        return images_stack, targets

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            index (int): Index

        Returns:
            image (Tensor), target (boxes and labels): Sample and meta data,
            optionally transformed by the respective transforms.
        """
        image, target = self.images[index], self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        target = {
            "image_id": torch.tensor([index]),
            "area": ((target["boxes"])[:, 3] - (target["boxes"])[:, 1])
            * ((target["boxes"])[:, 2] - (target["boxes"])[:, 0]),
            "iscrowd": torch.zeros_like(target["labels"]),
            **target,
        }
        return image, target


@dataclass
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def get_train_val_test_dataloaders(
    root: str, scale: float, batch_size: int, seed: int = 42
) -> DataLoaders:
    pines_dataset = PinesDataset(root, scale=scale)
    train_set, val_set, test_set = random_split(
        dataset=pines_dataset,
        lengths=[0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(seed),
    )
    train_dataloader = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_set, batch_size=batch_size, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn
    )
    return DataLoaders(
        train=train_dataloader,
        validation=val_dataloader,
        test=test_dataloader,
    )


def collate_fn(batch):
    """Convert a batch in a zip tuple"""
    return tuple(zip(*batch))


def prepare_batch(
    batch, device: str | torch.device
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Convert a collate batch in correct format"""
    images_batch, targets_batch = batch
    images = [image.to(device) for image in images_batch]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]
    return images, targets


def save_train_val_test_sets(
    root: str,
    scale: float,
    seed: int = 42,
    lengths: list[float] = [0.8, 0.1, 0.1],
    save_dir: str = "../datasets/",
) -> None:
    rootpath = Path(root)
    pines_dataset = PinesDataset(root, scale=scale)
    train_set, val_set, test_set = random_split(
        dataset=pines_dataset,
        lengths=lengths,
        generator=torch.Generator().manual_seed(seed),
    )
    torch.save(
        train_set, os.path.join(save_dir, f"{rootpath.name}_train_set.pt")
    )
    torch.save(
        val_set, os.path.join(save_dir, f"{rootpath.name}_validation_set.pt")
    )
    torch.save(
        test_set, os.path.join(save_dir, f"{rootpath.name}_test_set.pt")
    )
