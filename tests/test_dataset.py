import dataclasses
import os

import pytest
import torch
import pandas as pd
from torchvision import transforms, io
from torch.utils.data import DataLoader

from models.dataset import (
    PinesDataset,
    collate_fn,
    get_train_val_test_dataloaders,
    prepare_batch,
)


@pytest.fixture(scope="session")
def pines_dataset_root(tmpdir_factory):
    """Create images and label.csv files at `tmpdir_factory/data`"""
    temp_dir = tmpdir_factory.mktemp("data")
    data = {
        "Label": [
            "photo_1",
            "photo_1",
            "photo_1",
            "photo_2",
            "photo_2",
            "photo_2",
            "photo_2",
            "photo_3",
            "photo_3",
            "photo_3",
        ],
        "X1": [2625, 3785, 2112, 3469, 2625, 3785, 2112, 2992, 2625, 3500],
        "Y1": [4449, 4026, 4222, 4228, 4449, 4026, 4222, 4961, 4449, 4896],
        "X2": [2634, 3801, 2136, 3492, 2634, 3801, 2136, 3010, 2634, 3519],
        "Y2": [4458, 4043, 4244, 4248, 4458, 4043, 4244, 4991, 4458, 4920],
    }
    photo_1_path = temp_dir.join("photo_1.jpg")
    photo_2_path = temp_dir.join("photo_2.jpg")
    photo_3_path = temp_dir.join("photo_3.jpg")

    img = torch.zeros(3, 5460, 8192, dtype=torch.uint8)

    io.write_jpeg(img, str(photo_1_path))
    io.write_jpeg(img, str(photo_2_path))
    io.write_jpeg(img, str(photo_3_path))
    pd.DataFrame(data).to_csv(
        os.path.join(temp_dir, "labels.csv"), index=False
    )
    return str(temp_dir)


@pytest.fixture
def pines_dataset(pines_dataset_root):
    """Instanciate a PinesDataset from a root"""
    return PinesDataset(root=pines_dataset_root, scale=1.0)


class TestsDataset:
    def test_pines_dataset_with_transforms_in_init(self, pines_dataset_root):
        pines_ds = PinesDataset(
            root=pines_dataset_root,
            scale=1.0,
            transform=transforms.Resize((240, 240)),
        )
        sample_image, _ = pines_ds[0]
        assert sample_image.shape == (3, 240, 240)

    def test_pines_dataset_len(self, pines_dataset):
        assert len(pines_dataset) == 10

    def test_pines_dataset_getitem(self, pines_dataset):
        sample = pines_dataset[0]
        assert isinstance(sample[0], torch.Tensor)
        assert isinstance(sample[1], dict)
        assert "boxes" in sample[1]
        assert "labels" in sample[1]
        assert "image_id" in sample[1]

    def test_pines_dataset_transforms(self, pines_dataset):
        # Test transforms applied on a sample image from the dataset
        sample_image, _ = pines_dataset[0]
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
            ]
        )
        transformed_image = transform(sample_image)
        assert isinstance(transformed_image, torch.Tensor)
        assert transformed_image.shape == (3, 224, 224)

    def test_pines_dataset_dataloader(self, pines_dataset):
        # Test creation of a DataLoader from PinesDataset
        dataloader = DataLoader(
            pines_dataset, batch_size=2, collate_fn=collate_fn
        )
        batch = next(iter(dataloader))
        assert isinstance(batch, tuple)
        assert isinstance(batch[0][0], torch.Tensor)
        assert isinstance(batch[1][0], dict)
        assert batch[0][0].shape == (3, 320, 320)

    def test_get_train_val_test_dataloaders(self, pines_dataset_root):
        # Test the get_train_val_test_dataloaders function
        dataloaders = get_train_val_test_dataloaders(
            pines_dataset_root, scale=1.0, batch_size=1, seed=42
        )
        assert dataclasses.is_dataclass(dataloaders)
        assert [k for k in dataclasses.asdict(dataloaders).keys()] == [
            "train",
            "validation",
            "test",
        ]
        assert isinstance(dataloaders.train, DataLoader)
        assert len(dataloaders.train) == 8
        assert isinstance(dataloaders.validation, DataLoader)
        assert len(dataloaders.validation) == 1
        assert isinstance(dataloaders.test, DataLoader)
        assert len(dataloaders.test) == 1

    def test_collate_fn(self):
        batch = [
            (
                torch.randn(3, 320, 320),
                {"boxes": torch.randn(2, 4), "labels": torch.tensor([1, 2])},
            ),
            (
                torch.randn(3, 320, 320),
                {
                    "boxes": torch.randn(3, 4),
                    "labels": torch.tensor([1, 3, 2]),
                },
            ),
            (
                torch.randn(3, 320, 320),
                {"boxes": torch.randn(1, 4), "labels": torch.tensor([2])},
            ),
        ]
        output = collate_fn(batch)

        assert isinstance(output, tuple)
        assert len(output) == 2
        assert isinstance(output[0], tuple)
        assert isinstance(output[1], tuple)
        assert len(output[0]) == len(output[1]) == 3
        assert isinstance(output[1][0], dict)

    def test_prepare_batch(self):
        batch = (
            [torch.randn(3, 224, 224), torch.randn(3, 224, 224)],
            [{"label": torch.tensor(1)}, {"label": torch.tensor(0)}],
        )
        device = torch.device("cpu")
        images, targets = prepare_batch(batch, device)
        assert isinstance(images, list)
        assert isinstance(targets, list)
        assert len(images) == len(targets) == 2
        assert all(isinstance(img, torch.Tensor) for img in images)
        for t in targets:
            assert isinstance(t, dict)
            for v in t.values():
                assert isinstance(v, torch.Tensor)
                assert v.device == device

    @pytest.mark.skip()
    def test_save_train_val_test_sets():
        # TODO:
        ...
