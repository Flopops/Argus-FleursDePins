""" Tests of utils/data.py. """

import pytest
import pandas as pd
import torch

from utils.data import (
    ColumnsLetters,
    get_range_pair,
    get_df_in_range,
    update_df_label,
    get_cropped_image_and_target,
    get_cropped_images_and_targets_from_df,
    get_nb_objects_in_circle,
)


class MockBBox:
    def __init__(self, minx, miny, maxx, maxy):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy

    def to_xyxy(self):
        return [self.minx, self.miny, self.maxx, self.maxy]


class MockObjectPrediction:
    def __init__(self, bbox):
        self.bbox = bbox


class TestsUtilsData:
    @pytest.mark.parametrize(
        "i, next_i, expected",
        [
            (0, 1000, (0, 319)),
            (100, 1000, (100, 419)),
            (500, 1000, (500, 819)),
            (960, 1000, (680, 999)),
            (5440, 5460, (5140, 5459)),
        ],
    )
    def test_get_range_pair(self, i, next_i, expected):
        assert get_range_pair(i, next_i, 320) == expected

    def test_get_df_in_range(self):
        expected = pd.DataFrame(
            {
                "X1": [20, 30, 40],
                "Y1": [25, 35, 45],
                "X2": [21, 31, 41],
                "Y2": [26, 36, 46],
            }
        )
        result = get_df_in_range(
            df=pd.DataFrame(
                {
                    "X1": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                    "Y1": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95],
                    "X2": [1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
                    "Y2": [6, 16, 26, 36, 46, 56, 66, 76, 86, 96],
                }
            ),
            start=19,
            stop=50,
            col_letter=ColumnsLetters.X,
        )
        assert result.shape == expected.shape
        assert result.columns.tolist() == expected.columns.tolist()
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )

    def test_update_df_label(self):
        expected = pd.DataFrame(
            {
                "X1": [3, 13, 23],
                "Y1": [15, 25, 35],
                "X2": [23, 33, 43],
                "Y2": [35, 45, 50],
            }
        )
        result = update_df_label(
            df=pd.DataFrame(
                {
                    "X1": [10, 20, 30],
                    "Y1": [20, 30, 40],
                    "X2": [30, 40, 50],
                    "Y2": [40, 50, 60],
                }
            ),
            top=5,
            left=7,
            size=50,
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_get_cropped_image_and_target(self):
        img = torch.rand(3, 500, 500)
        df = pd.DataFrame(
            {
                "X1": [100, 200, 300],
                "Y1": [100, 200, 300],
                "X2": [200, 300, 400],
                "Y2": [200, 300, 400],
            }
        )
        i_h = 50
        i_w = 50
        size = 300

        cropped_img, label = get_cropped_image_and_target(
            img, df, i_h, i_w, size
        )

        # Test cropped image
        assert cropped_img.shape == (3, size, size)
        assert isinstance(cropped_img, torch.Tensor)
        assert cropped_img.dtype == torch.float32
        assert (cropped_img.min() >= 0) and (cropped_img.max() <= 1)

        # Test label
        assert isinstance(label, dict)
        assert set(label.keys()) == {"boxes", "labels"}
        assert isinstance(label["boxes"], torch.Tensor)
        assert label["boxes"].dtype == torch.float32
        assert label["boxes"].shape == (3, 4)
        assert torch.all(label["boxes"][:, :2] >= 0)
        assert torch.all(label["boxes"][:, 2:] <= size)
        assert torch.equal(
            torch.tensor(
                [
                    [50.0, 50.0, 150.0, 150.0],
                    [150.0, 150.0, 250.0, 250.0],
                    [250.0, 250.0, 300.0, 300.0],
                ]
            ),
            label["boxes"],
        )
        assert torch.all(
            label["labels"] == torch.ones((3,), dtype=torch.int64)
        )

    def test_get_cropped_images_and_targets_from_df(self):
        img = torch.randn((3, 100, 100))
        df = pd.DataFrame(
            {
                "X1": [10, 20, 30, 40, 50],
                "Y1": [10, 20, 30, 40, 50],
                "X2": [15, 25, 35, 45, 55],
                "Y2": [15, 25, 35, 45, 55],
            }
        )
        size = 29

        cropped_images, targets = get_cropped_images_and_targets_from_df(
            img, size, df
        )

        assert len(cropped_images) == len(targets) == 2
        for cropped_image in cropped_images:
            assert cropped_image.shape == (3, size, size)
        assert torch.equal(
            targets[0]["boxes"],
            torch.tensor([[10.0, 10.0, 15.0, 15.0], [20.0, 20.0, 25.0, 25.0]]),
        )
        assert torch.equal(
            targets[1]["boxes"],
            torch.tensor(
                [
                    [1.0, 1.0, 6.0, 6.0],
                    [11.0, 11.0, 16.0, 16.0],
                    [21.0, 21.0, 26.0, 26.0],
                ]
            ),
        )

    def test_no_objects_in_circle(self):
        assert get_nb_objects_in_circle((100, 100), []) == 0

    def test_some_objects_in_circle(self):
        object_predictions = [
            MockObjectPrediction(MockBBox(50, 50, 51, 51)),
            MockObjectPrediction(MockBBox(98, 98, 99, 99)),
            # TODO: BBox on edge
        ]
        dim = (100, 100)  # Mock dimensions
        assert (
            get_nb_objects_in_circle(dim, object_predictions) == 1
        )  # Adjust based on your test setup

    def test_all_objects_in_circle(self):
        object_predictions = [
            MockObjectPrediction(MockBBox(40, 40, 45, 45)),
            MockObjectPrediction(MockBBox(50, 50, 51, 51)),
        ]
        dim = (100, 100)
        assert get_nb_objects_in_circle(dim, object_predictions) == len(
            object_predictions
        )
