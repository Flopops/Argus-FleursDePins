""" Utils for data manipulation. """

from enum import Enum
from itertools import product

import pandas as pd
from sahi.prediction import ObjectPrediction
import torch
import torchvision.transforms.functional as TF


def get_range_pair(i: int, next_i: int, size: int) -> tuple[int, int]:
    """For example 0,319 and 2680,2999 at the edge of the side"""
    return (
        (next_i - size, next_i - 1) if i + size > next_i else (i, i + size - 1)
    )


class ColumnsLetters(Enum):
    X = "X"
    Y = "Y"


def get_df_in_range(
    df: pd.DataFrame, start: int, stop: int, col_letter: ColumnsLetters
) -> pd.DataFrame:
    return df[
        (start < df[f"{col_letter.value}2"])
        & (df[f"{col_letter.value}1"] < stop)
    ]


def update_df_label(
    df: pd.DataFrame, top: int, left: int, size: int
) -> pd.DataFrame:
    df = df.copy()
    df.loc[:, ["X1", "X2"]] -= left
    df.loc[:, ["Y1", "Y2"]] -= top
    df.loc[:, ["X1", "Y1"]] = df.loc[:, ["X1", "Y1"]].clip(lower=0)
    df.loc[:, ["X2", "Y2"]] = df.loc[:, ["X2", "Y2"]].clip(upper=size)
    return df


def get_cropped_image_and_target(
    img: torch.Tensor, df: pd.DataFrame, i_h: int, i_w: int, size: int
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    updated_df = update_df_label(df, i_h, i_w, size)
    img = TF.convert_image_dtype(TF.crop(img, i_h, i_w, size, size))
    boxes = torch.as_tensor(updated_df.to_numpy(), dtype=torch.float32)
    label = {
        "boxes": boxes,
        "labels": torch.ones(len(boxes), dtype=torch.int64),
    }
    return img, label


def get_cropped_images_and_targets_from_df(
    img: torch.Tensor, size: int, df: pd.DataFrame
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    _, img_height, img_width = img.shape
    cropped_images, targets = [], []
    for w in range(0, img_width, size):
        w, next_w = get_range_pair(w, img_width, size)
        df_in_w_range = get_df_in_range(df, w, next_w, ColumnsLetters.X)
        if df_in_w_range.empty:
            continue
        for i_h in range(0, img_height, size):
            i_h, i_next_h = get_range_pair(i_h, img_height, size)
            df_in_range = get_df_in_range(
                df_in_w_range, i_h, i_next_h, ColumnsLetters.Y
            )
            if df_in_range.empty:
                continue
            cropped_image, target = get_cropped_image_and_target(
                img, df_in_range, i_h, w, size
            )
            cropped_images.append(cropped_image)
            targets.append(target)
    return cropped_images, targets


# TODO: test this function
def get_nb_objects_in_circle(
    dim: tuple[int, int],
    object_prediction_list: list[ObjectPrediction],
    radius_percentage: float = 0.8,
) -> int:
    h, w = dim
    circle_center = (w / 2, h / 2)
    radius_sq = ((min(h, w) * radius_percentage) / 2) ** 2
    return sum(
        any(
            (x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2
            <= radius_sq
            for x, y in product(
                [int(b) for b in object_prediction.bbox.to_xyxy()][::2],
                [int(b) for b in object_prediction.bbox.to_xyxy()][1::2],
            )
        )
        for object_prediction in object_prediction_list
    )
