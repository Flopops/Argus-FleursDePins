""" Utils for files manipulation. """

import json
import os

import exifread
from PyQt5 import QtCore
import torch
from torchvision import io


class PathVerifier:
    """
    An object verifier.
    Verify if the argument path exist.

    Args:
        dir_paths (list of str): List of paths to check
    """

    def __init__(self, *dir_paths) -> None:
        self._dir_paths: list[str] = list(dir_paths)

    def verify_paths_exist(self) -> None:
        for path in self._dir_paths:
            if not os.path.exists(path):
                raise FileExistsError(f"Path {path} doesn't exist")


def load_json(file_path: str) -> dict[str, str | int | float]:
    """
    Load settings such as paths from a json file

    Args:
        file_path (str): The path to the json file.

    Returns:
        A dictionary containing the loaded json data.
    """
    with open(file_path, encoding="UTF-8") as json_file:
        return json.load(json_file)


def read_image_from_name(root: str, img_name: str) -> torch.Tensor:
    """
    Reads an image from the given root path and returns it as a tensor
    if the image exists.

    Args:
        root (str): The path to the root directory of the image.
        img_name (str): The name of the image file without the extension.

    Returns:
        A torch tensor representing the image.
    """
    img_path = os.path.join(root, img_name)
    pv = PathVerifier(img_path)
    pv.verify_paths_exist()
    return io.read_image(img_path)


def images_paths_from_dir(qdir: QtCore.QDir) -> list[str]:
    """
    Load all accepted images from one directory.

    :param qdir: QDir of the directory with all images
    :return: A sorted list of all images
    """
    if qdir.exists() is False:
        return []
    return [
        qdir.path() + "/" + f
        for f in (
            qdir.entryList(filters=QtCore.QDir.Files, sort=QtCore.QDir.Name)
        )
        if f.endswith((".jpg", ".JPG", ".png", ".PNG"))
    ]


def get_dimensions(image_path: str) -> tuple[int, int]:
    """Check in Exif data, the dimension of an image."""
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f, details=False)
    if "Image ImageLength" and "Image ImageWidth" in tags.keys():
        return int(str(tags["Image ImageLength"])), int(
            str(tags["Image ImageWidth"])
        )
    if "EXIF ExifImageLength" and "EXIF ExifImageWidth" in tags.keys():
        return int(str(tags["EXIF ExifImageLength"])), int(
            str(tags["EXIF ExifImageWidth"])
        )
    else:
        raise KeyError
