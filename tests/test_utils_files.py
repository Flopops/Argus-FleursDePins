""" Tests for utils/files.py. """

import os
import json
import pytest
import tempfile

from PIL import Image
import piexif
import torch
from torchvision import io
from PyQt5 import QtCore

from utils.files_manipulator import (
    PathVerifier,
    load_json,
    read_image_from_name,
    images_paths_from_dir,
    get_dimensions,
)


@pytest.fixture
def empty_tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def create_test_files(directory, filenames):
    for name in filenames:
        with open(os.path.join(directory, name), "w") as f:
            f.write("test")


def create_test_image(
    image_path: str, dimensions: tuple[int, int], with_exif=True
):
    image = Image.new("RGB", dimensions, color="white")
    image.save(image_path)
    if not with_exif:
        return
    piexif.insert(
        piexif.dump(
            {
                "0th": {
                    piexif.ImageIFD.ImageWidth: dimensions[0],
                    piexif.ImageIFD.ImageLength: dimensions[1],
                },
                "Exif": {},
                "1st": {},
                "thumbnail": None,
                "GPS": {},
            }
        ),
        image_path,
    )


@pytest.fixture
def image_with_exif():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmpfile:
        create_test_image(tmpfile.name, (1024, 768))
        yield tmpfile.name


# TODO: regroupe all the fixture to gether
@pytest.fixture
def image_without_exif():
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmpfile:
        create_test_image(tmpfile.name, (1024, 768), with_exif=False)
        yield tmpfile.name


class TestsUtilsFiles:
    def test_pathverifier_input_directory_doesnt_exists(self):
        with pytest.raises(FileExistsError):
            print("In pytest raises check")
            PathVerifier("/diqoidzq/idqizi/ziioa", "/tmp").verify_paths_exist()
        with pytest.raises(FileExistsError):
            PathVerifier(
                "djzioq/dzlqdé", "diqzojid", "dizq/djziq/iodzq/"
            ).verify_paths_exist()

    def test_load_json(self, tmpdir_factory):
        data = {"foo": "bar", "baz": 42}
        file_path = tmpdir_factory.mktemp("test").join("test.json")
        with open(file_path, "w", encoding="utf-8") as tmpfile:
            json.dump(data, tmpfile)
        expected_result = {"foo": "bar", "baz": 42}
        assert load_json(str(file_path)) == expected_result

    def test_load_json_raises_exception_for_invalid_file_path(self):
        with pytest.raises(FileNotFoundError):
            load_json("non_existent_file.json")

    def test_read_image_from_name(self, tmpdir):
        tmpdir.mkdir("images")
        image_file = str(tmpdir.join("images/image.jpg"))
        img = torch.zeros(3, 2, 2, dtype=torch.uint8)
        io.write_jpeg(img, image_file)
        root = os.path.dirname(image_file)
        img_name = os.path.basename(image_file)
        tensor = read_image_from_name(root, img_name)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 2, 2)

        # assert torch.all(result == image_file)

    def test_read_image_raises_exception_for_invalid_image_name(self):
        with pytest.raises(Exception):
            read_image_from_name("toto", "non_existent_image.JPG")

    def test_all_from_one_dir_nonexistent_directory(self):
        nonexistent_dir = QtCore.QDir("nonexistent/path")
        assert images_paths_from_dir(nonexistent_dir) == []

    def test_all_from_one_dir_empty_directory(self, empty_tmpdir):
        empty_dir = QtCore.QDir(empty_tmpdir)
        assert images_paths_from_dir(empty_dir) == []

    def test_all_from_one_dir_with_files(self, empty_tmpdir):
        test_files = ["image1.jpg", "image2.png", "document.txt"]
        create_test_files(empty_tmpdir, test_files)
        expected_result = [
            empty_tmpdir + "/" + f for f in test_files[:2]
        ]  # Only .jpg and .png are accepted
        test_dir = QtCore.QDir(empty_tmpdir)
        assert sorted(images_paths_from_dir(test_dir)) == sorted(
            expected_result
        )

    def test_get_dimensions_with_valid_image(self, image_with_exif):
        length, width = get_dimensions(image_with_exif)
        assert length == 768 and width == 1024

    def test_get_dimensions_with_no_exif_data(self, image_without_exif):
        with pytest.raises(KeyError):
            get_dimensions(image_without_exif)
