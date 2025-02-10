"""Tests of the merger.py module"""

import os

import pandas as pd
import pytest
import argparse
from unittest.mock import patch

from utils.merger import Merger, parse_options


@pytest.fixture
def io_paths(tmp_path):
    input_path = tmp_path / "input"
    input_path.mkdir()
    output_path = tmp_path / "output"
    output_path.mkdir()
    yield tmp_path, output_path


@pytest.fixture
def io_paths_with_files(io_paths):
    input_path, output_path = io_paths
    df = pd.DataFrame(
        data=[
            ("path:plot_of_land_2020_0", 10, 10, 100, 100),
            ("path:plot_of_land_2020_0", 10, 10, 100, 100),
        ],
        columns=["Label", "Width", "Height", "BX", "BY"],
    )
    df.to_csv(os.path.join(input_path, "labels0.csv"))
    df = pd.DataFrame(
        data=[
            ("path:plot_of_land_2020_1", 10, 10, 100, 100),
            ("path:plot_of_land_2020_1", 10, 10, 100, 100),
        ],
        columns=["Label", "Width", "Height", "BX", "BY"],
    )
    df.to_csv(os.path.join(input_path, "labels1.csv"))
    df = pd.DataFrame(
        data=[
            ("path:543:plot_of_land_2020_2", 10, 10, 100, 100),
        ],
        columns=["Label", "Width", "Height", "BX", "BY"],
    )
    df.to_csv(os.path.join(input_path, "labels2.csv"))
    return input_path, output_path


class TestsUtilsMerger:
    def test_create_merger_object(self, io_paths):
        _, output_path = io_paths
        m = Merger(output_path)
        actual_df = m.merge_csv_files(output_path)
        assert actual_df.empty

    def test_merge_csv_files(self, io_paths_with_files):
        input_path, output_path = io_paths_with_files
        merger = Merger(input_path)
        merger.merge_csv_files()
        # Checking no files in output_path
        assert os.listdir(output_path) == []
        actual_df = merger.merge_csv_files(output_path)
        assert os.path.exists(os.path.join(output_path, "labels.csv"))
        assert actual_df.shape == (5, 8)
        # TODO test CY and CX are equal to BX BY + Width Height from a random row

    def test_parse_options(self):
        with patch(
            "argparse.ArgumentParser.parse_args",
            return_value=argparse.Namespace(
                input="path/to/input", output="path/to/output"
            ),
        ):
            options = parse_options()
            assert options.input == "path/to/input"
            assert options.output == "path/to/output"
