"""
Merge all csv file given an input directory path and an ouput 
directorypath where the merged csv files will be written.

To add images to the dataset.
DATASET is the main folder containing images and one labels.csv file.
Plot_05_2023 is containing all images.
Plot_csv is containing all csv.

1. Run python src/utils/merger.py -i /mnt/sda1/1-Pins/BRUT_2023/Plot_csv -o /mnt/sda1/1-Pins/DATASET -w
2.

usage: merger.py [-h] -i INPUT -o OUTPUT

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input dir path
  -o OUTPUT, --output OUTPUT
                        output dir path
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd

from utils.files_manipulator import PathVerifier


class Merger:
    """
    An object utility.
    Only merge csv for now.

    Args:
        input_path (str): path of the files to merge
    """

    def __init__(
        self,
        input_path: str | Path,
    ) -> None:
        self._input_path = str(input_path)

    @property
    def _csv_files(self) -> list[str]:
        """List of all csv files in input_path."""
        return [
            os.path.join(self._input_path, c)
            for c in os.listdir(self._input_path)
            if c.endswith(".csv")
        ]

    def merge_csv_files(
        self, output_path: str | Path = None, label_prefix: str = ""
    ) -> pd.DataFrame:
        """Merge all csv files in output_path/output_file_name."""
        if not self._csv_files:
            return pd.DataFrame(columns=["Label", "X1", "Y1", "X2", "Y2"])
        df_concat_labels = pd.concat(
            map(lambda x: pd.read_csv(x), self._csv_files)
        )
        df_concat_labels.Label = df_concat_labels.Label.apply(
            lambda x: label_prefix + x.replace(".tif", ".JPG")
        )
        df_concat_labels = df_concat_labels.sort_values(by=["Label"])
        df_concat_labels = df_concat_labels.rename(
            {"BX": "X1", "BY": "Y1"}, axis="columns"
        )
        df_concat_labels["X2"] = (
            df_concat_labels["X1"] + df_concat_labels.Width
        )
        df_concat_labels["Y2"] = (
            df_concat_labels["Y1"] + df_concat_labels.Height
        )
        if output_path is not None:
            df_concat_labels.to_csv(
                path_or_buf=os.path.join(str(output_path), "labels.csv"),
                columns=["Label", "X1", "Y1", "X2", "Y2"],
                index=False,
            )
        return df_concat_labels


def parse_options() -> argparse.Namespace:
    """Parses command line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Merge all csv file given an input directory path and an ouput directory"
        "path where the merged csv files will be written"
    )
    parser.add_argument(
        "-i", "--input", help="input dir path", required=True, type=str
    )
    parser.add_argument("-o", "--output", help="output dir path", type=str)
    parser.add_argument(
        "-p",
        "--prefix",
        help="prefix before each Label in rows",
        type=str,
        default="",
    )
    return parser.parse_args()


def adding_labels_to_dataset(df_labels):
    dataset_csv_path = "/mnt/sda1/1-Pins/DATASET/dataset.csv"
    df_dataset = pd.read_csv(dataset_csv_path)
    df_new_dataset = pd.concat([df_dataset, df_labels])
    df_new_dataset.to_csv(
        path_or_buf=dataset_csv_path,
        columns=["Label", "X1", "Y1", "X2", "Y2"],
        index=False,
    )


def adding_photos_to_dataset(raw_images_path: str):
    # WARNING: Only works for 2023
    plot_images_path = "/mnt/sda1/1-Pins/RAW_PHOTOS/" + raw_images_path
    for image in os.listdir(plot_images_path):
        src_path = os.path.join(plot_images_path, image)
        dst_path = os.path.join(
            "/mnt/sda1/1-Pins/DATASET",
            args.prefix + "_" + image.replace(".jpg", ".JPG"),
        )
        shutil.move(src=src_path, dst=dst_path)


if __name__ == "__main__":
    args = parse_options()
    paths_to_verify = [args.input]
    if args.output is not None:
        paths_to_verify.append(args.output)
    fv = PathVerifier(*paths_to_verify)
    fv.verify_paths_exist()
    merger = Merger(args.input)
    # Will write the csv if output not None
    df_labels = merger.merge_csv_files(args.output, args.prefix.lower() + "_")
    # adding_labels_to_dataset()
    # adding_photos_to_dataset("MATOUNEYRES_2023")
