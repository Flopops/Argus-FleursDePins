#!/usr/bin/env python

"""Module which convert Lambert 93 coordinates into WGS84 into a kml

usage: csv2kml4path.py [-h] [--delta DELTA] [--threshold-z THRESHOLD_Z]

Order Lambert-93 coordinates from csv into WGS84.

options:
  -h, --help            show this help message and exit
  --delta DELTA, -d DELTA
                        Delta distance in meter above the tree peaks.
  --threshold-z THRESHOLD_Z, -t THRESHOLD_Z
                        Elevation minimum for the trees.
"""
import abc
import argparse
import enum
import math
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pyproj
import simplekml
from sklearn.decomposition import PCA

Coord = tuple[float, float, float]
Coords = list[Coord]


def parse_options() -> argparse.Namespace:
    """Parses command line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Order Lambert-93 coordinates from csv into WGS84."
    )
    parser.add_argument(
        "--delta",
        "-d",
        type=int,
        default=10,
        help="Delta distance in meter above the tree peaks.",
    )
    parser.add_argument(
        "--threshold-z",
        "-t",
        type=int,
        default=10,
        help="Elevation minimum for the trees.",
    )
    return parser.parse_args()


def get_angle_from_pca(coords: Coords):
    """Calculate the dominant orientation using PCA."""
    points = np.array(coords)
    pca = PCA(n_components=2)
    pca.fit(points)
    angle_x = -math.degrees(
        math.atan2(pca.components_[0][1], pca.components_[0][0])
    )
    angle_y = -math.degrees(
        math.atan2(pca.components_[1][1], pca.components_[1][0])
    )
    return angle_x, angle_y


def get_path_for_plot(df: pd.DataFrame, outliers: list[int], threshold_z: int):
    """Compute an optimized path for plot with a given list of coordinates.

    Returns:
        Coords: Reorder list of coordinates.
    """
    coords = list(zip(df.x, df.y, df.z))

    clean_coordinates(
        coords=coords,
        outliers=outliers,
        zip_zh=zip(df.z, df.h),
        threshold_z=threshold_z,
    )
    angle_from_pca = get_angle_from_pca(coords)


class ParcelleType(enum.Enum):
    COL0 = "premiere_colonne"
    C1 = "secteur_C1"
    C2 = "secteur_C2"
    ALL = "parcelle_entiere"


def get_path_for_castillon(
    df: pd.DataFrame,
    threshold_z: int,
    parcelle_type: ParcelleType = ParcelleType.ALL,
) -> Coords:
    """Compute an optimized path for castillonville.

    Returns:
        Coords: Reorder list of coordinates.
    """

    coords = list(zip(df.x, df.y, df.z))

    clean_coordinates(
        coords=coords,
        outliers=[915, 747],
        zip_zh=zip(df.z, df.h),
        threshold_z=threshold_z,
    )

    # Rotated coordinates for extracting col0
    coords_r = get_rotated_coords(coords, angle=1)
    coords_rs = sorted(coords_r, key=lambda c: [-c[0], c[1]])

    reorder_coords_0 = retrieve_reorder_coords(
        coords=coords,
        coords_rotated=coords_r,
        coords_rotated_reordered=sorted(coords_rs[:22], key=lambda c: c[1]),
    )

    sort_key = lambda c: [-c[1], c[0]]
    match (parcelle_type):
        case ParcelleType.COL0:
            return reorder_coords_0
        case ParcelleType.C1:
            pair = [22, 536]
        case ParcelleType.C2:
            pair = [536, None]
            sort_key = lambda c: [c[1], c[0]]
            reorder_coords_0 = []
            coords_rs.pop(759)
        case ParcelleType.ALL:
            pair = [22, None]
        case _:
            raise ValueError()

    coords_rtr = get_rotated_coords(coords_rs[pair[0] : pair[1]], angle=-10)

    reorder_coords_rows = retrieve_reorder_coords(
        coords=coords,
        coords_rotated=coords_r,
        coords_rotated_reordered=retrieve_reorder_coords(
            coords=coords_rs[pair[0] : pair[1]],
            coords_rotated=coords_rtr,
            coords_rotated_reordered=reorder_coords(
                coords_rtr, sort_key=sort_key
            ),
        ),
    )
    return reorder_coords_0 + reorder_coords_rows


def get_path_for_matouneyres2(
    df: pd.DataFrame,
    threshold_z: int = 10,
) -> Coords:
    coords = list(zip(df.x, df.y, df.z))
    clean_coordinates(
        coords=coords,
        outliers=[838, 782, 754, 598, 464],
        zip_zh=zip(df.z, df.h),
        threshold_z=threshold_z,
    )
    coords_r = get_rotated_coords(coords, angle=-40)
    return retrieve_reorder_coords(
        coords=coords,
        coords_rotated=coords_r,
        coords_rotated_reordered=reorder_coords(
            coords=coords_r,
            sort_key=lambda c: [-c[0], c[1]],
            flip_key=lambda c: c[1],
            dist_min=1.62,
            axis=0,
        ),
    )


def clean_coordinates(
    coords: Coords,
    outliers: list[int],
    zip_zh: Iterable,
    threshold_z: int,
) -> None:
    """
    Cleaning the list of coordinates coords by removing outliers
    and trees under a height threshold.
    """
    for o in outliers:
        coords.pop(o)
    threshold_z_wgs84: float = get_threshold_z_in_wgs84(zip_zh, threshold_z)
    for i in range(len(coords) - 1, -1, -1):
        if coords[i][2] < threshold_z_wgs84:
            coords.pop(i)


def get_threshold_z_in_wgs84(zip_zh: Iterable, threshold_z: int) -> float:
    """Compute the 0 local in wgs84 and add it the threshold_z."""
    return (
        np.mean(np.array([z - h for z, h in zip_zh])).astype(float)
        + threshold_z
    )


def get_rotated_coords(coords: Coords, angle: int) -> Coords:
    """Rotate each point in coords at a specific angle."""
    return [rotate_point(coord, angle=angle) for coord in coords]


def rotate_point(point: Coord, angle: int) -> Coord:
    """Rotate a point at a given angle."""
    x, y, z = point
    angle_rad = math.radians(angle)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    x_rotated = x * cos_angle - y * sin_angle
    y_rotated = x * sin_angle + y * cos_angle
    return (x_rotated, y_rotated, z)


def retrieve_reorder_coords(
    coords: Coords,
    coords_rotated: Coords,
    coords_rotated_reordered: Coords,
) -> Coords:
    """Reorder coords in the order of the rotated reordered given coordinates."""
    return [
        coords[i]
        for i in [coords_rotated.index(v) for v in coords_rotated_reordered]
    ]


def reorder_coords(
    coords: Coords,
    sort_key: Callable = lambda c: [-c[1], c[0]],
    flip_key: Callable = lambda c: c[0],
    dist_min: float = 1.0,
    flip: bool = True,
    axis: int = 1,
) -> Coords:
    return flip_columns_1_out_of_2(
        coords=sorted(coords, key=sort_key),
        dist_min=dist_min,
        key=flip_key,
        flip=flip,
        axis=axis,
    )


def flip_columns_1_out_of_2(
    coords: Coords,
    dist_min: float = 1.0,
    key: Callable = lambda c: c[0],
    flip: bool = True,
    axis: int = 1,
) -> Coords:
    """Specific sort for a boustrophedon path on coords."""
    sort_start_idx = 0
    # outliers_idx = []
    for i in range(len(coords) - 1):
        if dist(coords[i + 1], coords[i], axis=axis) > dist_min:
            # TODO: test and put in prod
            # if i - sort_start_idx < 5:
            #     outliers_idx.append(*list(range(sort_start_idx, i+1)))
            #     sort_start_idx = i + 1
            #     continue
            coords[sort_start_idx : i + 1] = sorted(
                coords[sort_start_idx : i + 1], key=key, reverse=flip
            )
            sort_start_idx = i + 1
            flip = not flip
    coords[sort_start_idx:] = sorted(
        coords[sort_start_idx:], key=key, reverse=flip
    )
    # for i in sorted(outliers_idx, reverse=True):
    #     coords.pop(i)
    return coords


def dist(p1: Coord, p2: Coord, axis: int = 0) -> float:
    """Computes the distance between two points along an given axis."""
    return abs(p1[axis] - p2[axis])


class CoordinatesConverter(abc.ABC):
    """
    Abstract class for AtoBConverter
    where A and B are two different coordinates system.
    """

    @abc.abstractmethod
    def convert(self) -> Coords:
        """Converts A coordinates to B coordinates.

        Returns:
            Coord: A list of tuples containing the converted B coordinates.
        """


class Lambert93ToWgs84Convertor(CoordinatesConverter):
    def __init__(self, df: pd.DataFrame, delta: int = 10) -> None:
        """Initializes the Lambert93ToWgs84Convertor class.

        Args:
            csv_file (str): The name of the CSV file containing Lambert-93 coordinates.
            delta (int, optional): Delta distance in meters above the tree peaks. Defaults to 10.
            sealvl (int, optional): Sea level. Defaults to 0.
        """

        self.df = df
        self.delta = delta

    def convert(self) -> Coords:
        """Converts Lambert-93 coordinates to WGS84 coordinates.

        Returns:
            Coords: A list of tuples containing the converted WGS84 coordinates.
        """
        rgf93: pyproj.CRS = pyproj.CRS.from_epsg(2154)
        wgs84: pyproj.CRS = pyproj.CRS.from_epsg(4326)
        transformer = pyproj.Transformer.from_crs(rgf93, wgs84)
        coords = []
        for x, y, z in zip(self.df.x, self.df.y, self.df.z):
            lat, long, elev = transformer.transform(x, y, z)
            alt = elev + self.delta
            coords.append((long, lat, alt))
        return coords


class KmlCreator:
    def __init__(self, coords: Coords, kml_name: str) -> None:
        """Initializes the KmlCreator class.

        Args:
            coords (Coords): A list of tuples containing the converted WGS84 coordinates.
            kml_name (str): The name of the KML file to be created.
        """
        self.coords = coords
        self.kml_name = kml_name
        self.altitudemode = simplekml.AltitudeMode.relativeToGround

    def create_kml(self) -> None:
        """Create the Kml object and save it in a file named kml_name."""
        kml = simplekml.Kml()
        linestring = kml.newlinestring(name="Waypoints connected")
        linestring.coords = self.coords
        linestring.altitudemode = self.altitudemode
        linestring.extrude = 1
        kml.save(self.kml_name)


if __name__ == "__main__":
    args = parse_options()

    # TODO: if args.castillon: with --castillon given

    # Castillon
    reordered_castillon_coords = get_path_for_castillon(
        df=pd.read_csv("input/cimes_castillon1et2.csv"),
        threshold_z=args.threshold_z,
    )
    castillon_kml_name = f"output/castillon/castillon_delta{args.delta}.kml"

    # Matouneyres 2
    # reordered_matouneyres_coords = get_path_for_castillon(
    #     df=pd.read_csv("input/cimes_matouneyres2.csv"),
    #     threshold_z=args.threshold_z,
    # )
    # matouneyres_kml_name = f"output/matouneyres/matouneyres2_parcelle_entiere_delta{args.delta}_WGS84.kml"

    rgf93_convertor = Lambert93ToWgs84Convertor(
        df=pd.DataFrame(reordered_castillon_coords, columns=["x", "y", "z"]),
        delta=args.delta,
    )

    kml_creator = KmlCreator(rgf93_convertor.convert(), castillon_kml_name)
    kml_creator.create_kml()
