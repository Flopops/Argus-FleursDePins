import os
import argparse
from unittest.mock import patch
from xml.etree import ElementTree as ET

import pytest
import pandas as pd

from csv2kml4path import (
    parse_options,
    get_threshold_z_in_wgs84,
    clean_coordinates,
    rotate_point,
    get_rotated_coords,
    retrieve_reorder_coords,
    dist,
    flip_columns_1_out_of_2,
    reorder_coords,
    ParcelleType,
    get_path_for_castillon,
    CoordinatesConverter,
    Lambert93ToWgs84Convertor,
    KmlCreator,
)


def test_parse_options():
    with patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(delta=5, threshold_z=15),
    ):
        options = parse_options()
        assert options.delta == 5
        assert options.threshold_z == 15


def test_get_threshold_z_in_wgs84():
    df = pd.DataFrame(
        data={
            "z": [102, 104, 106],
            "h": [2, 4, 6],
        }
    )
    actuel_threshold_z_wgs84 = get_threshold_z_in_wgs84(
        zip(df.z, df.h), threshold_z=5
    )
    assert actuel_threshold_z_wgs84 == 105.0


def test_clean_coordinates():
    df = pd.DataFrame(
        data={
            "x": list(range(0, 1000)),
            "y": list(range(0, 1000)),
            "z": [150] * 950 + [105] * 50,
            "h": [50] * 950 + [5] * 50,
        }
    )
    raw_coords = list(zip(df.x, df.y, df.z))
    clean_coordinates(raw_coords, [915, 747], zip(df.z, df.h), threshold_z=10)
    assert len(raw_coords) == 948
    assert raw_coords[747] == (748, 748, 150)
    assert raw_coords[915] == (917, 917, 150)


def test_rotate_point():
    rotated_point = rotate_point(
        point=(399242.50670129404, 6412023.861878326, 128.67240350277112),
        angle=45,
    )
    expected_rotated_point = (
        -4251678.470037718,
        4816292.63769052,
        128.67240350277112,
    )
    assert rotated_point == expected_rotated_point


def test_get_rotated_coords():
    points = [
        (399196.65235161263, 6412085.989691866, 126.69240350277111),
        (399242.50670129404, 6412023.861878326, 128.67240350277112),
    ]
    rotated_points = get_rotated_coords(points, angle=30)
    expected_rotated_points = [
        (-2860328.5528037306, 5752627.684499246, 126.69240350277111),
        (-2860257.777865263, 5752596.807409281, 128.67240350277112),
    ]
    assert rotated_points == expected_rotated_points


@pytest.mark.parametrize(
    "coords, coords_rotated, coords_rotated_reordered, expected",
    [
        (
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)],  # coords
            [(0, 1, 0), (0, 0, 1), (1, 0, 0)],  # coords_rotated
            [(0, 0, 1), (1, 0, 0), (0, 1, 0)],  # coords_rotated_reordered
            [(0, 1, 0), (0, 0, 1), (1, 0, 0)],  # expected
        ),
        (
            [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
            [(4, 5, 6), (7, 8, 9), (1, 2, 3)],
            [(7, 8, 9), (1, 2, 3), (4, 5, 6)],
            [(4, 5, 6), (7, 8, 9), (1, 2, 3)],
        ),
        (
            [(1, 1, 1), (2, 2, 2), (3, 3, 3)],
            [(3, 3, 3), (1, 1, 1), (2, 2, 2)],
            [(1, 1, 1), (2, 2, 2), (3, 3, 3)],
            [(2, 2, 2), (3, 3, 3), (1, 1, 1)],
        ),
    ],
)
def test_retrieve_reorder_coords(
    coords, coords_rotated, coords_rotated_reordered, expected
):
    retrieved_coords = retrieve_reorder_coords(
        coords=coords,
        coords_rotated=coords_rotated,
        coords_rotated_reordered=coords_rotated_reordered,
    )
    assert retrieved_coords == expected


@pytest.mark.parametrize(
    "p1, p2, axis, expected_dist",
    [
        ((1.0, 2.0, 3.0), (2.0, 3.0, 4.0), 0, 1),
        ((1.0, 3.0, 3.0), (2.0, 3.0, 4.0), 1, 0),
        ((1.0, 3.0, 7.5), (2.0, 3.0, 5.0), 2, 2.5),
    ],
)
def test_dist(p1, p2, axis, expected_dist):
    assert dist(p1, p2, axis=axis) == expected_dist


@pytest.mark.parametrize(
    "input_coords, dist_min, key, flip, axis, expected",
    [
        (
            [
                (0, 0, 0),
                (0, 1, 0),
                (0, 2, 0),
                (0, 3, 0),
                (1, 0, 0),
                (1, 1, 0),
                (1, 2, 0),
                (1, 3, 0),
            ],  # input_coords
            0.9,  # dist_min
            lambda c: c[1],  # key
            True,  # flip
            0,  # axis
            [
                (0, 3, 0),
                (0, 2, 0),
                (0, 1, 0),
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (1, 2, 0),
                (1, 3, 0),
            ],  # expected
        ),
        (
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (3, 1),
                (4, 1),
                (5, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (5, 2),
                (6, 2),
            ],
            0.9,
            lambda c: c[0],
            False,
            1,
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (6, 0),
                (5, 1),
                (4, 1),
                (3, 1),
                (2, 1),
                (1, 1),
                (0, 1),
                (0, 2),
                (1, 2),
                (2, 2),
                (5, 2),
                (6, 2),
            ],
        ),
    ],
)
def test_flip_columns_1_out_of_2(
    input_coords, dist_min, key, flip, axis, expected
):
    flipped_coords = flip_columns_1_out_of_2(
        input_coords, dist_min=dist_min, key=key, flip=flip, axis=axis
    )
    print(flipped_coords)
    assert flipped_coords == expected


@pytest.mark.parametrize(
    "coords, sort_key, flip_key, dist_min, flip, axis, expected",
    [
        (
            [
                (1.0, 2.0, 105.0),
                (2.0, 3.0, 105.0),
                (3.0, 4.0, 105.0),
                (4.0, 5.0, 105.0),
                (5.0, 6.0, 105.0),
            ],
            lambda c: [-c[1], c[0]],
            lambda c: c[0],
            1.0,
            True,
            1,
            [
                (5.0, 6.0, 105.0),
                (4.0, 5.0, 105.0),
                (3.0, 4.0, 105.0),
                (2.0, 3.0, 105.0),
                (1.0, 2.0, 105.0),
            ],
        ),
        (
            [
                (1.0, 1.0, 105.0),
                (2.0, 1.0, 105.0),
                (3.0, 1.0, 105.0),
                (4.0, 1.0, 105.0),
                (5.0, 1.0, 105.0),
            ],
            lambda c: [-c[1], c[0]],
            lambda c: c[0],
            1.0,
            True,
            1,
            [
                (5.0, 1.0, 105.0),
                (4.0, 1.0, 105.0),
                (3.0, 1.0, 105.0),
                (2.0, 1.0, 105.0),
                (1.0, 1.0, 105.0),
            ],
        ),
        (
            [
                (1.0, 1.0, 105.0),
                (2.0, 2.0, 105.0),
                (3.0, 3.0, 105.0),
                (4.0, 4.0, 105.0),
                (5.0, 5.0, 105.0),
            ],
            lambda c: [-c[1], c[0]],
            lambda c: c[0],
            1.0,
            False,
            1,
            [
                (1.0, 1.0, 105.0),
                (2.0, 2.0, 105.0),
                (3.0, 3.0, 105.0),
                (4.0, 4.0, 105.0),
                (5.0, 5.0, 105.0),
            ],
        ),
    ],
)
def test_reorder_coords(
    coords, sort_key, flip_key, dist_min, flip, axis, expected
):
    actual = reorder_coords(coords, sort_key, flip_key, dist_min, flip, axis)
    assert actual == expected


def test_coordinates_converter():
    class DummyCoordinatesConverter(CoordinatesConverter):
        def convert(self) -> list[tuple[float, float, float]]:
            return [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]

    converter = DummyCoordinatesConverter()
    converted_coords = converter.convert()
    assert converted_coords == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]


def test_lambert93_to_wgs84_convertor():
    convertor = Lambert93ToWgs84Convertor(
        df=pd.DataFrame(
            data={
                "x": [700000, 800000],
                "y": [6600000, 7600000],
                "z": [100, 200],
            }
        ),
        delta=10,
    )

    converted_coords = convertor.convert()

    expected_coords = [
        (3.0000000000000004, 46.499999999999986, 110.0),
        (4.561673860392856, 55.45063458704315, 210.0),
    ]

    assert pytest.approx(converted_coords) == expected_coords


@pytest.mark.parametrize(
    "kml_file, parcelle_type",
    [
        (
            "kmz/output/castillon/castillon_premiere_colonne_delta10_WGS84.kml",
            ParcelleType.COL0,
        ),
        (
            "kmz/output/castillon/castillon_parcelle_C1_delta10_WGS84.kml",
            ParcelleType.C1,
        ),
        (
            "kmz/output/castillon/castillon_parcelle_C2_delta10_WGS84.kml",
            ParcelleType.C2,
        ),
        (
            "kmz/output/castillon/castillon_parcelle_entiere_delta10_WGS84.kml",
            ParcelleType.ALL,
        ),
    ],
)
def test_get_path_for_castillon(kml_file, parcelle_type):
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    coords_text = (
        ET.parse(kml_file)
        .getroot()
        .find(".//kml:LineString", ns)
        .find("kml:coordinates", ns)
        .text.strip()
    )
    coords_parsed = [
        tuple(map(float, coord.split(","))) for coord in coords_text.split(" ")
    ]
    actual_coords = get_path_for_castillon(
        df=pd.read_csv("kmz/input/cimes_castillon1et2.csv"),
        threshold_z=10,
        parcelle_type=parcelle_type,
    )
    actual_coords_converted = Lambert93ToWgs84Convertor(
        df=pd.DataFrame(actual_coords, columns=["x", "y", "z"]),
        delta=10,
    ).convert()
    assert coords_parsed == actual_coords_converted


def test_kml_creator():
    kml_creator = KmlCreator(
        coords=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)],
        kml_name="/tmp/test_kml_creation.kml",
    )
    kml_creator.create_kml()
    assert os.path.exists(kml_creator.kml_name)

    # Parse the KML file and verify its content
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    coords_text = (
        ET.parse(kml_creator.kml_name)
        .getroot()
        .find(".//kml:LineString", ns)
        .find("kml:coordinates", ns)
        .text.strip()
    )
    coords_parsed = [
        tuple(map(float, coord.split(","))) for coord in coords_text.split(" ")
    ]
    assert coords_parsed == kml_creator.coords

    # Clean up the test KML file
    os.remove(kml_creator.kml_name)
