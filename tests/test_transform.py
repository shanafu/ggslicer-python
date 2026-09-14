import os

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

from ggslicer import transform_points, read_minc_transform


def test_transform_points_applies_a_translation_transform_exactly():
    df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 0, 0], "z": [0, 0, 0], "label": ["a", "b", "c"]})
    t = sitk.TranslationTransform(3, (10, 5, -2))
    out = transform_points(df, t)

    assert list(out["x"]) == [10, 11, 12]
    assert list(out["y"]) == [5, 5, 5]
    assert list(out["z"]) == [-2, -2, -2]
    assert list(out["label"]) == list(df["label"])


def test_transform_points_with_invert_applies_the_inverse_transform():
    df = pd.DataFrame({"x": [10], "y": [5], "z": [-2]})
    t = sitk.TranslationTransform(3, (10, 5, -2))
    out = transform_points(df, t, invert=True)
    assert np.allclose([out["x"][0], out["y"][0], out["z"][0]], [0, 0, 0])


def test_transform_points_errors_clearly_when_inverting_a_displacement_field_transform():
    vec_img = sitk.Image([3, 3, 3], sitk.sitkVectorFloat64, 3)
    t = sitk.DisplacementFieldTransform(vec_img)
    df = pd.DataFrame({"x": [0], "y": [0], "z": [0]})
    with pytest.raises(RuntimeError):
        transform_points(df, t, invert=True)


def test_transform_points_accepts_a_transform_given_as_a_file_path(tmp_path):
    t = sitk.TranslationTransform(3, (1, 2, 3))
    path = str(tmp_path / "t.tfm")
    sitk.WriteTransform(t, path)

    df = pd.DataFrame({"x": [0], "y": [0], "z": [0]})
    out = transform_points(df, path)
    assert np.allclose([out["x"][0], out["y"][0], out["z"][0]], [1, 2, 3])


def test_transform_points_respects_custom_column_names():
    df = pd.DataFrame({"px": [0], "py": [0], "pz": [0], "other": ["kept"]})
    t = sitk.TranslationTransform(3, (1, 2, 3))
    out = transform_points(df, t, x_col="px", y_col="py", z_col="pz")
    assert np.allclose([out["px"][0], out["py"][0], out["pz"][0]], [1, 2, 3])
    assert out["other"][0] == "kept"


def _write_test_xfm(path, body_lines):
    with open(path, "w") as f:
        f.write("MNI Transform File\n%test\n\n")
        f.write("\n".join(body_lines) + "\n")


def test_read_minc_transform_parses_a_single_linear_block(tmp_path):
    path = str(tmp_path / "t.xfm")
    _write_test_xfm(path, [
        "Transform_Type = Linear;",
        "Linear_Transform =",
        " 1 0 0 5",
        " 0 1 0 6",
        " 0 0 1 7;",
    ])

    # corrected=False: this test is about the block-parsing logic itself,
    # not the ReadImage_fix()-compatibility conjugation (see its own tests
    # below), so check the transform exactly as written in the file.
    t = read_minc_transform(path, corrected=False)
    out = t.TransformPoint((0, 0, 0))
    assert np.allclose(out, [5, 6, 7])


def test_read_minc_transform_parses_a_single_grid_transform_block(tmp_path):
    grid_path = str(tmp_path / "grid.mnc")
    vec_img = sitk.Image([3, 3, 3], sitk.sitkVectorFloat64, 3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                vec_img.SetPixel((i, j, k), (1.0, 2.0, 3.0))
    sitk.WriteImage(vec_img, grid_path)

    xfm_path = str(tmp_path / "test.xfm")
    _write_test_xfm(xfm_path, [
        "Transform_Type = Grid_Transform;",
        "Displacement_Volume = grid.mnc;",
    ])

    t = read_minc_transform(xfm_path, corrected=False)
    out = t.TransformPoint((0, 0, 0))
    assert np.allclose(out, [1, 2, 3])


def test_read_minc_transform_concatenates_multiple_blocks_in_the_correct_order(tmp_path):
    grid_path = str(tmp_path / "grid.mnc")
    vec_img = sitk.Image([3, 3, 3], sitk.sitkVectorFloat64, 3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                vec_img.SetPixel((i, j, k), (100.0, 0.0, 0.0))
    sitk.WriteImage(vec_img, grid_path)

    xfm_path = str(tmp_path / "test.xfm")
    _write_test_xfm(xfm_path, [
        "Transform_Type = Linear;",
        "Linear_Transform =",
        " 2 0 0 0",
        " 0 2 0 0",
        " 0 0 2 0;",
        "Transform_Type = Grid_Transform;",
        "Displacement_Volume = grid.mnc;",
    ])

    t = read_minc_transform(xfm_path, corrected=False)
    assert isinstance(t, sitk.CompositeTransform)

    # file order [Linear, Grid]: apply linear (scale by 2) first, then grid (+100 in x)
    out = t.TransformPoint((1, 0, 0))
    assert np.allclose(out, [102, 0, 0])


def test_read_minc_transform_resolves_displacement_volume_relative_to_xfm_directory(tmp_path):
    vec_img = sitk.Image([2, 2, 2], sitk.sitkVectorFloat64, 3)
    sitk.WriteImage(vec_img, str(tmp_path / "somegrid.mnc"))

    xfm_path = str(tmp_path / "test.xfm")
    _write_test_xfm(xfm_path, [
        "Transform_Type = Grid_Transform;",
        "Displacement_Volume = somegrid.mnc;",
    ])

    read_minc_transform(xfm_path, corrected=False)  # should not raise


def test_read_minc_transform_errors_clearly_on_malformed_or_unsupported_input(tmp_path):
    not_an_xfm = str(tmp_path / "not.xfm")
    with open(not_an_xfm, "w") as f:
        f.write("not a transform file")
    with pytest.raises(ValueError):
        read_minc_transform(not_an_xfm)

    bad_linear = str(tmp_path / "bad.xfm")
    _write_test_xfm(bad_linear, ["Transform_Type = Linear;", "Linear_Transform =", " 1 2 3;"])
    with pytest.raises(ValueError):
        read_minc_transform(bad_linear)

    unsupported = str(tmp_path / "unsupported.xfm")
    _write_test_xfm(unsupported, ["Transform_Type = Thin_Plate_Spline_Transform;"])
    with pytest.raises(ValueError, match="Unsupported"):
        read_minc_transform(unsupported)


def test_transform_points_routes_xfm_paths_through_read_minc_transform(tmp_path):
    path = str(tmp_path / "t.xfm")
    _write_test_xfm(path, [
        "Transform_Type = Linear;",
        "Linear_Transform =",
        " 1 0 0 1",
        " 0 1 0 2",
        " 0 0 1 3;",
    ])

    # transform_points()/_read_transform_file() call read_minc_transform(path)
    # with its default corrected=True, so the result is conjugated: for
    # point (0,0,0), N(T_raw(N(0,0,0))) = N(T_raw(0,0,0)) = N(1,2,3) = (-1,-2,3)
    # (N negates x/y only -- see _conjugate_minc_native_transform()).
    df = pd.DataFrame({"x": [0], "y": [0], "z": [0]})
    out = transform_points(df, path)
    assert np.allclose([out["x"][0], out["y"][0], out["z"][0]], [-1, -2, 3])


def test_read_minc_transform_default_corrected_conjugates_by_negating_xy(tmp_path):
    path = str(tmp_path / "t.xfm")
    _write_test_xfm(path, [
        "Transform_Type = Linear;",
        "Linear_Transform =",
        " 1 0 0 1",
        " 0 1 0 2",
        " 0 0 1 3;",
    ])

    t_raw = read_minc_transform(path, corrected=False)
    t_corrected = read_minc_transform(path)  # default True

    p = (4, 5, 6)
    n_p = (-p[0], -p[1], p[2])
    raw_result = t_raw.TransformPoint(n_p)
    expected = (-raw_result[0], -raw_result[1], raw_result[2])

    assert np.allclose(t_corrected.TransformPoint(p), expected)
    # z is never touched by the conjugation
    assert np.isclose(
        t_corrected.TransformPoint((0, 0, 9))[2],
        t_raw.TransformPoint((0, 0, 9))[2],
    )


def test_read_minc_transform_conjugation_applies_to_multi_block_file(tmp_path):
    grid_path = str(tmp_path / "grid.mnc")
    vec_img = sitk.Image([3, 3, 3], sitk.sitkVectorFloat64, 3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                vec_img.SetPixel((i, j, k), (10.0, 20.0, 30.0))
    sitk.WriteImage(vec_img, grid_path)

    xfm_path = str(tmp_path / "test.xfm")
    _write_test_xfm(xfm_path, [
        "Transform_Type = Linear;",
        "Linear_Transform =",
        " 1 0 0 1",
        " 0 1 0 2",
        " 0 0 1 3;",
        "Transform_Type = Grid_Transform;",
        "Displacement_Volume = grid.mnc;",
    ])

    t_raw = read_minc_transform(xfm_path, corrected=False)
    t_corrected = read_minc_transform(xfm_path)

    p = (0.5, 0.5, 0.5)
    n_p = (-p[0], -p[1], p[2])
    raw_result = t_raw.TransformPoint(n_p)
    expected = (-raw_result[0], -raw_result[1], raw_result[2])
    assert np.allclose(t_corrected.TransformPoint(p), expected)
