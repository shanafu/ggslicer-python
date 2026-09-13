import numpy as np
import pytest
import SimpleITK as sitk

from ggslicer import (
    SliceGeometry,
    SlicePackage,
    SlicePackageSet,
    slice_grid,
    slice_grid_layers,
)


def simple_image(make_image, n=20):
    return make_image((n, n, n), lambda idx: 0.0)


def test_slice_grid_returns_the_expected_columns_and_both_parts(make_image):
    img = simple_image(make_image)
    geom = SliceGeometry.from_image_axis(img, "z", 10)
    out = slice_grid(geometry=geom)

    assert {"package", "k", "part", "grid_axis", "line_id", "vertex", "x", "y", "z"}.issubset(out.columns)
    assert set(out["part"].unique()) == {"box", "grid"}
    assert set(out["grid_axis"].unique()) == {"i", "j"}


def test_box_sits_exactly_at_the_slices_true_extent_when_padding_is_zero(make_image):
    img = simple_image(make_image)
    geom = SliceGeometry.from_image_axis(img, "z", 10)
    extent = geom.extent

    out = slice_grid(geometry=geom, padding=0)
    box = out[out["part"] == "box"]
    assert np.isclose(box["x"].max() - box["x"].min(), extent[0])
    assert np.isclose(box["y"].max() - box["y"].min(), extent[1])


def test_padding_extends_the_box_beyond_the_slices_true_extent(make_image):
    img = simple_image(make_image)
    geom = SliceGeometry.from_image_axis(img, "z", 10)
    extent = geom.extent

    out = slice_grid(geometry=geom, padding=2)
    box = out[out["part"] == "box"]
    assert np.isclose(box["x"].max() - box["x"].min(), extent[0] + 4)
    assert np.isclose(box["y"].max() - box["y"].min(), extent[1] + 4)


def test_default_spacing_is_scale_adaptive(make_image):
    img = simple_image(make_image, 40)
    geom = SliceGeometry.from_image_axis(img, "z", 20)
    extent = geom.extent
    expected_spacing = min(extent) / 10

    out = slice_grid(geometry=geom)
    grid_i = out[(out["part"] == "grid") & (out["grid_axis"] == "i")]
    coords = sorted(grid_i.groupby("line_id")["y"].first().round(6).unique())
    observed_spacing = np.diff(coords)
    assert np.allclose(observed_spacing, expected_spacing, atol=1e-6)


def test_explicit_spacing_and_point_spacing_control_line_count_and_density(make_image):
    img = simple_image(make_image, 40)
    geom = SliceGeometry.from_image_axis(img, "z", 20)
    extent = geom.extent

    out = slice_grid(geometry=geom, spacing=5, point_spacing=1, padding=0)
    grid_i = out[(out["part"] == "grid") & (out["grid_axis"] == "i")]
    n_lines = grid_i["line_id"].nunique()
    expected_n_lines = len(np.arange(0, int(np.floor(extent[0] / 5 + 1e-9)) + 1))
    assert n_lines == expected_n_lines

    one_line = grid_i[grid_i["line_id"] == grid_i["line_id"].iloc[0]]
    assert len(one_line) == round(extent[1] / 1) + 1


def test_each_slice_in_a_multi_slice_slicepackage_gets_its_own_grid(make_image):
    img = simple_image(make_image)
    pkg = SlicePackage.from_image_axis(img, "z")
    pkg.size_k = 3
    out = slice_grid(geometry=pkg)
    assert set(out["k"].unique()) == {0, 1, 2}
    counts = out.groupby("k").size()
    assert counts.nunique() == 1


def test_each_package_in_a_slicepackageset_gets_its_own_independent_grid(make_image):
    img = simple_image(make_image)
    pset = SlicePackageSet.from_orthogonal_triplet(img, {"x": 0, "y": 0, "z": 0})
    out = slice_grid(geometry=pset)
    assert set(out["package"].unique()) == {"sagittal", "coronal", "axial"}


def test_slice_grid_requires_exactly_one_of_geometry_or_axis_and_coordinate(make_image):
    img = simple_image(make_image)
    geom = SliceGeometry.from_image_axis(img, "z", 10)
    with pytest.raises(ValueError):
        slice_grid(img, axis="z", coordinate=10, geometry=geom)
    with pytest.raises(ValueError, match="Provide either"):
        slice_grid()
    with pytest.raises(ValueError, match="Provide either"):
        slice_grid(axis="z")


def test_slice_grid_requires_image_when_geometry_is_not_supplied(make_image):
    with pytest.raises(ValueError, match="image.*required"):
        slice_grid(axis="z", coordinate=10)


def test_slice_grid_validates_spacing_point_spacing_padding(make_image):
    img = simple_image(make_image)
    geom = SliceGeometry.from_image_axis(img, "z", 10)
    with pytest.raises(ValueError):
        slice_grid(geometry=geom, spacing=-1)
    with pytest.raises(ValueError):
        slice_grid(geometry=geom, point_spacing=0)
    with pytest.raises(ValueError):
        slice_grid(geometry=geom, padding=-1000)


def test_slice_grid_layers_returns_two_styled_layers_that_build_into_a_plot(make_image):
    from plotnine import ggplot, aes

    img = simple_image(make_image)
    geom = SliceGeometry.from_image_axis(img, "z", 10)
    grid_df = slice_grid(geometry=geom)

    layers = slice_grid_layers(grid_df, box_color="blue", grid_color="green")
    assert set(layers.keys()) == {"box", "grid"}
    assert layers["box"].aes_params["color"] == "blue"
    assert layers["grid"].aes_params["color"] == "green"

    plt = ggplot(grid_df, aes(x="x", y="y")) + layers["box"] + layers["grid"]
    assert plt is not None
