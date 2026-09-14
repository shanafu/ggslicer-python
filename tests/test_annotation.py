import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
from plotnine import ggplot, aes, geom_text

from ggslicer import (
    slice_label_contours,
    contour_centroids,
    slice_label_annotations,
    slice_label_annotations_layer,
)


def _two_blob_label_image(n=60):
    img = sitk.Image([n, n, 3], sitk.sitkFloat32)
    for i in range(n):
        for j in range(n):
            val = 0.0
            if (i - 15) ** 2 + (j - 15) ** 2 <= 64:
                val = 1.0
            if (i - 45) ** 2 + (j - 15) ** 2 <= 64:
                val = 1.0
            if (i - 30) ** 2 + (j - 45) ** 2 <= 100:
                val = 2.0
            for k in range(3):
                img.SetPixel((i, j, k), val)
    return img


def test_contour_centroids_finds_one_centroid_per_connected_component_not_per_label():
    img = _two_blob_label_image()
    contours = slice_label_contours(img, axis="z", coordinate=0)

    out = contour_centroids(contours)
    for col in ("package", "k", "label", "name", "obj", "x", "y", "z", "area"):
        assert col in out.columns
    assert len(out) == 3  # label 1 has 2 components, label 2 has 1

    label1 = out[out["label"] == 1]
    assert len(label1) == 2
    assert set(np.round(label1["x"])) == {15, 45}
    assert np.allclose(label1["y"], 15, atol=1e-6)

    label2 = out[out["label"] == 2]
    assert len(label2) == 1
    assert round(label2["x"].iloc[0]) == 30
    assert round(label2["y"].iloc[0]) == 45


def test_computed_areas_approximate_true_circle_areas():
    img = _two_blob_label_image()
    contours = slice_label_contours(img, axis="z", coordinate=0)
    out = contour_centroids(contours)

    assert np.all(np.abs(out.loc[out["label"] == 1, "area"] - np.pi * 64) < 15)
    assert np.all(np.abs(out.loc[out["label"] == 2, "area"] - np.pi * 100) < 15)


def test_label_names_maps_values_falling_back_to_numeric_id_when_unmapped():
    img = _two_blob_label_image()
    contours = slice_label_contours(img, axis="z", coordinate=0)

    named = contour_centroids(contours, label_names={1: "RegionA", 2: "RegionB"})
    assert all(named.loc[named["label"] == 1, "name"] == "RegionA")
    assert all(named.loc[named["label"] == 2, "name"] == "RegionB")

    unnamed = contour_centroids(contours)
    assert all(unnamed.loc[unnamed["label"] == 1, "name"] == "1")
    assert all(unnamed.loc[unnamed["label"] == 2, "name"] == "2")

    partial = contour_centroids(contours, label_names={1: "RegionA"})
    assert all(partial.loc[partial["label"] == 1, "name"] == "RegionA")
    assert all(partial.loc[partial["label"] == 2, "name"] == "2")


def test_min_area_filters_out_small_connected_components():
    img = _two_blob_label_image()
    contours = slice_label_contours(img, axis="z", coordinate=0)

    all_out = contour_centroids(contours)
    assert len(all_out) == 3

    filtered = contour_centroids(contours, min_area=250)
    assert len(filtered) == 1
    assert filtered["label"].iloc[0] == 2

    none = contour_centroids(contours, min_area=100000)
    assert len(none) == 0
    for col in ("package", "k", "label", "name", "obj", "x", "y", "z", "area"):
        assert col in none.columns


def test_contour_centroids_validates_its_arguments():
    img = _two_blob_label_image()
    contours = slice_label_contours(img, axis="z", coordinate=0)

    with pytest.raises(ValueError, match="columns"):
        contour_centroids(pd.DataFrame({"x": [1], "y": [1]}))
    with pytest.raises(ValueError):
        contour_centroids(contours, min_area=-1)


def test_slice_label_annotations_matches_manual_two_step():
    img = _two_blob_label_image()

    manual = contour_centroids(
        slice_label_contours(img, axis="z", coordinate=0),
        label_names={1: "RegionA", 2: "RegionB"}, min_area=10,
    )
    direct = slice_label_annotations(
        img, axis="z", coordinate=0,
        label_names={1: "RegionA", 2: "RegionB"}, min_area=10,
    )
    pd.testing.assert_frame_equal(manual.reset_index(drop=True), direct.reset_index(drop=True))


def test_slice_label_annotations_passes_min_vertices_through():
    img = _two_blob_label_image()
    out = slice_label_annotations(img, axis="z", coordinate=0, min_vertices=100000)
    assert len(out) == 0


def test_thin_arc_centroid_lands_within_its_own_band():
    # A thin ~80-degree arc slice of an annulus (radius 15-25) -- unlike a
    # nearly-*complete* ring (whose true area centroid is mathematically at
    # the ring's own geometric center, since a symmetric ring's mass is
    # evenly distributed all the way around it), a genuine one-sided arc has
    # no such symmetry, so its centroid should land within its own band.
    n = 60
    cx, cy = 30, 30
    img = sitk.Image([n, n, 3], sitk.sitkFloat32)
    for i in range(n):
        for j in range(n):
            dx, dy = i - cx, j - cy
            r = np.hypot(dx, dy)
            ang = np.arctan2(dy, dx)
            val = 1.0 if (15 <= r <= 25 and -0.7 < ang < 0.7) else 0.0
            for k in range(3):
                img.SetPixel((i, j, k), val)

    contours = slice_label_contours(img, axis="z", coordinate=0)
    out = contour_centroids(contours)
    assert len(out) == 1

    dist_from_center = np.hypot(out["x"].iloc[0] - cx, out["y"].iloc[0] - cy)
    assert 10 < dist_from_center < 25


def test_slice_label_annotations_layer_builds_a_usable_plotnine_layer():
    img = _two_blob_label_image()
    out = slice_label_annotations(img, axis="z", coordinate=0)

    layer = slice_label_annotations_layer(out)
    assert isinstance(layer, geom_text)

    p = ggplot(out, aes(x="x", y="y")) + layer
    fig = p.draw()
    assert fig is not None
