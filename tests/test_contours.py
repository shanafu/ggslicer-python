import numpy as np
import pytest
import SimpleITK as sitk

from ggslicer import SliceGeometry, slice_contours, slice_label_contours


def circle_image(make_image, n=20, radius=5, center=(10, 10)):
    return make_image(
        (n, n, n),
        lambda idx: 1.0 if (idx[0] - center[0]) ** 2 + (idx[1] - center[1]) ** 2 <= radius ** 2 else 0.0,
    )


def two_band_label_image(make_image, n=20):
    return make_image(
        (n, n, n),
        lambda idx: 1.0 if idx[0] < 10 else (2.0 if idx[0] < 15 else 0.0),
    )


def test_slice_contours_traces_a_circle_at_the_expected_radius(make_image):
    img = circle_image(make_image)
    out = slice_contours(img, axis="z", coordinate=10, levels=[0.5])

    assert {"package", "k", "level", "obj", "vertex", "x", "y", "z"}.issubset(out.columns)
    assert len(out) > 0
    assert (out["z"] == 10).all()

    dist_from_center = np.sqrt((out["x"] - 10) ** 2 + (out["y"] - 10) ** 2)
    assert (np.abs(dist_from_center - 5) < 1.5).all()


def test_slice_contours_supports_multiple_levels_in_one_call(make_image):
    img = make_image((20, 20, 20), lambda idx: idx[0] / 2)
    out = slice_contours(img, axis="z", coordinate=5, levels=[2, 5, 8])
    assert set(out["level"].unique()) == {2, 5, 8}


def test_slice_contours_errors_on_invalid_mask_fill_or_empty_levels(make_image):
    img = circle_image(make_image)
    with pytest.raises(ValueError):
        slice_contours(img, axis="z", coordinate=10, levels=[0.5], mask_fill="nope")
    with pytest.raises(ValueError):
        slice_contours(img, axis="z", coordinate=10, levels=[])


def test_mask_fill_zero_fabricates_a_boundary_that_nan_does_not(make_image):
    img = two_band_label_image(make_image)
    mask = make_image((20, 20, 20), lambda idx: 1.0 if idx[0] >= 5 else 0.0)

    out_zero = slice_contours(img, axis="z", coordinate=10, levels=[0.5], mask=mask, mask_fill="zero")
    out_nan = slice_contours(img, axis="z", coordinate=10, levels=[0.5], mask=mask, mask_fill="nan")

    assert len(out_zero) > len(out_nan)
    assert (out_zero["x"] < 7).any()
    assert not (out_nan["x"] < 7).any()


def test_slice_label_contours_traces_each_labels_own_boundary_separately(make_image):
    img = two_band_label_image(make_image)
    out = slice_label_contours(img, axis="z", coordinate=10)

    assert set(out["label"].unique()) == {1.0, 2.0}
    assert {"package", "k", "label", "obj", "vertex", "x", "y", "z"}.issubset(out.columns)
    assert (out["label"] == 2.0).sum() > (out["label"] == 1.0).sum()


def test_slice_label_contours_respects_the_labels_subsetting_argument(make_image):
    img = two_band_label_image(make_image)
    out = slice_label_contours(img, axis="z", coordinate=10, labels=[1])
    assert set(out["label"].unique()) == {1.0}


def test_min_vertices_drops_small_contour_paths_while_keeping_larger_ones(make_image):
    def value_fn(idx):
        is_big = (idx[0] - 15) ** 2 + (idx[1] - 15) ** 2 <= 64
        is_island = idx[0] == 2 and idx[1] == 2
        return 1.0 if (is_big or is_island) else 0.0

    img = make_image((30, 30, 30), value_fn)

    out_unfiltered = slice_contours(img, axis="z", coordinate=15, levels=[0.5])
    n_paths_unfiltered = len(out_unfiltered.groupby(["package", "k", "level", "obj"]))
    assert n_paths_unfiltered > 1

    out_filtered = slice_contours(img, axis="z", coordinate=15, levels=[0.5], min_vertices=10)
    groups = out_filtered.groupby(["package", "k", "level", "obj"])
    assert len(groups) == 1
    assert (groups.size() >= 10).all()


def test_slice_contours_and_slice_label_contours_accept_the_geometry_escape_hatch(make_image):
    img = circle_image(make_image)
    geom = SliceGeometry.from_image_axis(img, "z", 10)
    out = slice_contours(img, geometry=geom, levels=[0.5])
    assert len(out) > 0

    label_img = two_band_label_image(make_image)
    geom2 = SliceGeometry.from_image_axis(label_img, "z", 10)
    outl = slice_label_contours(label_img, geometry=geom2)
    assert len(outl) > 0


def test_slice_contours_require_exactly_one_of_geometry_or_axis_and_coordinate(make_image):
    img = circle_image(make_image)
    geom = SliceGeometry.from_image_axis(img, "z", 10)
    with pytest.raises(ValueError):
        slice_contours(img, axis="z", coordinate=10, geometry=geom, levels=[0.5])
    with pytest.raises(ValueError, match="Provide either"):
        slice_contours(img, levels=[0.5])
    with pytest.raises(ValueError):
        slice_label_contours(img, geometry=geom, axis="z")
    with pytest.raises(ValueError, match="Provide either"):
        slice_label_contours(img)


def test_slice_contours_accepts_file_paths_for_image_and_mask(make_image, tmp_path):
    img = circle_image(make_image)
    mask = make_image((20, 20, 20), lambda idx: 1.0)
    img_path = str(tmp_path / "img.nii.gz")
    mask_path = str(tmp_path / "mask.nii.gz")
    sitk.WriteImage(img, img_path)
    sitk.WriteImage(mask, mask_path)

    out = slice_contours(img_path, axis="z", coordinate=10, levels=[0.5], mask=mask_path)
    assert len(out) > 0
