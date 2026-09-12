import numpy as np
import pandas as pd
import pytest

from ggslicer import SliceGeometry


def test_constructor_validates_its_inputs():
    with pytest.raises(ValueError, match="length-3"):
        SliceGeometry([0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [2, 2])
    with pytest.raises(ValueError, match="unit vector"):
        SliceGeometry([0, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1], [2, 2])
    with pytest.raises(ValueError, match="orthogonal"):
        SliceGeometry([0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1], [2, 2])
    with pytest.raises(ValueError, match="spacing"):
        SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1], [2, 2])
    with pytest.raises(ValueError, match="size"):
        SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [0, 2])
    with pytest.raises(ValueError, match="size"):
        SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [2.5, 2])


def test_properties_return_the_constructed_geometry():
    s = SliceGeometry(origin=[1, 2, 3], direction_i=[1, 0, 0], direction_j=[0, 1, 0], spacing=[1, 2], size=[3, 4])
    assert np.allclose(s.origin, [1, 2, 3])
    assert np.allclose(s.direction_i, [1, 0, 0])
    assert np.allclose(s.direction_j, [0, 1, 0])
    assert np.allclose(s.spacing, [1, 2])
    assert list(s.size) == [3, 4]
    assert np.allclose(s.normal, [0, 0, 1])
    assert np.allclose(s.extent, [2, 6])
    assert np.allclose(s.center, [1 + 1, 2 + 3, 3])


def test_bounds_returns_the_four_corners():
    s = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 2], [3, 4])
    b = s.bounds
    assert len(b) == 4
    assert set(b.columns) == {"corner", "x", "y", "z"}
    row = b[b["corner"] == "i1_j1"]
    assert np.allclose(row[["x", "y", "z"]].to_numpy()[0], [2, 6, 0])


def test_sample_points_is_correct_and_0_indexed():
    s = SliceGeometry(origin=[0, 0, 0], direction_i=[1, 0, 0], direction_j=[0, 1, 0], spacing=[1, 2], size=[3, 4])
    pts = s.sample_points
    assert len(pts) == 12
    assert list(pts.columns) == ["i", "j", "x", "y", "z"]
    assert pts["i"].min() == 0 and pts["i"].max() == 2
    assert pts["j"].min() == 0 and pts["j"].max() == 3
    corner = pts[(pts["i"] == 2) & (pts["j"] == 3)]
    assert np.allclose(corner[["x", "y", "z"]].to_numpy()[0], [2, 6, 0])


def test_sample_points_handles_an_oblique_plane():
    di = np.array([1, 1, 0]) / np.sqrt(2)
    dj = np.array([-1, 1, 0]) / np.sqrt(2)
    s = SliceGeometry(origin=[5, 5, 5], direction_i=di, direction_j=dj, spacing=[1, 1], size=[2, 2])
    pts = s.sample_points
    p10 = pts[(pts["i"] == 1) & (pts["j"] == 0)][["x", "y", "z"]].to_numpy()[0]
    p01 = pts[(pts["i"] == 0) & (pts["j"] == 1)][["x", "y", "z"]].to_numpy()[0]
    assert np.allclose(p10, np.array([5, 5, 5]) + di)
    assert np.allclose(p01, np.array([5, 5, 5]) + dj)


def test_sample_points_is_memoized_and_invalidated_by_setters():
    s = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [2, 2])
    pts1 = s.sample_points
    pts2 = s.sample_points
    assert pts1 is pts2

    s.origin = [10, 0, 0]
    pts3 = s.sample_points
    assert pts3 is not pts1
    assert pts3["x"].min() == 10


def test_each_setter_invalidates_the_cache():
    s = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [2, 2])

    s.sample_points
    s.set_direction([0, 1, 0], [-1, 0, 0])
    assert np.allclose(s.direction_i, [0, 1, 0])

    s.sample_points
    s.spacing = [5, 5]
    assert (s.sample_points["x"] % 5 == 0).all()

    s.sample_points
    s.size = [4, 4]
    assert len(s.sample_points) == 16


def test_translate_returns_a_new_independent_slice_geometry():
    s = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [2, 2])
    normal = s.normal
    s2 = s.translate(5 * normal)

    assert np.allclose(s.origin, [0, 0, 0])
    assert np.allclose(s2.origin, [0, 0, 5])
    assert np.allclose(s2.direction, s.direction)
    assert np.allclose(s2.spacing, s.spacing)
    assert np.array_equal(s2.size, s.size)

    with pytest.raises(ValueError, match="length-3"):
        s.translate([1, 2])


def test_set_center_repositions_keeping_direction_spacing_size():
    s = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [3, 3])
    s.center = [10, 10, 0]
    assert np.allclose(s.center, [10, 10, 0])
    assert np.allclose(s.origin, [9, 9, 0])
    assert np.allclose(s.spacing, [1, 1])
    assert list(s.size) == [3, 3]
    with pytest.raises(ValueError):
        s.set_center([1, 2])


def test_set_spacing_i_j_update_one_axis_independently():
    s = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [3, 3])
    s.set_spacing_i(2)
    assert np.allclose(s.spacing, [2, 1])
    s.set_spacing_j(3)
    assert np.allclose(s.spacing, [2, 3])
    with pytest.raises(ValueError, match="> 0"):
        s.set_spacing_i(0)
    with pytest.raises(ValueError, match="> 0"):
        s.set_spacing_j(-1)


def test_set_size_i_j_update_one_axis_independently():
    s = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [3, 3])
    s.set_size_i(5)
    assert list(s.size) == [5, 3]
    s.set_size_j(6)
    assert list(s.size) == [5, 6]
    with pytest.raises(ValueError, match=">= 1"):
        s.set_size_i(0)
    with pytest.raises(ValueError, match=">= 1"):
        s.set_size_j(2.5)


def test_set_extent_derives_spacing_or_size():
    s = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [5, 5])
    s.set_extent([8, 8], adjust="spacing")
    assert np.allclose(s.spacing, [2, 2])
    assert list(s.size) == [5, 5]

    s.set_extent([8, 8], adjust="size")
    assert list(s.size) == [5, 5]

    with pytest.raises(ValueError, match="non-negative"):
        s.set_extent([-1, 1])

    s_single = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [1, 3])
    with pytest.raises(ValueError, match="Cannot derive"):
        s_single.set_extent([4, 4], adjust="spacing")

    s.extent = [4, 4]  # property setter defaults to adjust="spacing"
    assert np.allclose(s.extent, [4, 4])


def test_from_center_builds_a_slice_centered_on_the_given_point():
    s = SliceGeometry.from_center([1, 2, 3], [1, 0, 0], [0, 1, 0], [1, 1], [3, 3])
    assert np.allclose(s.center, [1, 2, 3])
    assert list(s.size) == [3, 3]


def test_from_normal_derives_an_orthonormal_basis_with_the_requested_normal():
    s = SliceGeometry.from_normal(origin=[0, 0, 0], normal=[0, 0, 5], spacing=[1, 1], size=[3, 3])
    assert np.allclose(s.normal, [0, 0, 1])

    s2 = SliceGeometry.from_normal(
        origin=[0, 0, 0], normal=[1, 0, 0], spacing=[1, 1], size=[3, 3], direction_i=[1, 1, 0]
    )
    assert np.allclose(s2.normal, [1, 0, 0])
    assert np.isclose(np.dot(s2.direction_i, s2.normal), 0)

    with pytest.raises(ValueError, match="nonzero"):
        SliceGeometry.from_normal([0, 0, 0], [0, 0, 0], [1, 1], [3, 3])
    with pytest.raises(ValueError, match="parallel"):
        SliceGeometry.from_normal([0, 0, 0], [0, 0, 1], [1, 1], [3, 3], direction_i=[0, 0, 2])


def test_from_image_axis_matches_the_source_images_own_geometry(make_image):
    img = make_image((10, 10, 10), lambda idx: 0, origin=(-5, -5, -5), spacing=(1, 1, 1))

    s = SliceGeometry.from_image_axis(img, "z", 0)
    assert list(s.size) == [10, 10]
    assert np.allclose(s.spacing, [1, 1])
    assert np.allclose(s.origin, [-5, -5, 0])

    s_alias = SliceGeometry.from_image_axis(img, "axial", 3.4)
    assert np.isclose(s_alias.origin[2], 3)

    s_num = SliceGeometry.from_image_axis(img, 3, 0)
    assert np.allclose(s_num.origin, s.origin)

    with pytest.raises(ValueError, match="Invalid `axis`"):
        SliceGeometry.from_image_axis(img, "bogus", 0)

    img2d = make_image((4, 4), lambda idx: 0)
    with pytest.raises(ValueError, match="3D image"):
        SliceGeometry.from_image_axis(img2d, "z", 0)


def test_from_bounds_round_trips_bounds_and_validates_consistency():
    s = SliceGeometry([1, 2, 3], [1, 0, 0], [0, 1, 0], [1, 2], [4, 5])
    b = s.bounds

    s2 = SliceGeometry.from_bounds(b, size=s.size)
    assert np.allclose(s2.origin, s.origin)
    assert np.allclose(s2.direction_i, s.direction_i)
    assert np.allclose(s2.direction_j, s.direction_j)
    assert list(s2.size) == list(s.size)

    s3 = SliceGeometry.from_bounds(b, spacing=s.spacing)
    assert list(s3.size) == list(s.size)

    with pytest.raises(ValueError, match="exactly one"):
        SliceGeometry.from_bounds(b, spacing=[1, 1], size=[4, 5])

    missing = b[b["corner"] != "i1_j0"]
    with pytest.raises(ValueError, match="must contain rows"):
        SliceGeometry.from_bounds(missing, size=[4, 5])

    b_bad = b.copy()
    b_bad.loc[b_bad["corner"] == "i1_j1", "x"] = 999
    with pytest.raises(ValueError, match="inconsistent"):
        SliceGeometry.from_bounds(b_bad, size=s.size)

    b_degenerate = b.copy()
    b_degenerate.loc[b_degenerate["corner"] == "i1_j0", ["x", "y", "z"]] = b_degenerate.loc[
        b_degenerate["corner"] == "i0_j0", ["x", "y", "z"]
    ].to_numpy()
    with pytest.raises(ValueError, match="Degenerate"):
        SliceGeometry.from_bounds(b_degenerate, size=[4, 5])


def test_as_sitk_reference_image_matches_the_manual_geometry():
    s = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 2], [3, 4])
    ref = s.as_sitk_reference_image(spacing_k=0.5)

    assert ref.GetSize() == (3, 4, 1)
    assert ref.GetOrigin() == (0, 0, 0)
    assert ref.GetSpacing() == (1, 2, 0.5)

    pt = ref.TransformIndexToPhysicalPoint((2, 3, 0))
    expected = s.sample_points[(s.sample_points["i"] == 2) & (s.sample_points["j"] == 3)][["x", "y", "z"]].to_numpy()[0]
    assert np.allclose(pt, expected)


def test_str_and_repr_run_without_error():
    s = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [2, 2])
    assert "<SliceGeometry>" in str(s)
    assert "normal:" in str(s)
    assert "extent:" in str(s)
    assert "SliceGeometry(" in repr(s)
