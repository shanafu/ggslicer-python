import numpy as np
import pytest

from ggslicer import SliceGeometry, SlicePackage


def base_slice(**overrides):
    defaults = dict(origin=[0, 0, 0], direction_i=[1, 0, 0], direction_j=[0, 1, 0], spacing=[1, 1], size=[3, 3])
    defaults.update(overrides)
    return SliceGeometry(**defaults)


def test_constructor_validates_its_inputs():
    with pytest.raises(ValueError, match="SliceGeometry"):
        SlicePackage(base_slice="not a slice", spacing_k=1, size_k=2)
    with pytest.raises(ValueError, match="> 0"):
        SlicePackage(base_slice=base_slice(), spacing_k=0, size_k=2)
    with pytest.raises(ValueError, match="> 0"):
        SlicePackage(base_slice=base_slice(), spacing_k=-1, size_k=2)
    with pytest.raises(ValueError, match=">= 1"):
        SlicePackage(base_slice=base_slice(), spacing_k=1, size_k=0)
    with pytest.raises(ValueError, match=">= 1"):
        SlicePackage(base_slice=base_slice(), spacing_k=1, size_k=1.5)


def test_size_spacing_normal_combine_base_slice_with_k_axis():
    pkg = SlicePackage(base_slice=base_slice(), spacing_k=2, size_k=4)
    assert list(pkg.size) == [3, 3, 4]
    assert np.allclose(pkg.spacing, [1, 1, 2])
    assert np.allclose(pkg.normal, [0, 0, 1])


def test_get_slice_returns_the_correctly_translated_kth_slice():
    pkg = SlicePackage(base_slice=base_slice(), spacing_k=2, size_k=4)
    assert np.allclose(pkg.get_slice(0).origin, [0, 0, 0])
    assert np.allclose(pkg.get_slice(2).origin, [0, 0, 4])
    assert np.allclose(pkg[2].origin, [0, 0, 4])
    assert list(pkg.get_slice(2).size) == list(pkg.base_slice.size)

    with pytest.raises(ValueError, match="0:\\(size_k"):
        pkg.get_slice(-1)
    with pytest.raises(ValueError, match="0:\\(size_k"):
        pkg.get_slice(4)
    with pytest.raises(ValueError, match="0:\\(size_k"):
        pkg.get_slice(1.5)


def test_sample_points_produces_the_full_3d_stack():
    pkg = SlicePackage(base_slice=base_slice(), spacing_k=2, size_k=4)
    pts = pkg.sample_points
    assert len(pts) == 3 * 3 * 4
    assert list(pts.columns) == ["i", "j", "k", "x", "y", "z"]
    row = pts[(pts["i"] == 0) & (pts["j"] == 0) & (pts["k"] == 2)]
    assert np.allclose(row[["x", "y", "z"]].to_numpy()[0], [0, 0, 4])


def test_sample_points_is_memoized_and_invalidated():
    pkg = SlicePackage(base_slice=base_slice(), spacing_k=1, size_k=2)
    pts1 = pkg.sample_points
    assert pkg.sample_points is pts1

    pkg.spacing_k = 3
    assert pkg.sample_points["z"].max() == 3

    pkg.size_k = 5
    assert len(pkg.sample_points) == 3 * 3 * 5


def test_as_sitk_reference_image_spans_the_whole_stack():
    pkg = SlicePackage(base_slice=base_slice(), spacing_k=2, size_k=4)
    ref = pkg.as_sitk_reference_image()
    assert ref.GetSize() == (3, 3, 4)
    assert ref.GetSpacing() == (1, 1, 2)
    assert ref.GetDirection() == (1, 0, 0, 0, 1, 0, 0, 0, 1)


def test_sample_intensity_resamples_in_one_call_and_matches_source_pixels(make_image):
    pkg = SlicePackage(base_slice=base_slice(size=[5, 5]), spacing_k=1, size_k=10)
    src = make_image((5, 5, 10), lambda idx: 100 * idx[2] + 10 * idx[0] + idx[1])

    res = pkg.sample_intensity(src)
    assert list(res.columns) == ["i", "j", "k", "x", "y", "z", "intensity"]

    row = res[(res["i"] == 2) & (res["j"] == 1) & (res["k"] == 1)]
    assert row["intensity"].iloc[0] == 100 * 1 + 10 * 2 + 1


def test_sample_intensity_marks_out_of_bounds_as_nan(make_image):
    small_src = make_image((5, 5, 10), lambda idx: 1)
    big_pkg = SlicePackage(base_slice=base_slice(size=[10, 10]), spacing_k=1, size_k=1)

    res = big_pkg.sample_intensity(small_src)
    assert res["intensity"].isna().any()


def test_sample_intensity_rejects_non_3d_images(make_image):
    pkg = SlicePackage(base_slice=base_slice(), spacing_k=1, size_k=2)
    img4d = make_image((2, 2, 2, 2), lambda idx: 0)
    with pytest.raises(ValueError, match="3D image"):
        pkg.sample_intensity(img4d)


def test_base_slice_getter_returns_a_clone_not_the_live_slice():
    pkg = SlicePackage(base_slice=base_slice(), spacing_k=1, size_k=2)
    pkg.sample_points  # populate cache

    clone = pkg.base_slice
    clone.origin = [99, 99, 99]

    assert np.allclose(pkg.base_slice.origin, [0, 0, 0])
    assert pkg.sample_points["x"].iloc[0] == 0  # cache untouched by external mutation


def test_set_base_slice_replaces_and_invalidates_cache():
    pkg = SlicePackage(base_slice=base_slice(), spacing_k=1, size_k=2)
    pkg.sample_points

    new_base = SliceGeometry([1, 1, 1], [1, 0, 0], [0, 1, 0], [2, 2], [4, 4])
    pkg.base_slice = new_base

    assert np.allclose(pkg.base_slice.origin, [1, 1, 1])
    assert list(pkg.size) == [4, 4, 2]
    with pytest.raises(ValueError, match="SliceGeometry"):
        pkg.base_slice = "not a slice"


def test_spacing_size_setters_update_all_3_axes_at_once():
    pkg = SlicePackage(base_slice=base_slice(), spacing_k=1, size_k=2)
    pkg.spacing = [2, 3, 4]
    assert np.allclose(pkg.spacing, [2, 3, 4])
    pkg.size = [5, 6, 7]
    assert list(pkg.size) == [5, 6, 7]
    with pytest.raises(ValueError, match="length-3"):
        pkg.spacing = [1, 1]
    with pytest.raises(ValueError, match="length-3"):
        pkg.size = [1, 1]


def test_from_extent_k_derives_spacing_k_from_a_total_thickness():
    pkg = SlicePackage.from_extent_k(base_slice=base_slice(), extent_k=10, size_k=6)
    assert pkg.spacing_k == 2
    assert pkg.size_k == 6
    with pytest.raises(ValueError, match=">= 2"):
        SlicePackage.from_extent_k(base_slice(), extent_k=10, size_k=1)
    with pytest.raises(ValueError, match="> 0"):
        SlicePackage.from_extent_k(base_slice(), extent_k=-1, size_k=6)


def test_from_center_k_treats_the_given_slice_as_the_stacks_middle():
    center = base_slice()
    pkg = SlicePackage.from_center_k(center_slice=center, spacing_k=2, size_k=5)

    middle = pkg.get_slice(2)
    assert np.allclose(middle.origin, center.origin)
    assert np.allclose(pkg.get_slice(0).origin, center.origin - 4 * center.normal)


def test_from_slices_adopts_a_list_of_prebuilt_parallel_slices():
    base = base_slice()
    slices = [base.translate(k * 2 * base.normal) for k in range(4)]

    pkg = SlicePackage.from_slices(slices)
    assert pkg.spacing_k == 2
    assert pkg.size_k == 4
    assert np.allclose(pkg.base_slice.origin, base.origin)

    with pytest.raises(ValueError, match="at least 1"):
        SlicePackage.from_slices([])
    with pytest.raises(ValueError, match="SliceGeometry"):
        SlicePackage.from_slices([base, "not a slice"])

    different_size = base.translate([0, 0, 0])
    different_size.size = [5, 5]
    with pytest.raises(ValueError, match="same size"):
        SlicePackage.from_slices([base, different_size])

    uneven = list(slices)
    uneven[2] = uneven[2].translate([0, 0, 0.5])
    with pytest.raises(ValueError, match="evenly spaced"):
        SlicePackage.from_slices(uneven)

    off_axis = list(slices)
    off_axis[1] = off_axis[1].translate([0.1, 0, 0])
    with pytest.raises(ValueError, match="aligned along the shared normal"):
        SlicePackage.from_slices(off_axis)


def test_from_slices_supports_a_single_slice_given_an_explicit_spacing_k():
    base = base_slice()

    with pytest.raises(ValueError, match="spacing_k.*must be supplied explicitly"):
        SlicePackage.from_slices([base])

    pkg = SlicePackage.from_slices([base], spacing_k=2)
    assert pkg.size_k == 1
    assert pkg.spacing_k == 2
    assert np.allclose(pkg.base_slice.origin, base.origin)

    # spacing_k is ignored (derived from the data instead) once there are >= 2 slices
    slices = [base.translate(k * 2 * base.normal) for k in range(4)]
    pkg2 = SlicePackage.from_slices(slices, spacing_k=999)
    assert pkg2.spacing_k == 2


def test_from_image_axis_spans_the_whole_image_at_its_own_resolution(make_image):
    img = make_image((6, 7, 8), lambda idx: 0, origin=(-3, -3.5, -4), spacing=(1, 1, 1))
    pkg = SlicePackage.from_image_axis(img, "axial")

    assert list(pkg.size) == [6, 7, 8]
    assert np.allclose(pkg.spacing, [1, 1, 1])
    assert np.allclose(pkg.base_slice.origin, img.GetOrigin())

    pkg_num = SlicePackage.from_image_axis(img, 3)
    assert list(pkg_num.size) == list(pkg.size)

    img4d = make_image((2, 2, 2, 2), lambda idx: 0)
    with pytest.raises(ValueError, match="3D image"):
        SlicePackage.from_image_axis(img4d, "z")


def test_str_runs_without_error():
    pkg = SlicePackage(base_slice=base_slice(), spacing_k=1, size_k=2)
    s = str(pkg)
    assert "<SlicePackage>" in s
    assert "normal:" in s
    assert "base origin:" in s
