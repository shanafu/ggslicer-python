import pytest
import SimpleITK as sitk

from ggslicer import (
    SliceGeometry,
    SlicePackageSet,
    discrete_data_names,
    sample_images,
    build_slice_geometry,
    slice_image,
)
from ggslicer.api import _resolve_interpolator


def test_discrete_data_names_includes_the_agreed_privileged_names():
    names = discrete_data_names()
    for expected in ["mask", "label", "labels", "segmentation", "segmentations", "atlas"]:
        assert expected in names


def test_resolve_interpolator_matches_whole_tokens_not_substrings():
    names = discrete_data_names()
    assert _resolve_interpolator("brain_mask", sitk.sitkLinear, names) == sitk.sitkNearestNeighbor
    assert _resolve_interpolator("aseg.mgz", sitk.sitkLinear, names) == sitk.sitkNearestNeighbor
    assert _resolve_interpolator("landmasking_score", sitk.sitkLinear, names) == sitk.sitkLinear
    assert _resolve_interpolator("tstat1", sitk.sitkLinear, names) == sitk.sitkLinear


def test_sample_images_defaults_a_single_non_dict_image_to_an_intensity_column(make_image):
    img = make_image((4, 4, 4), lambda idx: idx[0] + 10 * idx[1] + 100 * idx[2])
    geom = SliceGeometry.from_image_axis(img, "z", 1)
    out = sample_images(geom, img)
    assert "intensity" in out.columns


def test_sample_images_names_columns_after_the_given_dict_keys(make_image):
    img = make_image((4, 4, 4), lambda idx: idx[0] + 10 * idx[1] + 100 * idx[2])
    mask = make_image((4, 4, 4), lambda idx: float((idx[0] + idx[1]) % 2 == 0))
    geom = SliceGeometry.from_image_axis(img, "z", 1)
    out = sample_images(geom, {"value": img, "mask": mask})
    assert {"value", "mask"}.issubset(out.columns)
    assert "intensity" not in out.columns


def test_sample_images_accepts_a_file_path_and_reads_it_internally(make_image, tmp_path):
    img = make_image((4, 4, 4), lambda idx: idx[0] + 10 * idx[1] + 100 * idx[2])
    path = str(tmp_path / "img.nii.gz")
    sitk.WriteImage(img, path)
    geom = SliceGeometry.from_image_axis(img, "z", 1)
    out = sample_images(geom, path)
    assert "intensity" in out.columns
    assert len(out) == 16


def test_interpolator_overrides_wins_over_the_name_based_rule(make_image):
    img = make_image((4, 4, 4), lambda idx: idx[0] + 10 * idx[1] + 100 * idx[2])
    geom = SliceGeometry.from_image_axis(img, "z", 1)
    sample_images(geom, {"value": img}, interpolator_overrides={"value": sitk.sitkNearestNeighbor})


def test_sample_images_broadcasts_a_3d_images_values_across_extra_index_combinations(make_image):
    main4d = make_image((4, 4, 4, 2), lambda idx: idx[0] + 10 * idx[1] + 100 * idx[2] + 1000 * idx[3])
    mask3d = make_image((4, 4, 4), lambda idx: float((idx[0] + idx[1]) % 2 == 0))
    geom = SliceGeometry.from_image_axis(mask3d, "z", 1)
    out = sample_images(geom, {"value": main4d, "mask": mask3d}, extra_index={"t": [0, 1]})
    assert {"t", "value", "mask"}.issubset(out.columns)
    assert len(out) == 16 * 2
    grouped = out.groupby(["i", "j", "k"])["mask"]
    assert (grouped.nunique() == 1).all()


def test_sample_images_errors_clearly_when_an_images_dims_dont_match_extra_index(make_image):
    ref3d = make_image((2, 2, 2), lambda idx: sum(idx))
    main5d = make_image((2, 2, 2, 2, 2), lambda idx: sum(idx))
    geom = SliceGeometry.from_image_axis(ref3d, "z", 0)
    with pytest.raises(ValueError, match="extra_index"):
        sample_images(geom, {"value": main5d}, extra_index={"t": [0, 1]})


def test_build_slice_geometry_returns_one_package_for_evenly_spaced_coordinates(make_image):
    img = make_image((4, 4, 4), lambda idx: sum(idx))
    pset = build_slice_geometry(img, "z", [0, 1, 2])
    assert isinstance(pset, SlicePackageSet)
    assert len(pset.package_names) == 1


def test_build_slice_geometry_returns_one_package_per_slice_for_uneven_coordinates(make_image):
    img = make_image((4, 4, 4), lambda idx: sum(idx))
    pset = build_slice_geometry(img, "z", [0, 1, 3])
    assert len(pset.package_names) == 3


def test_build_slice_geometry_with_a_single_coordinate(make_image):
    img = make_image((4, 4, 4), lambda idx: sum(idx))
    pset = build_slice_geometry(img, "z", 1)
    assert len(pset.package_names) == 1


def test_slice_image_requires_exactly_one_of_geometry_or_axis_and_coordinate(make_image):
    img = make_image((4, 4, 4), lambda idx: sum(idx))
    geom = SliceGeometry.from_image_axis(img, "z", 1)
    with pytest.raises(ValueError):
        slice_image(img, axis="z", coordinate=1, geometry=geom)
    with pytest.raises(ValueError, match="Provide either"):
        slice_image(img)
    with pytest.raises(ValueError, match="Provide either"):
        slice_image(img, axis="z")


def test_slice_image_samples_the_main_image_plus_extra_images_together(make_image):
    img = make_image((4, 4, 4), lambda idx: idx[0] + 10 * idx[1] + 100 * idx[2])
    mask = make_image((4, 4, 4), lambda idx: float((idx[0] + idx[1]) % 2 == 0))
    out = slice_image(img, axis="z", coordinate=1, extra_images={"mask": mask})
    assert {"value", "mask", "package", "x", "y", "z"}.issubset(out.columns)


def test_slice_image_accepts_the_geometry_escape_hatch(make_image):
    img = make_image((4, 4, 4), lambda idx: sum(idx))
    geom = SliceGeometry.from_image_axis(img, "z", 1)
    out = slice_image(img, geometry=geom)
    assert "value" in out.columns
