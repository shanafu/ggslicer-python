import numpy as np
import pytest
import SimpleITK as sitk

from ggslicer import orientation_correction, ReadImage_fix, WriteImage_fix


def _make_image(size, value=0.0, origin=None):
    img = sitk.Image(list(size), sitk.sitkFloat32)
    if origin is not None:
        img.SetOrigin(origin)
    for idx in np.ndindex(*size):
        img.SetPixel(idx, float(value))
    return img


def test_orientation_correction_negates_xy_of_origin_and_direction_leaves_z_and_pixels_untouched():
    img = _make_image((3, 3, 3), origin=(-10, -20, -30))
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    corrected = orientation_correction(img)

    assert np.allclose(corrected.GetOrigin(), (10, 20, -30))
    assert np.allclose(corrected.GetDirection(), (-1, 0, 0, 0, -1, 0, 0, 0, 1))
    assert np.array_equal(sitk.GetArrayFromImage(corrected), sitk.GetArrayFromImage(img))

    # doesn't mutate the caller's image
    assert np.allclose(img.GetOrigin(), (-10, -20, -30))


def test_orientation_correction_is_its_own_inverse():
    img = _make_image((3, 3, 3), origin=(-10, -20, -30))
    twice = orientation_correction(orientation_correction(img))
    assert np.allclose(twice.GetOrigin(), img.GetOrigin())
    assert np.allclose(twice.GetDirection(), img.GetDirection())
    assert np.array_equal(sitk.GetArrayFromImage(twice), sitk.GetArrayFromImage(img))


def test_orientation_correction_negates_a_non_identity_direction_matrix_correctly():
    img = _make_image((3, 3, 3), origin=(1, 2, 3))
    img.SetDirection((0, 1, 0, 1, 0, 0, 0, 0, 1))

    corrected = orientation_correction(img)
    assert np.allclose(corrected.GetOrigin(), (-1, -2, 3))
    assert np.allclose(corrected.GetDirection(), (0, -1, 0, -1, 0, 0, 0, 0, 1))


def test_orientation_correction_preserves_original_file_type_metadata():
    img = _make_image((2, 2, 2))
    img.SetMetaData("OriginalFileType", "MINC")
    corrected = orientation_correction(img)
    assert corrected.GetMetaData("OriginalFileType") == "MINC"


def test_read_image_fix_leaves_non_minc_file_unchanged_tagging_it_other(tmp_path):
    path = str(tmp_path / "img.nii.gz")
    sitk.WriteImage(_make_image((4, 4, 4), origin=(-10, -20, -30)), path)

    raw = sitk.ReadImage(path)
    fixed = ReadImage_fix(path)
    assert np.allclose(fixed.GetOrigin(), raw.GetOrigin())
    assert np.allclose(fixed.GetDirection(), raw.GetDirection())
    assert fixed.GetMetaData("OriginalFileType") == "Other"


def test_read_image_fix_applies_orientation_correction_to_a_minc_file(tmp_path):
    path = str(tmp_path / "img.mnc")
    sitk.WriteImage(_make_image((4, 4, 4), value=42.0, origin=(-5, -7, -9)), path)

    raw = sitk.ReadImage(path)
    fixed = ReadImage_fix(path)

    assert np.allclose(fixed.GetOrigin(), (5, 7, -9))
    assert fixed.GetMetaData("OriginalFileType") == "MINC"
    assert np.array_equal(sitk.GetArrayFromImage(fixed), sitk.GetArrayFromImage(raw))


def test_write_image_fix_round_trips_minc_to_minc_back_to_exact_original_header(tmp_path):
    in_path = str(tmp_path / "in.mnc")
    out_path = str(tmp_path / "out.mnc")
    sitk.WriteImage(_make_image((4, 4, 4), value=42.0, origin=(-5, -7, -9)), in_path)

    raw_original = sitk.ReadImage(in_path)
    fixed = ReadImage_fix(in_path)
    WriteImage_fix(fixed, out_path)

    reread = sitk.ReadImage(out_path)
    assert np.allclose(reread.GetOrigin(), raw_original.GetOrigin())
    assert np.allclose(reread.GetDirection(), raw_original.GetDirection())
    assert np.array_equal(sitk.GetArrayFromImage(reread), sitk.GetArrayFromImage(raw_original))


def test_write_image_fix_writes_minc_to_non_minc_with_corrected_header_as_is(tmp_path):
    in_path = str(tmp_path / "in.mnc")
    out_path = str(tmp_path / "out.nii.gz")
    sitk.WriteImage(_make_image((4, 4, 4), value=42.0, origin=(-5, -7, -9)), in_path)

    fixed = ReadImage_fix(in_path)
    WriteImage_fix(fixed, out_path)

    reread = sitk.ReadImage(out_path)
    assert np.allclose(reread.GetOrigin(), fixed.GetOrigin())
    assert np.allclose(reread.GetDirection(), fixed.GetDirection())


def test_write_image_fix_writes_non_originally_minc_image_unchanged_regardless_of_output_format(tmp_path):
    img = _make_image((3, 3, 3), origin=(1, 2, 3))
    img.SetMetaData("OriginalFileType", "Other")

    out_mnc = str(tmp_path / "out.mnc")
    WriteImage_fix(img, out_mnc)
    reread = sitk.ReadImage(out_mnc)
    assert np.allclose(reread.GetOrigin(), img.GetOrigin())
    assert np.allclose(reread.GetDirection(), img.GetDirection())


def test_write_image_fix_errors_on_a_non_sitk_image_input(tmp_path):
    with pytest.raises(ValueError):
        WriteImage_fix("not an image", str(tmp_path / "out.mnc"))
