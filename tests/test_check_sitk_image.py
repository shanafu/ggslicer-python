import pytest

from ggslicer import SliceGeometry, SlicePackage, SlicePackageSet, WriteImage_fix
from ggslicer._utils import check_sitk_image


def test_passes_a_real_sitk_image_through_silently(make_image):
    img = make_image((2, 2, 2), lambda idx: 0)
    assert check_sitk_image(img) is None  # does not raise


def test_gives_a_specific_actionable_error_for_a_file_path():
    with pytest.raises(ValueError) as excinfo:
        check_sitk_image("brain.nii.gz")
    msg = str(excinfo.value)
    assert "not a file path" in msg
    assert "brain.nii.gz" in msg
    assert "ReadImage_fix" in msg


def test_gives_a_generic_type_error_for_other_wrong_types():
    with pytest.raises(ValueError) as excinfo:
        check_sitk_image(42)
    msg = str(excinfo.value)
    assert "SimpleITK Image object" in msg
    assert "int" in msg
    assert "file path" not in msg


def test_uses_the_supplied_arg_name_in_the_message():
    with pytest.raises(ValueError, match="moving_image"):
        check_sitk_image("x.mnc", arg_name="moving_image")


# --- integration: every public function taking `image` rejects a path clearly ---

def test_public_functions_taking_image_reject_a_file_path_with_a_clear_error():
    base = SliceGeometry([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1], [3, 3])
    pkg = SlicePackage(base_slice=base, spacing_k=1, size_k=2)
    path = "brain.nii.gz"

    with pytest.raises(ValueError, match="not a file path"):
        SliceGeometry.from_image_axis(path, "z", 0)
    with pytest.raises(ValueError, match="not a file path"):
        SlicePackage.from_image_axis(path, "z")
    with pytest.raises(ValueError, match="not a file path"):
        pkg.sample_intensity(path)

    sset = SlicePackageSet({"only": pkg})
    with pytest.raises(ValueError, match="not a file path"):
        sset.sample_intensity(path)
    with pytest.raises(ValueError, match="not a file path"):
        SlicePackageSet.from_orthogonal_triplet(path, {"x": 0})

    with pytest.raises(ValueError, match="not a file path"):
        WriteImage_fix(path, "out.nii.gz")
