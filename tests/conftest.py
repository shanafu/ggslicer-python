import itertools
import os
from pathlib import Path

import pytest
import SimpleITK as sitk


def filled_sitk_image(size, value_fn, origin=None, spacing=None):
    """Build a SimpleITK image of arbitrary dimension, with pixel values
    assigned by value_fn(idx), where idx is the 0-indexed integer index
    tuple for that pixel.
    """
    img = sitk.Image(list(size), sitk.sitkFloat32)
    if origin is not None:
        img.SetOrigin(origin)
    if spacing is not None:
        img.SetSpacing(spacing)
    for idx in itertools.product(*[range(s) for s in size]):
        img.SetPixel(idx, value_fn(idx))
    return img


@pytest.fixture
def make_image():
    return filled_sitk_image


def _resolve_testdata_dir():
    """Locate the real-image testdata/ fixture directory (a sibling of every
    repo, never committed to git). Honors GGSLICER_TESTDATA_DIR if set;
    otherwise tries a few relative candidates, mirroring ggslicer-r's
    tests/testthat/helper-testdata.R.
    """
    env_dir = os.environ.get("GGSLICER_TESTDATA_DIR")
    if env_dir:
        return Path(env_dir)
    here = Path(__file__).resolve().parent
    for candidate in (here.parents[1] / "testdata", here.parents[0] / "testdata"):
        if candidate.is_dir():
            return candidate
    return here.parents[1] / "testdata"


@pytest.fixture
def testdata_dir():
    """The testdata/ directory, or skip the test if it isn't available."""
    path = _resolve_testdata_dir()
    if not path.is_dir():
        pytest.skip("testdata/ not available")
    return path
