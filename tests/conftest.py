import itertools

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
