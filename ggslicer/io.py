"""Reading and writing image files, with MINC orientation correction."""

import SimpleITK
from pathlib import Path

from ._utils import check_sitk_image


def orientation_correction(image):
    """Flip a MINC image's x/y orientation for display, preserving pixel data.

    MINC files, when read via SimpleITK, often have direction cosines that
    are the mirror image of what a NIfTI conversion of the same anatomy
    would show. This flips the pixel data along x/y (via
    ``SimpleITK.FlipImageFilter()``) and recomputes the origin so the image
    occupies the same physical bounding box, mirrored -- a display-
    orientation fix, not a coordinate registration between formats (it does
    not, and is not meant to, make a MINC-read image's world coordinates
    numerically match an independently-converted NIfTI file of the same
    anatomy; see the geometry-model notes in ``CLAUDE.md`` for the real-data
    investigation confirming this).

    `z` is never flipped. Used by both `ReadImage_fix()` (to correct on
    read) and `WriteImage_fix()` (to undo the correction on write, for MINC
    output).

    Parameters
    ----------
    image : SimpleITK.Image

    Returns
    -------
    SimpleITK.Image
        The flipped image, with `OriginalFileType` metadata copied over if
        present.
    """
    flip_filter = SimpleITK.FlipImageFilter()
    flip_filter.SetFlipAxes([True, True, False])  # flip x and y, not z

    flipped_image = flip_filter.Execute(image)

    if image.HasMetaDataKey("OriginalFileType"):
        flipped_image.SetMetaData("OriginalFileType", image.GetMetaData("OriginalFileType"))
    return flipped_image


def ReadImage_fix(file):
    """Read an image file, correcting MINC orientation for display.

    Reads `file` via ``SimpleITK.ReadImage()``. If `file` is a MINC file
    (.mnc/.minc), also applies `orientation_correction()` and records the
    original MINC direction matrix and file type as image metadata
    (`OriginalFileType`, `OriginalDirection`), so `WriteImage_fix()` can
    later undo the correction if writing back out to MINC or another
    format.

    Parameters
    ----------
    file : str
        Path to an image file.

    Returns
    -------
    SimpleITK.Image
    """
    image = SimpleITK.ReadImage(file)

    extension = Path(file).suffix.lower()
    is_minc = extension in [".mnc", ".minc"]

    if not is_minc:
        image.SetMetaData("OriginalFileType", "Other")
        return image

    image.SetMetaData("OriginalFileType", "MINC")

    orig_direction = image.GetDirection()
    image.SetMetaData("OriginalDirection", ",".join(map(str, orig_direction)))

    corrected_image = orientation_correction(image)
    corrected_image.SetMetaData("OriginalFileType", "MINC")
    corrected_image.SetMetaData("OriginalDirection", ",".join(map(str, orig_direction)))

    return corrected_image


def WriteImage_fix(image, output_file):
    """Write an image file, restoring MINC orientation if needed.

    Writes `image` via ``SimpleITK.WriteImage()``. If `image` was
    originally read from a MINC file (tracked via the `OriginalFileType`
    metadata `ReadImage_fix()` sets), the `orientation_correction()`
    applied on read is undone before writing: fully (re-flipping) if the
    output is also MINC, or by setting a generic RAS direction matrix if
    the output is NIfTI (`.nii` specifically). Images not originally read
    as MINC, and non-`.nii` non-MINC output formats, are written unchanged.

    Note a real, pre-existing cross-language inconsistency, not addressed
    here: the R package's ``WriteImage_fix()`` restores the image's actual
    original MINC direction matrix (from metadata) for *any* non-MINC
    output format, not just `.nii`. Left as-is pending a deliberate
    decision to reconcile the two (a behavior change, not a pure
    reorganization).

    Parameters
    ----------
    image : SimpleITK.Image
        Typically from `ReadImage_fix()`.
    output_file : str
        Output file path.

    Returns
    -------
    None
    """
    check_sitk_image(image)

    was_original_minc = (
        image.HasMetaDataKey("OriginalFileType")
        and image.GetMetaData("OriginalFileType") == "MINC"
    )

    output_extension = Path(output_file).suffix.lower()
    is_output_minc = output_extension in [".mnc", ".minc"]
    is_output_nifti = output_extension in [".nii"]

    image_to_write = image

    if was_original_minc and is_output_minc:
        # MINC -> MINC: undo the reading correction to restore proper MINC orientation.
        image_to_write = orientation_correction(image)
    elif was_original_minc and is_output_nifti:
        # MINC -> NIfTI: set a generic RAS direction matrix.
        image_to_write = SimpleITK.Image(image)
        ras_direction = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        image_to_write.SetDirection(ras_direction)

    SimpleITK.WriteImage(image_to_write, output_file)
