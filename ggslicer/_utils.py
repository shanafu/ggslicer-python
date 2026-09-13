"""Small internal helpers shared across ggslicer modules."""

import SimpleITK as sitk


def check_sitk_image(image, arg_name="image"):
    """Validate that `image` is a SimpleITK Image object.

    Gives a specifically helpful error if the caller passed a file path
    instead (an easy mistake: several functions elsewhere in this package,
    e.g. ``slice_axis()``/``ReadImage_fix()``, take a path and read it
    internally, but the functions that call this validator expect an
    already-loaded image).

    Parameters
    ----------
    image : object
        The value to check.
    arg_name : str, optional
        Name to use for `image` in the error message.

    Returns
    -------
    None
    """
    if isinstance(image, sitk.Image):
        return
    if isinstance(image, str):
        raise ValueError(
            f"{arg_name} must be a SimpleITK Image object, not a file path "
            f"(got \"{image}\"). Read the file first, e.g. "
            f"{arg_name} = ReadImage_fix(\"{image}\"), then pass that."
        )
    raise ValueError(
        f"{arg_name} must be a SimpleITK Image object (e.g. from ReadImage_fix() "
        f"or SimpleITK.ReadImage()); got type '{type(image).__name__}' instead."
    )
