"""User-facing sampling API: turn an image (+ geometry) into a tidy DataFrame.

Three layers, each a thin wrapper over the one below:

* :func:`sample_images` -- the primitive: given a geometry (any of
  ``SliceGeometry``/``SlicePackage``/``SlicePackageSet``) and one or more
  named images, returns one tidy DataFrame with one column per image.
* :func:`build_slice_geometry` -- given an image, an axis, and one or more
  world coordinates along it, builds the corresponding geometry, always
  returning a ``SlicePackageSet`` regardless of whether the coordinates are
  evenly spaced.
* :func:`slice_image` -- the main entry point: image + axis + coordinate (or
  a hand-built ``geometry=`` escape hatch) + optional extra images, straight
  to a tidy DataFrame ready for ``plotnine``.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd
import SimpleITK as sitk

from ._utils import check_sitk_image
from .geometry import SliceGeometry, SlicePackage, SlicePackageSet
from .io import ReadImage_fix

_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


def _as_slice_package_set(x):
    """Coerce a single item to a SlicePackageSet: pass a SlicePackageSet
    through unchanged; wrap a bare SlicePackage or SliceGeometry as a
    single-package set (auto-named "package_1").
    """
    if isinstance(x, SlicePackageSet):
        return x
    if isinstance(x, (SlicePackage, SliceGeometry)):
        return SlicePackageSet.from_slice_packages(x)
    raise ValueError("geometry must be a SliceGeometry, SlicePackage, or SlicePackageSet object.")


def discrete_data_names():
    """The default set of image names treated as discrete (categorical) data.

    Used by :func:`sample_images`/:func:`slice_image` to decide which images
    (masks, labels, atlases) always sample with nearest-neighbor
    interpolation, regardless of the `interpolator` requested for everything
    else. Matching is by whole token (splitting the image's name on runs of
    non-alphanumeric characters), case-insensitively -- so "brain_mask"
    matches (token "mask") but "landmasking_score" does not (no token equals
    a discrete name exactly).

    Returns
    -------
    list of str
    """
    return [
        "mask", "label", "labels", "segmentation", "segmentations", "atlas",
        "seg", "aseg", "aparc", "parcellation", "parcellations", "parcels",
        "roi", "rois", "annotation", "annotations", "regions",
    ]


def _resolve_interpolator(name, interpolator, discrete_names):
    """Nearest-neighbor if any whole token of `name` matches `discrete_names`
    (case-insensitively), otherwise `interpolator` unchanged.
    """
    tokens = [t for t in _TOKEN_SPLIT_RE.split(name.lower()) if t]
    discrete_lower = {d.lower() for d in discrete_names}
    if any(t in discrete_lower for t in tokens):
        return sitk.sitkNearestNeighbor
    return interpolator


def sample_images(geometry, images, interpolator=sitk.sitkLinear,
                   interpolator_overrides=None, discrete_names=None, extra_index=None):
    """Sample one or more SimpleITK images onto a shared slice geometry.

    Each image may be given as an already-loaded ``SimpleITK.Image`` or as a
    file path (read internally via :func:`ReadImage_fix`). The interpolator
    used for each image is `interpolator` by default, except that images
    whose name matches `discrete_names` (e.g. "mask", "label", "atlas")
    always use nearest-neighbor, and `interpolator_overrides` always wins
    over both. Images with more than 3 dimensions are indexed via
    `extra_index`; images that are exactly 3D ignore `extra_index` and have
    their (single) sampled value broadcast across every combination present
    from higher-dimensional images.

    Parameters
    ----------
    geometry : SliceGeometry, SlicePackage, or SlicePackageSet
    images : SimpleITK.Image, str, or dict
        A single image/path (sampled into an "intensity" column), or a dict
        of images/paths (sampled into columns named after the dict keys).
    interpolator : SimpleITK interpolator constant, optional
        Default interpolator (e.g. `sitk.sitkLinear`, the default, or
        `sitk.sitkNearestNeighbor`), used for any image not matched by
        `discrete_names` or `interpolator_overrides`.
    interpolator_overrides : dict, optional
        Maps an image's name to an explicit interpolator, taking precedence
        over both `interpolator` and the `discrete_names` rule.
    discrete_names : list of str, optional
        Names (matched by whole token, case-insensitively) that always use
        nearest-neighbor. Defaults to :func:`discrete_data_names`.
    extra_index : dict, optional
        One 0-indexed index sequence per non-spatial dimension shared by any
        higher-dimensional (4D/5D) image in `images` (e.g. `{"t": range(10)}`).
        Images that are exactly 3D ignore this and are sampled once,
        broadcasting across every combination.

    Returns
    -------
    pandas.DataFrame
        Columns `package`, any keys in `extra_index`, `i`, `j`, `k`, `x`,
        `y`, `z`, and one column per name in `images`.
    """
    pkgset = _as_slice_package_set(geometry)

    interpolator_overrides = dict(interpolator_overrides) if interpolator_overrides else {}
    discrete_names = discrete_names if discrete_names is not None else discrete_data_names()
    extra_index = dict(extra_index) if extra_index else {}

    if not isinstance(images, dict):
        images = {"intensity": images}

    n_extra = len(extra_index)
    base = pkgset.sample_points
    if n_extra == 0:
        out = base.copy()
    else:
        names = list(extra_index.keys())
        rows = []
        for combo_values in itertools.product(*extra_index.values()):
            chunk = base.copy()
            for nm, v in zip(names, combo_values):
                chunk[nm] = v
            rows.append(chunk)
        out = pd.concat(rows, ignore_index=True)
        ordered_cols = ["package"] + names + [c for c in out.columns if c not in ("package", *names)]
        out = out[ordered_cols]

    for name, img in images.items():
        if isinstance(img, str):
            img = ReadImage_fix(img)
        check_sitk_image(img, arg_name=f"images['{name}']")

        dim = img.GetDimension()
        needed_n_extra = dim - 3
        if needed_n_extra == 0:
            img_extra_index = {}
        elif needed_n_extra == n_extra:
            img_extra_index = extra_index
        else:
            raise ValueError(
                f"Image '{name}' has dimension {dim}, which needs extra_index with "
                f"{needed_n_extra} entries, but extra_index has {n_extra}."
            )

        interp = interpolator_overrides.get(name)
        if interp is None:
            interp = _resolve_interpolator(name, interpolator, discrete_names)

        sampled = pkgset.sample_intensity(img, extra_index=img_extra_index, interpolator=interp)
        join_cols = ["package"] + list(img_extra_index.keys()) + ["i", "j", "k"]
        sampled = sampled[join_cols + ["intensity"]].rename(columns={"intensity": name})

        out = out.merge(sampled, on=join_cols, how="left")

    return out


def build_slice_geometry(image, axis, coordinate):
    """Build a slice geometry from an image and world coordinates along one axis.

    Given an image, an out-of-plane axis, and one or more world coordinates
    along that axis, builds the corresponding axis-aligned slice(s) (via
    `SliceGeometry.from_image_axis()`) and always returns a
    `SlicePackageSet`, for a uniform return type regardless of input:

    * A single `coordinate` gives a `SlicePackageSet` with one single-slice package.
    * Multiple, evenly-spaced coordinates give a `SlicePackageSet` with one
      regular `SlicePackage` (so the whole stack resamples in a single call).
    * Multiple, unevenly-spaced coordinates give a `SlicePackageSet` with one
      single-slice package per coordinate (since a `SlicePackage` cannot
      represent uneven spacing) -- auto-detected, not something the caller
      needs to specify.

    Parameters
    ----------
    image : SimpleITK.Image
        A 3D image.
    axis : int or str
        Which image axis is out-of-plane: 1/2/3, "x"/"y"/"z", or (RAS+)
        "sagittal"/"coronal"/"axial"/"horizontal".
    coordinate : float or array-like of float
        One or more world coordinates along `axis` (each snapped to the
        nearest voxel plane). Need not be evenly spaced.

    Returns
    -------
    SlicePackageSet
    """
    check_sitk_image(image)
    coordinate = np.atleast_1d(np.asarray(coordinate, dtype=float))
    if coordinate.size < 1:
        raise ValueError("coordinate must have at least one value.")

    slices = [SliceGeometry.from_image_axis(image, axis, float(c)) for c in coordinate]

    if len(slices) == 1:
        return SlicePackageSet.from_slice_packages(slices[0])

    try:
        pkg = SlicePackage.from_slices(slices)
    except ValueError:
        pkg = None
    if pkg is not None:
        return SlicePackageSet.from_slice_packages(pkg)
    return SlicePackageSet.from_slice_packages(slices)


def slice_image(image, axis=None, coordinate=None, extra_images=None,
                 interpolator=sitk.sitkLinear, interpolator_overrides=None,
                 discrete_names=None, extra_index=None, geometry=None):
    """Sample a slice (or slices) of an image, ready for grammar-of-graphics plotting.

    The main user-facing entry point: given an image, an axis, and one or
    more world coordinates along it (or, via the `geometry` escape hatch, a
    hand-built `SliceGeometry`/`SlicePackage`/`SlicePackageSet` for oblique
    or custom cases), returns a tidy DataFrame with one row per sample point
    and one column per image (the main `image`, named "value", plus any
    `extra_images`) -- ready to hand to `plotnine`.

    Internally, this is :func:`build_slice_geometry` (unless `geometry` is
    supplied directly) followed by :func:`sample_images`; see those for the
    geometry-construction and interpolator-selection details.

    Parameters
    ----------
    image : SimpleITK.Image or str
        The main image, or a file path (read internally via `ReadImage_fix`).
    axis : int or str, optional
        Which image axis is out-of-plane (see `build_slice_geometry`).
        Required unless `geometry` is supplied; must not be supplied
        together with `geometry`.
    coordinate : float or array-like of float, optional
        One or more world coordinates along `axis` (see
        `build_slice_geometry`). Required unless `geometry` is supplied;
        must not be supplied together with `geometry`.
    extra_images : dict, optional
        Additional images/paths to sample onto the same geometry (e.g.
        `{"mask": "brainmask.nii.gz"}`), sampled into columns named after
        the dict keys.
    interpolator : SimpleITK interpolator constant, optional
        Default interpolator for `image` and any `extra_images` not
        otherwise matched; see `sample_images`.
    interpolator_overrides : dict, optional
        Explicit per-image interpolator overrides; see `sample_images`.
    discrete_names : list of str, optional
        Names treated as discrete (categorical) data; see `discrete_data_names`.
    extra_index : dict, optional
        Non-spatial index selection for 4D/5D images; see `sample_images`.
    geometry : SliceGeometry, SlicePackage, or SlicePackageSet, optional
        A pre-built geometry, used instead of `axis`/`coordinate` (e.g. for
        an oblique DICOM-derived geometry). Must not be supplied together
        with `axis`/`coordinate`.

    Returns
    -------
    pandas.DataFrame
        See `sample_images`.
    """
    has_geometry = geometry is not None
    if has_geometry and (axis is not None or coordinate is not None):
        raise ValueError("axis/coordinate must not be supplied together with geometry.")
    if not has_geometry and (axis is None or coordinate is None):
        raise ValueError("Provide either geometry, or both axis and coordinate.")

    if isinstance(image, str):
        image = ReadImage_fix(image)
    check_sitk_image(image)

    geom = geometry if has_geometry else build_slice_geometry(image, axis, coordinate)

    images = {"value": image}
    if extra_images:
        images.update(extra_images)

    return sample_images(
        geom, images,
        interpolator=interpolator,
        interpolator_overrides=interpolator_overrides,
        discrete_names=discrete_names,
        extra_index=extra_index,
    )
