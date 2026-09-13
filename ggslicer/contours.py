"""Contour-path extraction: the polyline counterpart to api.py's point sampling.

slice_contours() / slice_label_contours() are the contour-path counterparts to
slice_image(): instead of per-voxel intensities, they return the vertices of
contour lines traced by marching squares (via the ``contourpy`` package) --
ready for a ``geom_path()``-equivalent (``plotnine.geom_path``) rather than a
filled raster.

Contouring happens in each slice's own local in-plane coordinates and is then
mapped to world coordinates via that slice's SliceGeometry, so (unlike the
legacy ggslicer R functions this is inspired by) it works for oblique planes,
not just axis-aligned ones.
"""

from __future__ import annotations

import contourpy
import numpy as np
import pandas as pd
import SimpleITK as sitk

from ._utils import check_sitk_image
from .api import _as_slice_package_set, build_slice_geometry
from .readwrite import ReadImage_fix


def _slice_intensity_matrix(values, n_i, n_j):
    """Reshape one slice's sampled intensity column (i-fastest, then j, then
    k row order, as SlicePackage.sample_points/sample_intensity always
    produce) into a (n_j, n_i) array suitable for contourpy.

    contourpy (unlike R's contourLines(), which wants dim(z) == c(length(x),
    length(y))) requires z.shape == (len(y), len(x)) -- confirmed empirically,
    not assumed. values is flat with i (-> x) varying fastest, so a plain
    C-order reshape to (n_j, n_i) lines up exactly: arr[j, i] = values[i + j*n_i].
    """
    return np.asarray(values).reshape((n_j, n_i))


def _apply_mask_fill(values, mask_values, mask_fill):
    values = np.array(values, dtype=float, copy=True)
    mask_values = np.asarray(mask_values)
    excluded = ~np.isnan(mask_values) & (mask_values < 0.5)
    values[excluded] = 0.0 if mask_fill == "zero" else np.nan
    return values


def _contour_path_to_world(path_xy, slice_geom):
    origin = slice_geom.origin
    di = slice_geom.direction_i
    dj = slice_geom.direction_j
    return origin + np.outer(path_xy[:, 0], di) + np.outer(path_xy[:, 1], dj)


def _resolve_contour_inputs(image, axis, coordinate, mask, geometry):
    has_geometry = geometry is not None
    if has_geometry and (axis is not None or coordinate is not None):
        raise ValueError("axis/coordinate must not be supplied together with geometry.")
    if not has_geometry and (axis is None or coordinate is None):
        raise ValueError("Provide either geometry, or both axis and coordinate.")

    if isinstance(image, str):
        image = ReadImage_fix(image)
    check_sitk_image(image)

    if mask is not None:
        if isinstance(mask, str):
            mask = ReadImage_fix(mask)
        check_sitk_image(mask, arg_name="mask")

    package_set = _as_slice_package_set(geometry) if has_geometry else build_slice_geometry(image, axis, coordinate)

    return image, mask, package_set


def _extract_contours(image, mask, package_set, mask_fill, levels=None, binarize_fn=None):
    rows = []

    for pkg_name, pkg in package_set.packages.items():
        sampled = pkg.sample_intensity(image, interpolator=sitk.sitkLinear)
        mask_sampled = pkg.sample_intensity(mask, interpolator=sitk.sitkNearestNeighbor) if mask is not None else None

        n_i, n_j = (int(v) for v in pkg.size[:2])
        spacing = pkg.spacing
        x_seq = np.linspace(0, (n_i - 1) * spacing[0], n_i)
        y_seq = np.linspace(0, (n_j - 1) * spacing[1], n_j)

        for k in sorted(sampled["k"].unique()):
            k = int(k)
            k_rows = sampled["k"] == k
            vals = sampled.loc[k_rows, "intensity"].to_numpy()
            if mask_sampled is not None:
                mvals = mask_sampled.loc[mask_sampled["k"] == k, "intensity"].to_numpy()
                vals = _apply_mask_fill(vals, mvals, mask_fill)

            slice_geom = pkg.get_slice(k)

            if binarize_fn is None:
                z = _slice_intensity_matrix(vals, n_i, n_j)
                cg = contourpy.contour_generator(x=x_seq, y=y_seq, z=z)
                for level in levels:
                    for obj_i, path in enumerate(cg.lines(level), start=1):
                        world = _contour_path_to_world(path, slice_geom)
                        rows.append(pd.DataFrame({
                            "package": pkg_name, "k": k, "level": level, "obj": obj_i,
                            "vertex": np.arange(1, len(path) + 1),
                            "x": world[:, 0], "y": world[:, 1], "z": world[:, 2],
                        }))
            else:
                per_label = binarize_fn(vals)
                for label, indicator in per_label.items():
                    z = _slice_intensity_matrix(indicator, n_i, n_j)
                    cg = contourpy.contour_generator(x=x_seq, y=y_seq, z=z)
                    for obj_i, path in enumerate(cg.lines(0.5), start=1):
                        world = _contour_path_to_world(path, slice_geom)
                        rows.append(pd.DataFrame({
                            "package": pkg_name, "k": k, "label": label, "obj": obj_i,
                            "vertex": np.arange(1, len(path) + 1),
                            "x": world[:, 0], "y": world[:, 1], "z": world[:, 2],
                        }))

    return rows


def _finalize_contours_df(rows, group_cols, min_vertices, empty_extra_col):
    if not rows:
        cols = ["package", "k", empty_extra_col, "obj", "vertex", "x", "y", "z"]
        return pd.DataFrame({c: [] for c in cols})

    out = pd.concat(rows, ignore_index=True)
    if min_vertices is not None:
        out = out.groupby(group_cols, group_keys=False).filter(lambda g: len(g) >= min_vertices)
    return out.sort_values(group_cols + ["vertex"]).reset_index(drop=True)


def slice_contours(image, axis=None, coordinate=None, levels=None,
                    mask=None, mask_fill="zero", min_vertices=None, geometry=None):
    """Extract iso-intensity contour paths from a slice (or slices) of an image.

    The contour-path counterpart to `slice_image()`: instead of per-voxel
    intensities, returns the vertices of iso-intensity contour lines (via
    `contourpy`, i.e. marching squares) at one or more requested `levels`,
    for one or more slices -- ready for a path-drawing geom rather than a
    filled raster. Suitable for overlaying a statistical-map threshold or
    any other iso-intensity boundary on a continuous image.

    Contouring happens in each slice's own local in-plane coordinates and is
    then mapped to world coordinates via that slice's `SliceGeometry`, so it
    works for oblique planes, not just axis-aligned ones.

    Parameters
    ----------
    image : SimpleITK.Image or str
        The image to contour, or a file path (read internally via `ReadImage_fix`).
    axis : int or str, optional
        Which image axis is out-of-plane (see `build_slice_geometry`).
        Required unless `geometry` is supplied; must not be supplied
        together with `geometry`.
    coordinate : float or array-like of float, optional
        One or more world coordinates along `axis`. Required unless
        `geometry` is supplied; must not be supplied together with `geometry`.
    levels : array-like of float
        Intensity levels to contour.
    mask : SimpleITK.Image or str, optional
        Sampled nearest-neighbor onto the same geometry and applied per
        `mask_fill` before contouring.
    mask_fill : {"zero", "nan"}, optional
        How to treat voxels excluded by `mask` (values < 0.5): "zero"
        (default) zero-fills them, which -- since 0 is a value that can
        itself be a requested contour level -- also traces the mask's own
        edge as a contour (useful for drawing a mask/brain outline); "nan"
        fills them with NaN instead, which `contourpy` skips over without
        fabricating a boundary there, better suited to continuous data (e.g.
        a signed statistical map) where 0 is itself a meaningful in-range value.
    min_vertices : int, optional
        Drop any individual contour path (a package/k/level/obj group) with
        fewer than this many vertices, to remove small/noisy paths.
    geometry : SliceGeometry, SlicePackage, or SlicePackageSet, optional
        A pre-built geometry, used instead of `axis`/`coordinate`. Must not
        be supplied together with `axis`/`coordinate`.

    Returns
    -------
    pandas.DataFrame
        Columns `package`, `k`, `level`, `obj` (contour-path ID within that
        package/slice/level), `vertex` (1-based order within the path,
        always present and pre-sorted), `x`, `y`, `z` (world coordinates).
    """
    if mask_fill not in ("zero", "nan"):
        raise ValueError('mask_fill must be "zero" or "nan".')
    if levels is None or np.size(levels) < 1:
        raise ValueError("levels must be a sequence with at least one value.")
    levels = np.atleast_1d(np.asarray(levels, dtype=float))

    image, mask, package_set = _resolve_contour_inputs(image, axis, coordinate, mask, geometry)

    rows = _extract_contours(image, mask, package_set, mask_fill, levels=levels)

    return _finalize_contours_df(rows, ["package", "k", "level", "obj"], min_vertices, "level")


def slice_label_contours(image, axis=None, coordinate=None, labels=None,
                          mask=None, mask_fill="zero", min_vertices=None, geometry=None):
    """Extract label-boundary contour paths from a slice (or slices) of a label image.

    The discrete-data counterpart to `slice_contours()`: for a label/atlas
    image, rounds the sampled values to integers, finds the unique nonzero
    labels present in each slice (optionally restricted to `labels`), and
    traces each label's own boundary separately (contouring that label's
    binary indicator at the fixed threshold 0.5) -- ready for region-outline
    annotations, e.g. atlas boundaries drawn over a separately-plotted
    anatomical image without a solid fill obscuring it.

    As with `slice_contours()`, contouring happens in local in-plane
    coordinates and is mapped to world coordinates via each slice's own
    `SliceGeometry`, so it works for oblique planes too.

    Parameters
    ----------
    image : SimpleITK.Image or str
        The label image (or a file path), containing integer-valued (or
        integer-valued-once-rounded) label data. Always sampled
        nearest-neighbor, since label data is discrete.
    axis : int or str, optional
        See `slice_contours`. Required unless `geometry` is supplied.
    coordinate : float or array-like of float, optional
        See `slice_contours`. Required unless `geometry` is supplied.
    labels : array-like of float, optional
        Restrict which label values are contoured. Defaults to every
        nonzero label value present in each slice.
    mask : SimpleITK.Image or str, optional
        Sampled nearest-neighbor onto the same geometry and applied per
        `mask_fill` before contouring. Since 0 is already the "no label"
        sentinel that's always excluded from contouring, `mask_fill` makes
        no practical difference here (unlike `slice_contours`) -- it's
        offered for interface consistency, not because the two modes
        diverge for label data.
    mask_fill : {"zero", "nan"}, optional
        See `slice_contours`.
    min_vertices : int, optional
        Drop any individual contour path (a package/k/label/obj group) with
        fewer than this many vertices.
    geometry : SliceGeometry, SlicePackage, or SlicePackageSet, optional
        A pre-built geometry, used instead of `axis`/`coordinate`.

    Returns
    -------
    pandas.DataFrame
        Columns `package`, `k`, `label`, `obj` (contour-path ID within that
        package/slice/label), `vertex` (1-based order within the path),
        `x`, `y`, `z` (world coordinates).
    """
    if mask_fill not in ("zero", "nan"):
        raise ValueError('mask_fill must be "zero" or "nan".')
    if labels is not None:
        labels = np.atleast_1d(np.asarray(labels, dtype=float))

    image, mask, package_set = _resolve_contour_inputs(image, axis, coordinate, mask, geometry)

    def binarize_fn(vals):
        vals = np.round(vals)
        present = np.unique(vals[~np.isnan(vals)])
        present = present[present != 0]
        if labels is not None:
            present = np.intersect1d(present, labels)

        out = {}
        for lv in present:
            indicator = np.zeros(len(vals))
            indicator[~np.isnan(vals) & (vals == lv)] = 1.0
            indicator[np.isnan(vals)] = np.nan
            out[float(lv)] = indicator
        return out

    rows = _extract_contours(image, mask, package_set, mask_fill, binarize_fn=binarize_fn)

    return _finalize_contours_df(rows, ["package", "k", "label", "obj"], min_vertices, "label")
