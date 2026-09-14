"""Region-name text annotations for labeled slices -- places a label's
display name at the centroid of each of its connected components, ready for
``plotnine.geom_text()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from plotnine import aes, geom_text

from .contours import slice_label_contours


def _polygon_centroid_3d(v):
    """The area-weighted centroid of a closed, planar polygon embedded in
    3D (vertices already form a closed loop -- confirmed directly that both
    grDevices::contourLines() and contourpy always return a repeated first/
    last vertex -- so no separate wraparound edge is needed here). Computed
    via a triangle fan from the vertex mean (used purely for numerical
    conditioning, not as a geometric assumption -- the result is exact
    regardless of that reference point's position, including for concave
    shapes) using 3D cross products, so it needs no local 2D (i, j)
    reparameterization -- important because contour_centroids() only ever
    sees world x/y/z coordinates, with no SliceGeometry available to recover
    a local in-plane basis from.

    Returns (x, y, z, area). Falls back to the plain vertex mean when the
    enclosed area is ~0 (a degenerate/collinear contour).
    """
    ref = v.mean(axis=0)
    a = v - ref

    edge_a = a[:-1]
    edge_b = a[1:]

    cross = np.cross(edge_a, edge_b)
    tri_area_vec = 0.5 * cross
    tri_centroid_rel = (edge_a + edge_b) / 3.0

    total_area_vec = tri_area_vec.sum(axis=0)
    total_area = np.linalg.norm(total_area_vec)

    if total_area < np.sqrt(np.finfo(float).eps):
        return ref[0], ref[1], ref[2], 0.0

    n_hat = total_area_vec / total_area
    s = tri_area_vec @ n_hat
    centroid_rel = (s[:, None] * tri_centroid_rel).sum(axis=0) / s.sum()
    centroid = ref + centroid_rel

    return centroid[0], centroid[1], centroid[2], total_area


def _label_display_name(label, label_names):
    """Look up a display name for `label` from `label_names` (a dict, keys
    matched as the integer-rounded label value), falling back to the
    label's own integer value as a string when unmapped or when
    `label_names` is None. No lookup table for label -> region name ships
    with this package or with testdata/ (confirmed directly -- real atlases
    like DSURQE keep this in a separate CSV, not in the label image itself),
    so this must come from the caller.
    """
    key = int(round(label))
    if label_names is not None and key in label_names:
        return str(label_names[key])
    return str(key)


def contour_centroids(contours_df, label_names=None, min_area=None):
    """Compute one label per connected component from a slice_label_contours() DataFrame.

    Given the output of `slice_label_contours()`, finds the centroid of
    each connected component (each package/k/label/obj group already
    distinguishes separate components sharing the same label -- a label
    present as multiple disjoint regions in one slice gets one row per
    region here, not one row per label) -- ready for
    `plotnine.geom_text(aes(x="x", y="y", label="name"))` as region-name
    annotations.

    The centroid is the true, area-weighted centroid of the enclosed
    polygon (a triangle-fan computation, done directly in world
    coordinates), not just the mean of the traced boundary's vertices --
    more robust for irregular/concave region shapes, though even this can
    occasionally fall outside a very non-convex (e.g. crescent-shaped)
    region; inspect placements before relying on them for a specific
    figure.

    Parameters
    ----------
    contours_df : pandas.DataFrame
        As returned by `slice_label_contours()` (columns `package`, `k`,
        `label`, `obj`, `vertex`, `x`, `y`, `z`).
    label_names : dict, optional
        Maps a label's integer value to its display text (e.g.
        `{187: "Hippocampus"}`). Labels not present in `label_names` (or
        when `label_names` is None) fall back to their own integer value
        as a string.
    min_area : float, optional
        Drop any connected component whose true polygon area (world-space
        units^2, e.g. mm^2) is below this value -- a more accurate size
        filter than `slice_label_contours()`'s own `min_vertices` (which
        really tracks perimeter/boundary complexity, not enclosed area).

    Returns
    -------
    pandas.DataFrame
        Columns `package`, `k`, `label`, `name` (display text), `obj`,
        `x`, `y`, `z` (centroid world coordinates), `area` (world-space
        units^2).
    """
    required_cols = ("package", "k", "label", "obj", "vertex", "x", "y", "z")
    if not all(c in contours_df.columns for c in required_cols):
        raise ValueError(
            f"contours_df must have columns {required_cols} "
            "(as returned by slice_label_contours())."
        )
    if min_area is not None and (not np.isscalar(min_area) or min_area < 0):
        raise ValueError("min_area must be a single non-negative number (or None).")

    if len(contours_df) == 0:
        return pd.DataFrame({
            "package": pd.Series(dtype="object"), "k": pd.Series(dtype="int64"),
            "label": pd.Series(dtype="float64"), "name": pd.Series(dtype="object"),
            "obj": pd.Series(dtype="int64"), "x": pd.Series(dtype="float64"),
            "y": pd.Series(dtype="float64"), "z": pd.Series(dtype="float64"),
            "area": pd.Series(dtype="float64"),
        })

    ordered = contours_df.sort_values(["package", "k", "label", "obj", "vertex"])

    rows = []
    for (pkg, k, label, obj), group in ordered.groupby(["package", "k", "label", "obj"], sort=False):
        v = group[["x", "y", "z"]].to_numpy(dtype=float)
        cx, cy, cz, area = _polygon_centroid_3d(v)
        rows.append({
            "package": pkg, "k": k, "label": label, "obj": obj,
            "x": cx, "y": cy, "z": cz, "area": area,
        })
    out = pd.DataFrame(rows)

    if min_area is not None:
        out = out[out["area"] >= min_area].reset_index(drop=True)

    out["name"] = out["label"].apply(lambda lb: _label_display_name(lb, label_names))
    return out[["package", "k", "label", "name", "obj", "x", "y", "z", "area"]]


def slice_label_annotations(image, axis=None, coordinate=None, labels=None,
                             label_names=None, mask=None, mask_fill="zero",
                             min_vertices=None, min_area=None, geometry=None):
    """Place region-name text at the centroid of each labeled region in a slice.

    A one-call convenience wrapper: runs `slice_label_contours()` then
    `contour_centroids()` on the result -- for placing region-name text
    annotations directly from an image/axis/coordinate, without first
    building the contours DataFrame yourself. If you already need
    `slice_label_contours()`'s output separately (e.g. to also draw region
    outlines via `geom_path()`), call `contour_centroids()` directly on
    that existing result instead, to avoid recomputing the same contours
    twice.

    Parameters
    ----------
    image, axis, coordinate, labels, mask, mask_fill, min_vertices, geometry
        See `slice_label_contours()`.
    label_names : dict, optional
        Maps a label's integer value to its display text; see
        `contour_centroids()`.
    min_area : float, optional
        Drop any connected component whose true polygon area is below this
        value; see `contour_centroids()`. Independent of `min_vertices`,
        which is `slice_label_contours()`'s own perimeter/vertex-count-based
        filter.

    Returns
    -------
    pandas.DataFrame
        See `contour_centroids()`.
    """
    contours_df = slice_label_contours(
        image, axis=axis, coordinate=coordinate, labels=labels,
        mask=mask, mask_fill=mask_fill, min_vertices=min_vertices, geometry=geometry,
    )
    return contour_centroids(contours_df, label_names=label_names, min_area=min_area)


def slice_label_annotations_layer(annotations_df, color="black", size=11, alpha=1, fontweight="normal"):
    """Ready-made plotnine layer for region-name annotations produced by slice_label_annotations().

    Wraps `plotnine.geom_text()` for a `slice_label_annotations()`/
    `contour_centroids()` DataFrame, so it can be added to a plot directly
    -- the same ready-made-layer convenience `slice_grid_layers()`/
    `slice_warp_arrows_layer()` provide for their own outputs.

    Parameters
    ----------
    annotations_df : pandas.DataFrame
        As returned by `slice_label_annotations()`/`contour_centroids()`.
    color, size, alpha, fontweight : optional
        Style for the text labels, passed to `plotnine.geom_text()`.

    Returns
    -------
    plotnine.geoms.geom_text
        A single layer. Add it (e.g. `plt + layer`) to a plot that already
        maps x/y via its own `aes()`.
    """
    return geom_text(
        data=annotations_df,
        mapping=aes(x="x", y="y", label="name"),
        color=color, size=size, alpha=alpha, fontweight=fontweight,
    )
