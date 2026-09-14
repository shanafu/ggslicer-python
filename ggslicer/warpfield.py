"""Nonlinear-warp displacement-arrow overlays for a slice (or slices) --
visualizes a registration warp as arrows (geom_segment-ready) rather than a
warped grid, the way ``slice_grid()`` + ``transform_points()`` does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import SimpleITK as sitk
from plotnine import aes, arrow, geom_segment

from .geometry import SliceGeometry
from .grid import _resolve_geometry_only
from .transform import read_minc_transform


def _as_warp_transform(warp):
    """Normalize `warp` into a sitk.Transform.

    Accepts an already-built sitk.Transform (any subclass -- Affine,
    Composite, DisplacementFieldTransform, etc. -- used directly), a
    sitk.Image with 3 components per pixel (a raw displacement field, e.g.
    an ANTs ``*Warp.nii.gz`` -- confirmed directly this is how such files
    actually load via sitk.ReadImage(), as a genuine vector-pixel image,
    *not* a literal 4D scalar array -- auto-wrapped via
    sitk.DisplacementFieldTransform()), or a file path to either (".xfm"
    routed through read_minc_transform(), reusing its Grid_Transform/Linear/
    composite-block handling; any other path first tried as a vector image,
    falling back to sitk.ReadTransform() for a genuine transform-format file
    such as .tfm/.mat/.h5).
    """
    if isinstance(warp, str):
        if warp.lower().endswith(".xfm"):
            return read_minc_transform(warp)
        try:
            img = sitk.ReadImage(warp)
        except RuntimeError:
            img = None
        if img is not None:
            if img.GetNumberOfComponentsPerPixel() != 3:
                raise ValueError(
                    f'warp ("{warp}") is an image but does not have 3 components '
                    "per pixel, so it can't be a displacement field."
                )
            return sitk.DisplacementFieldTransform(img)
        return sitk.ReadTransform(warp)
    if isinstance(warp, sitk.Image):
        if warp.GetNumberOfComponentsPerPixel() != 3:
            raise ValueError("warp must be a 3-component vector image (a displacement field).")
        return sitk.DisplacementFieldTransform(warp)
    if isinstance(warp, sitk.Transform):
        return warp
    raise ValueError(
        "warp must be a SimpleITK Transform object, a 3-component vector Image "
        "(a displacement field), or a file path to either."
    )


def _slice_warp_arrows_one(slice_geom, transform, spacing, arrow_length):
    """Build the arrow overlay for a single SliceGeometry: a coarser lattice
    of anchor points (same origin/direction_i/direction_j as slice_geom, but
    spaced `spacing` apart in both directions -- mirroring slice_grid()'s
    own spacing convention exactly, so the two overlays share one "how
    coarse is reasonable" default), each displaced by `transform` and
    decomposed into in-plane (drawn) and normal (metadata-only) components.
    """
    extent = slice_geom.extent
    if spacing is None:
        spacing = float(np.min(extent)) / 10

    size = np.round(extent / spacing).astype(int) + 1

    coarse = SliceGeometry(
        origin=slice_geom.origin,
        direction_i=slice_geom.direction_i,
        direction_j=slice_geom.direction_j,
        spacing=[spacing, spacing],
        size=size,
    )
    points = coarse.sample_points

    di = slice_geom.direction_i
    dj = slice_geom.direction_j
    normal = slice_geom.normal

    starts = points[["x", "y", "z"]].to_numpy(dtype=float)
    n = len(starts)
    # SimpleITK has no vectorized/batch transform-point API (the same
    # constraint documented for transform_points()); here the point count is
    # deliberately small (a sparse arrow lattice, not a fine grid), so the
    # per-point binding-overhead cost documented in CLAUDE.md's performance
    # section is not a practical concern.
    ends_raw = np.empty((n, 3))
    for idx in range(n):
        ends_raw[idx] = transform.TransformPoint(tuple(starts[idx]))

    d = ends_raw - starts
    d_i = d @ di
    d_j = d @ dj
    d_n = d @ normal

    in_plane_mag = np.sqrt(d_i ** 2 + d_j ** 2)
    total_mag = np.sqrt(in_plane_mag ** 2 + d_n ** 2)

    if arrow_length is None:
        draw_i, draw_j = d_i, d_j
    else:
        # Direction is undefined where in-plane displacement is ~0 (pure
        # out-of-plane movement, or literally zero displacement) -- leave
        # those as a zero-length arrow rather than rescaling an arbitrary
        # direction; normal_displacement still reports the true (possibly
        # nonzero) movement at that point for optional color/alpha encoding.
        tiny = np.sqrt(np.finfo(float).eps)
        has_direction = in_plane_mag > tiny
        safe_mag = np.where(has_direction, in_plane_mag, 1.0)
        scale = np.where(has_direction, arrow_length / safe_mag, 0.0)
        draw_i, draw_j = d_i * scale, d_j * scale

    ends = starts + np.outer(draw_i, di) + np.outer(draw_j, dj)

    return pd.DataFrame({
        "i": points["i"].to_numpy(), "j": points["j"].to_numpy(),
        "x": starts[:, 0], "y": starts[:, 1], "z": starts[:, 2],
        "xend": ends[:, 0], "yend": ends[:, 1], "zend": ends[:, 2],
        "in_plane_displacement": in_plane_mag,
        "normal_displacement": d_n,
        "total_displacement": total_mag,
    })


def slice_warp_arrows(image=None, axis=None, coordinate=None, warp=None,
                       spacing=None, arrow_length=None, invert=False, geometry=None):
    """Build a per-slice overlay of nonlinear-warp displacement arrows.

    An alternative to warping a grid (`slice_grid()` + `transform_points()`)
    for visualizing a nonlinear registration warp: at a lattice of points
    across a slice, shows the local displacement as an arrow rather than the
    aggregate distortion of a warped grid line. Ready for
    `plotnine.geom_segment(aes(x="x", y="y", xend="xend", yend="yend"))`.

    `warp` is normalized to one `sitk.Transform` internally (see
    `_as_warp_transform()`), so this works uniformly whether you have a raw
    displacement-field image (e.g. an ANTs `*Warp.nii.gz`), a `.xfm` file,
    or an already-loaded transform object of any kind (affine, composite,
    displacement field) -- the displacement at each point is just
    `transform.TransformPoint(point) - point`, exactly the mechanism
    `transform_points()` already uses.

    Since plotting an arrow at every sampled voxel would be an illegible,
    overlapping mess, arrows are placed on a coarser lattice (`spacing`
    apart in both directions -- assumed equal, matching `slice_grid()`'s own
    convention and default), not at the source image's native resolution.

    Each 3D displacement vector is decomposed, using the slice's own
    orthonormal basis, into an in-plane part (`direction_i`/`direction_j`
    components -- this is what's actually drawn, so the arrow always stays
    exactly within the slice plane) and a normal (out-of-plane) part, which
    cannot be drawn as an in-plane arrow but is still reported
    (`normal_displacement`) so it can be mapped to e.g. `color`/`alpha` if a
    user wants to flag where the true 3D movement is mostly out-of-plane.

    Parameters
    ----------
    image : SimpleITK.Image or str, optional
        Required unless `geometry` is supplied.
    axis : int or str, optional
        Which image axis is out-of-plane (see `build_slice_geometry`).
        Required unless `geometry` is supplied; must not be supplied
        together with `geometry`.
    coordinate : float or array-like of float, optional
        See `build_slice_geometry`. Required unless `geometry` is supplied;
        must not be supplied together with `geometry`.
    warp : SimpleITK.Transform, SimpleITK.Image, or str
        A transform object, a 3-component vector image (a raw displacement
        field; auto-wrapped via `sitk.DisplacementFieldTransform()`), or a
        file path to either (".xfm" routed through `read_minc_transform`).
        If a raw image object is passed directly (not a path), be aware
        that constructing a `DisplacementFieldTransform()` from it
        moves/invalidates the source image object (a confirmed SimpleITK
        behavior, also noted for `read_minc_transform()`'s `Grid_Transform`
        handling) -- re-read the image if you need to use it again
        afterward.
    spacing : float, optional
        Distance between adjacent arrow anchor points, the same for both
        axes. Defaults (if None) to `min(extent) / 10`, matching
        `slice_grid()`'s own default, so an arrow overlay and a grid overlay
        default to the same visual density.
    arrow_length : float, optional
        If None (default), each arrow's drawn length is the true in-plane
        displacement at that point (world units). If given (a single
        positive number, world units), every arrow's in-plane component is
        instead rescaled to exactly this length (direction preserved) --
        useful for visually emphasizing direction when displacement
        magnitude varies a lot across the slice. The true magnitudes
        (`in_plane_displacement`/`normal_displacement`/`total_displacement`)
        are always reported regardless of this setting. Points with ~zero
        in-plane displacement (direction undefined) are left as a
        zero-length arrow even when `arrow_length` is set.
    invert : bool, optional
        If True, apply the inverse of `warp` (`GetInverse()`) -- see
        `transform_points()` for the same caveat regarding
        `DisplacementFieldTransform`/composite transforms.
    geometry : SliceGeometry, SlicePackage, or SlicePackageSet, optional
        A pre-built geometry, used instead of `axis`/`coordinate`.

    Returns
    -------
    pandas.DataFrame
        Columns `package`, `k`, `i`, `j` (lattice indices, 0-indexed, at the
        coarse `spacing` resolution -- not the source image's own voxel
        grid), `x`, `y`, `z` (arrow start, world coordinates), `xend`,
        `yend`, `zend` (arrow end, world coordinates, always exactly
        in-plane), `in_plane_displacement`, `normal_displacement` (signed),
        and `total_displacement` (the full, true 3D displacement magnitude,
        `sqrt(in_plane_displacement**2 + normal_displacement**2)`,
        regardless of `arrow_length`).
    """
    if warp is None:
        raise ValueError("warp must be supplied.")
    if spacing is not None and (not np.isscalar(spacing) or spacing <= 0):
        raise ValueError("spacing must be a single positive number (or None for the default).")
    if arrow_length is not None and (not np.isscalar(arrow_length) or arrow_length <= 0):
        raise ValueError("arrow_length must be a single positive number (or None for true displacement).")

    transform = _as_warp_transform(warp)
    if invert:
        transform = transform.GetInverse()

    package_set = _resolve_geometry_only(image, axis, coordinate, geometry)

    frames = []
    for pkg_name, pkg in package_set.packages.items():
        for k in range(pkg.size_k):
            slice_geom = pkg.get_slice(k)
            arrows = _slice_warp_arrows_one(slice_geom, transform, spacing, arrow_length)
            arrows.insert(0, "k", k)
            arrows.insert(0, "package", pkg_name)
            frames.append(arrows)

    out = pd.concat(frames, ignore_index=True)
    return out[[
        "package", "k", "i", "j", "x", "y", "z", "xend", "yend", "zend",
        "in_plane_displacement", "normal_displacement", "total_displacement",
    ]]


def slice_warp_arrows_layer(arrows_df, color="#2C7FB8", size=0.5, alpha=1,
                             arrow_length_inches=0.08, arrow_type="open"):
    """Ready-made plotnine layer for a warp-arrow overlay produced by slice_warp_arrows().

    Wraps `plotnine.geom_segment()` with an arrowhead (`plotnine.arrow()`),
    so a `slice_warp_arrows()` DataFrame can be added to a plot directly,
    without the caller having to remember the `arrow=` argument -- the same
    ready-made-layer convenience `slice_grid_layers()` provides for
    `slice_grid()`.

    Parameters
    ----------
    arrows_df : pandas.DataFrame
        As returned by `slice_warp_arrows()`.
    color, size, alpha : optional
        Style for the arrow segments.
    arrow_length_inches : float, optional
        Length of the drawn arrowhead itself (inches, via `plotnine.arrow()`'s
        `length`) -- unrelated to `slice_warp_arrows()`'s own `arrow_length`
        argument, which controls the *shaft*'s world-space length, not the
        arrowhead's on-page size.
    arrow_type : {"open", "closed"}, optional
        Passed to `plotnine.arrow()`.

    Returns
    -------
    plotnine.geoms.geom_segment
        A single layer, with an arrowhead. Add it (e.g. `plt + layer`) to a
        plot that already maps x/y via its own `aes()`.
    """
    if arrow_type not in ("open", "closed"):
        raise ValueError('arrow_type must be "open" or "closed".')
    return geom_segment(
        data=arrows_df,
        mapping=aes(x="x", y="y", xend="xend", yend="yend"),
        color=color, size=size, alpha=alpha,
        arrow=arrow(length=arrow_length_inches, type=arrow_type),
    )
