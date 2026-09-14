"""Reading MINC (.xfm) transforms and applying any SimpleITK transform to a
tidy data frame's world coordinates.
"""

from __future__ import annotations

import os
import re

import SimpleITK as sitk
from tqdm import tqdm


def _read_transform_file(path):
    """Read a file path as a SimpleITK transform, routing MINC .xfm files
    through read_minc_transform() (sitk.ReadTransform() silently mis-parses
    any .xfm containing a Grid_Transform block or multiple concatenated
    blocks -- confirmed directly, not assumed; see CLAUDE.md) and everything
    else through the normal SimpleITK reader.
    """
    if path.lower().endswith(".xfm"):
        return read_minc_transform(path)
    return sitk.ReadTransform(path)


_TYPE_RE = re.compile(r"Transform_Type\s*=\s*(\w+)\s*;")
_LINEAR_RE = re.compile(r"Linear_Transform\s*=([^;]+);")
_GRID_RE = re.compile(r"Displacement_Volume\s*=\s*([^;]+);")


def _conjugate_minc_native_transform(t):
    """Wrap a MINC-native-space transform t as N . t . N, where N negates
    x/y (the same fixed operation orientation_correction() applies to image
    headers), so it operates correctly on points from ReadImage_fix()-
    corrected images instead of raw-MINC-native ones. N is its own inverse
    (zero translation, diag(-1,-1,1) matrix), and sitk.CompositeTransform
    applies the *last*-in-list transform first, so [N, t, N] computes
    N(t(N(point))) -- verified directly against a real registration (see
    CLAUDE.md): all 8 real anatomical labels checked land on the correct
    target label with this conjugation, and land on the wrong one without
    it.
    """
    n = sitk.AffineTransform(3)
    n.SetMatrix([-1, 0, 0, 0, -1, 0, 0, 0, 1])
    n.SetTranslation([0, 0, 0])
    return sitk.CompositeTransform([n, t, n])


def read_minc_transform(path, corrected=True):
    """Read a MINC transform (.xfm) file as a SimpleITK transform.

    Parses an MNI transform file directly (no dependency beyond what this
    package already requires): one or more ``Transform_Type = Linear;``
    blocks (a 3x4 matrix) and/or ``Transform_Type = Grid_Transform;`` blocks
    (a reference to a companion displacement-field MINC volume, resolved
    relative to `path`'s own directory), optionally concatenated in one
    file. An ``Invert_Flag`` on a ``Grid_Transform`` block is ignored --
    confirmed (round-tripping a real forward/inverse transform pair) that
    the referenced displacement volume already contains the correct-
    direction field, needing no extra sign handling.

    ``sitk.ReadTransform()`` must not be used for a .xfm file containing a
    ``Grid_Transform`` block or multiple concatenated blocks -- confirmed it
    silently returns a transform with the wrong dimensions and all-zero
    displacement rather than erroring. (A pure single-``Linear``-block .xfm
    does parse correctly via ``sitk.ReadTransform()``, but this function
    handles that case too, for a single entry point regardless of content.)

    Parameters
    ----------
    path : str
        Path to a .xfm file.
    corrected : bool, optional
        If True (the default), the parsed transform is conjugated so it
        operates correctly on points from `ReadImage_fix()`-corrected
        images -- which is what every tidy DataFrame this package produces
        (`slice_image()`, `slice_grid()`, `slice_contours()`, ...) actually
        contains. This matters because a .xfm file's own matrix/
        displacement values are defined in MINC's *native* coordinate
        convention (the same one `orientation_correction()` corrects
        images out of), not the corrected one -- confirmed directly: a real
        registration transform applied to points from correctly-oriented
        images landed on the wrong anatomical label entirely without this
        conjugation, and matched exactly with it. Set `corrected=False` to
        get the transform exactly as written in the file (its native-MINC
        form), e.g. to compare against another MINC-native tool's own
        computation, or to apply it directly to points from a plainly-read
        (not `ReadImage_fix()`-corrected) MINC image.

    Returns
    -------
    SimpleITK.Transform
        With `corrected=False`: a single `AffineTransform` or
        `DisplacementFieldTransform` if `path` has exactly one block, or a
        `CompositeTransform` (applying the blocks in file order) if it has
        more than one. With `corrected=True` (default): the same, wrapped
        in an outer `CompositeTransform` that conjugates it by a fixed
        x/y-negating `AffineTransform`.
    """
    with open(path) as f:
        text = f.read()
    if "MNI Transform File" not in text:
        raise ValueError(f"path does not look like an MNI transform file (.xfm): {path}")

    matches = list(_TYPE_RE.finditer(text))
    if not matches:
        raise ValueError(f"No Transform_Type blocks found in: {path}")

    base_dir = os.path.dirname(os.path.abspath(path))
    transforms = []

    for i, m in enumerate(matches):
        ttype = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block_text = text[start:end]

        if ttype == "Linear":
            lm = _LINEAR_RE.search(block_text)
            if lm is None:
                raise ValueError(f"Malformed Linear_Transform block in: {path}")
            nums = [float(x) for x in lm.group(1).split()]
            if len(nums) != 12:
                raise ValueError(f"Malformed Linear_Transform block (expected 12 numbers) in: {path}")
            mat = [nums[i:i + 4] for i in range(0, 12, 4)]
            t = sitk.AffineTransform(3)
            t.SetMatrix([mat[r][c] for r in range(3) for c in range(3)])
            t.SetTranslation([mat[r][3] for r in range(3)])
            transforms.append(t)
        elif ttype == "Grid_Transform":
            gm = _GRID_RE.search(block_text)
            if gm is None:
                raise ValueError(f"Malformed Grid_Transform block in: {path}")
            vol_path = gm.group(1).strip()
            if not os.path.isabs(vol_path):
                vol_path = os.path.join(base_dir, vol_path)
            # Read plainly -- never through ReadImage_fix()/orientation_correction().
            grid_img = sitk.ReadImage(vol_path, sitk.sitkVectorFloat64)
            transforms.append(sitk.DisplacementFieldTransform(grid_img))
        else:
            raise ValueError(
                f"Unsupported Transform_Type '{ttype}' in {path} "
                '(only "Linear" and "Grid_Transform" are supported).'
            )

    if len(transforms) == 1:
        result = transforms[0]
    else:
        # File blocks are meant to apply in file order [T1, T2, ...];
        # sitk.CompositeTransform applies the *last*-listed transform
        # first, so the list must be given in reverse (verified with a
        # non-commuting synthetic case; see CLAUDE.md).
        result = sitk.CompositeTransform(list(reversed(transforms)))

    if not corrected:
        return result
    return _conjugate_minc_native_transform(result)


def transform_points(df, transform, invert=False, x_col="x", y_col="y", z_col="z"):
    """Apply a SimpleITK transform to a tidy DataFrame's world coordinates.

    A small, generic utility: given any tidy DataFrame with x/y/z (world
    coordinate) columns -- `slice_grid()`'s output, `slice_contours()`'s
    output, or anything else -- replaces those columns with their positions
    after applying `transform`, leaving every other column untouched. This
    is what makes visualizing a registration warp possible: build a grid
    (or contour, or any point set) on the fixed/reference image, then
    transform its points through the registration transform.

    Transforms one point at a time (SimpleITK has no vectorized/batch
    transform-point API), so this can take a while for a large DataFrame --
    progress is reported via `tqdm`.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have `x_col`/`y_col`/`z_col` columns.
    transform : SimpleITK.Transform or str
        A transform object, or a file path (read via `read_minc_transform`
        for .xfm, or `sitk.ReadTransform` otherwise).
    invert : bool, optional
        If True, apply the inverse of `transform` (`transform.GetInverse()`).
        Works for affine/linear transforms; SimpleITK does not support
        inverting a `DisplacementFieldTransform` (or a composite containing
        one) this way and will raise clearly if asked -- load the
        separately-computed inverse-warp file instead (the standard
        registration-tool convention, and why e.g. ANTs always writes both
        `*Warp.nii.gz` and `*InverseWarp.nii.gz`).
    x_col, y_col, z_col : str, optional
        Names of the world-coordinate columns to transform.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with `x_col`/`y_col`/`z_col` replaced by their
        transformed coordinates.
    """
    if isinstance(transform, str):
        transform = _read_transform_file(transform)
    if invert:
        transform = transform.GetInverse()

    out = df.copy()
    xs = out[x_col].to_numpy(dtype=float)
    ys = out[y_col].to_numpy(dtype=float)
    zs = out[z_col].to_numpy(dtype=float)

    new_x = [0.0] * len(out)
    new_y = [0.0] * len(out)
    new_z = [0.0] * len(out)
    for i in tqdm(range(len(out))):
        p = transform.TransformPoint((xs[i], ys[i], zs[i]))
        new_x[i], new_y[i], new_z[i] = p

    out[x_col] = new_x
    out[y_col] = new_y
    out[z_col] = new_z
    return out
