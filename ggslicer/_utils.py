"""Small internal helpers shared across ggslicer modules."""

import re

import numpy as np
import pandas as pd
import SimpleITK as sitk

_TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


def check_sitk_image(image, arg_name="image"):
    """Validate that `image` is a SimpleITK Image object.

    Gives a specifically helpful error if the caller passed a file path
    instead (an easy mistake: several functions elsewhere in this package,
    e.g. ``ReadImage_fix()``, take a path and read it internally, but the
    functions that call this validator expect an already-loaded image).

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


def discrete_data_names():
    """The default set of image names treated as discrete (categorical) data.

    Used by `sample_images()`/`slice_image()` to decide which images (masks,
    labels, atlases) always sample with nearest-neighbor interpolation,
    regardless of the `interpolator` requested for everything else, and by
    `suggest_contour_levels()` to reject a column that shouldn't be
    contoured. Matching is by whole token (splitting the image's name on
    runs of non-alphanumeric characters), case-insensitively -- so
    "brain_mask" matches (token "mask") but "landmasking_score" does not (no
    token equals a discrete name exactly).

    Returns
    -------
    list of str
    """
    return [
        "mask", "label", "labels", "segmentation", "segmentations", "atlas",
        "seg", "aseg", "aparc", "parcellation", "parcellations", "parcels",
        "roi", "rois", "annotation", "annotations", "regions",
    ]


def _matches_discrete_name(name, discrete_names):
    """Whether any whole token of `name` matches one of `discrete_names`
    (case-insensitively). Shared by api.py's `_resolve_interpolator()` and
    `suggest_contour_levels()`, so the token-matching rule only lives in
    one place.
    """
    tokens = [t for t in _TOKEN_SPLIT_RE.split(name.lower()) if t]
    discrete_lower = {d.lower() for d in discrete_names}
    return any(t in discrete_lower for t in tokens)


def _looks_discrete(values, integer_frac_threshold=0.99, max_unique=50):
    """Whether `values` looks like discrete/categorical data: mostly
    integer-valued and relatively few distinct values. A data-driven safety
    net for inputs with no name to check at all (a raw image) or an
    unusually-named discrete column -- used alongside, not instead of,
    `_matches_discrete_name()`. Thresholds are a reasonable starting
    heuristic, not empirically tuned.
    """
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return False
    frac_integer = np.mean(np.abs(v - np.round(v)) < 1e-6)
    n_unique = len(np.unique(v))
    return frac_integer >= integer_frac_threshold and n_unique <= max_unique


def _quantile_levels(values, n):
    """n interior quantiles of values (excludes the 0th/100th percentile,
    since the min/max aren't useful contour levels).
    """
    v = values[~np.isnan(values)]
    probs = np.linspace(0, 1, n + 2)[1:-1]
    return np.quantile(v, probs)


def _find_troughs(values, bw_method, prominence_frac=0.02):
    """Local minima of a Gaussian KDE of `values` at bandwidth `bw_method`
    (passed straight to `scipy.stats.gaussian_kde`), filtered to only those
    that are a genuine dip relative to a visibly higher point on *both*
    sides (a topographic-prominence-style filter), not just numerical
    wiggle in the density estimate -- e.g. a truly unimodal distribution's
    KDE can still show tiny spurious local minima out in its low-density
    tails, which have ~zero prominence and must not be reported as troughs.
    """
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(values, bw_method=bw_method)
    grid = np.linspace(values.min(), values.max(), 512)
    y = kde(grid)

    dy = np.diff(y)
    sgn = np.sign(dy)
    sign_diff = np.diff(sgn)
    local_min_idx = np.where(sign_diff == 2)[0] + 1
    if len(local_min_idx) == 0:
        return np.array([])
    local_max_idx = np.where(sign_diff == -2)[0] + 1
    boundary_idx = np.array([0, len(y) - 1])
    extrema_idx = np.sort(np.concatenate([local_max_idx, boundary_idx]))
    threshold = prominence_frac * y.max()

    keep = np.zeros(len(local_min_idx), dtype=bool)
    for i, idx in enumerate(local_min_idx):
        left = extrema_idx[extrema_idx < idx]
        right = extrema_idx[extrema_idx > idx]
        left_max = y[left.max()] if len(left) > 0 else y[idx]
        right_max = y[right.min()] if len(right) > 0 else y[idx]
        keep[i] = (min(left_max, right_max) - y[idx]) >= threshold

    return grid[local_min_idx[keep]]


def _trough_levels(values, min_n):
    """At least min_n local minima ("troughs") of a kernel density estimate
    of values -- natural boundaries between distinct populations/tissue
    classes. Uses one deliberately *oversmoothed*, deliberately *fixed*
    bandwidth (retrying at a narrower bandwidth whenever min_n isn't met was
    tried and confirmed empirically *not* to help -- for a genuinely
    unimodal distribution, narrowing only ever surfaces sampling noise as
    spurious troughs, never a real one), padding out any shortfall with
    quantile levels.
    """
    v = values[~np.isnan(values)]
    range_width = np.ptp(v)
    if range_width <= 0:
        return np.repeat(v[0], min_n)

    # scipy's gaussian_kde bw_method scalar is used directly as the
    # bandwidth *factor* (not a multiplier of a rule-of-thumb value the way
    # R's bw.nrd0() is), so seed it from Scott's rule (n**(-1/5) in 1D) and
    # double it for the same oversmoothing margin validated in R's version.
    scott_factor = len(v) ** (-1.0 / 5.0)
    bw = 2 * scott_factor
    troughs = _find_troughs(v, bw)

    if len(troughs) < min_n:
        extra = min_n - len(troughs)
        troughs = np.sort(np.concatenate([troughs, _quantile_levels(v, extra)]))
    return np.sort(troughs)


def suggest_contour_levels(x, column="value", method="quantile", n=5, min_n=1, discrete_names=None):
    """Suggest a set of contour levels from an image or a sampled DataFrame.

    Computes a reasonable, automatic set of intensity levels to pass as
    `slice_contours()`'s `levels` argument, instead of picking them by hand.
    Works on either a whole image (using its full voxel-intensity
    distribution) or a `slice_image()`/`sample_images()`-output DataFrame
    (using one named column) -- both reduce to one plain numeric array, and
    everything past that point is shared.

    Two methods are available. `"quantile"` (the default) returns `n`
    evenly-spaced interior quantiles -- simple, deterministic, and always
    returns exactly `n` levels regardless of the data's shape. `"troughs"`
    instead finds local minima of a kernel density estimate -- the natural
    boundaries between distinct populations in the data (e.g. tissue
    classes in an anatomical image) -- returning at least `min_n` of them;
    if fewer troughs exist than `min_n` (a unimodal distribution has none at
    all), the shortfall is padded out with quantile levels. Because this
    method depends on bandwidth selection, results are a reasonable
    heuristic, not a guaranteed-optimal set of levels -- inspect them before
    trusting them for a specific analysis claim.

    `levels` computed this way don't make sense for discrete/categorical
    data (masks, labels, atlases) -- this raises clearly if it looks like
    `x` (or the selected `column`) is one, checked two ways: by name
    (matching `discrete_data_names()`, exactly like `sample_images()`'s
    interpolator selection) and, since a raw image has no name to check, by
    inspecting the values themselves (mostly integer-valued, relatively few
    distinct values). Use `slice_label_contours()` for that data instead.

    Parameters
    ----------
    x : SimpleITK.Image, str, or pandas.DataFrame
        An image, a file path (read internally via `ReadImage_fix()`), or a
        DataFrame (e.g. from `slice_image()`).
    column : str, optional
        When `x` is a DataFrame, the column to use. Ignored otherwise.
        Default "value" matches `slice_image()`'s main-image column.
    method : {"quantile", "troughs"}, optional
    n : int, optional
        Number of levels to return, for `method="quantile"`.
    min_n : int, optional
        Minimum number of levels to return, for `method="troughs"`.
    discrete_names : list of str, optional
        Names (matched by whole token, case-insensitively) that mark
        `column` as discrete data. Defaults to `discrete_data_names()`.

    Returns
    -------
    numpy.ndarray
        Suggested contour levels.
    """
    if method not in ("quantile", "troughs"):
        raise ValueError('method must be "quantile" or "troughs".')
    discrete_names = discrete_names if discrete_names is not None else discrete_data_names()

    if isinstance(x, pd.DataFrame):
        if column not in x.columns:
            raise ValueError(f'x has no column named "{column}".')
        if _matches_discrete_name(column, discrete_names):
            raise ValueError(
                f'column ("{column}") looks like discrete/categorical data '
                "(matches a name in discrete_names); contour levels don't make sense "
                "for it. Use slice_label_contours() instead, or pass a different column."
            )
        values = x[column].to_numpy(dtype=float)
    else:
        from .io import ReadImage_fix

        if isinstance(x, str):
            x = ReadImage_fix(x)
        check_sitk_image(x)
        values = sitk.GetArrayFromImage(x).astype(float).ravel()

    if _looks_discrete(values):
        raise ValueError(
            "The sampled values look discrete/categorical (mostly integer-valued, "
            "few unique values); contour levels don't make sense for them. "
            "Use slice_label_contours() instead."
        )

    if method == "quantile":
        if not np.isscalar(n) or n < 1:
            raise ValueError("n must be a single positive integer.")
        return _quantile_levels(values, int(n))
    else:
        if not np.isscalar(min_n) or min_n < 1:
            raise ValueError("min_n must be a single positive integer.")
        return _trough_levels(values, int(min_n))
