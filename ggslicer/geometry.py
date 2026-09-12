"""Geometry classes for oblique-plane sampling of N-D SimpleITK images.

This module is a Python translation of ggslicer-r's ``R/geometry.R``:
``SliceGeometry``, ``SlicePackage``, ``SlicePackageSet``. See that file (and
the R package's docs) for the full design rationale; this module mirrors its
behavior, adapted to idiomatic Python (properties instead of get_x()/set_x()
method pairs where a single value is involved; classmethods instead of the
R6 "attach a function to the generator" static-constructor workaround).

Two behavioral differences from the R Numpy/array conventions are worth
noting up front, since they are easy to get backwards:

* SimpleITK's Python ``GetArrayFromImage()`` returns an array shaped
  ``(size_z, size_y, size_x)`` -- reversed relative to ``GetSize()`` -- unlike
  R's ``as.array()``, which preserves ``(i, j, k)`` order directly. This
  module's sample-point ordering (``i`` fastest, then ``j``, then ``k``)
  is deliberately chosen to match the *default* (``'C'``, row-major) flatten
  of that reversed array, so ``GetArrayFromImage(img).ravel()`` lines up
  element-for-element with ``sample_points`` with no transpose needed.
* ``np.nan`` doubles as both "not a number" and pandas' missing-value marker
  for float columns, so (unlike the R port) no NaN-to-NA conversion step is
  needed after resampling with a NaN default pixel value.
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd
import SimpleITK as sitk

_ORTHONORMAL_TOL = 1e-6


def _is_scalar(x):
    return np.ndim(x) == 0


def _validate_direction(direction_i, direction_j, tol=_ORTHONORMAL_TOL):
    direction_i = np.asarray(direction_i, dtype=float)
    direction_j = np.asarray(direction_j, dtype=float)
    if direction_i.shape != (3,):
        raise ValueError("direction_i must be a length-3 vector.")
    if direction_j.shape != (3,):
        raise ValueError("direction_j must be a length-3 vector.")
    if abs(np.linalg.norm(direction_i) - 1) > tol:
        raise ValueError("direction_i must be a unit vector (norm 1).")
    if abs(np.linalg.norm(direction_j) - 1) > tol:
        raise ValueError("direction_j must be a unit vector (norm 1).")
    if abs(np.dot(direction_i, direction_j)) > tol:
        raise ValueError("direction_i and direction_j must be orthogonal (dot product ~ 0).")
    return direction_i, direction_j


def _resolve_axis_index(axis):
    """Resolve an axis specification to a 0-indexed integer (0, 1, or 2).

    This is the standard axis-input convention for this package (mirroring
    ggslicer-r and this package's own legacy ``slice_axis()``): numeric
    ``1``/``2``/``3`` (1-based -- "1st/2nd/3rd axis" -- matching the legacy
    convention, *not* Python's native 0-indexing), Cartesian
    ``"x"``/``"y"``/``"z"``, or anatomical terms assuming right-anterior-
    superior (RAS+) positive orientation (``"sagittal"``/``"coronal"``/
    ``"axial"``, with ``"horizontal"`` accepted as a synonym for axial).

    The *returned* value is 0-indexed, ready for direct use as a Python/numpy
    array index. Because the numeric input convention is 1-based but the
    output is 0-based, this function is **not** idempotent -- do not pass an
    already-resolved index back into it. (The R version's resolver is
    1-indexed both in and out, so it is safely idempotent; this asymmetry is
    Python-specific.)
    """
    if isinstance(axis, numbers.Real) and not isinstance(axis, bool):
        if axis not in (1, 2, 3):
            raise ValueError("`axis` must be 1, 2, or 3 when given numerically.")
        return int(axis) - 1
    if not isinstance(axis, str):
        raise ValueError(
            "`axis` must be a single value: 1/2/3, \"x\"/\"y\"/\"z\", or an "
            "anatomical term (\"sagittal\"/\"coronal\"/\"axial\"/\"horizontal\")."
        )
    choice = axis.lower()
    if choice in ("x", "sagittal", "1"):
        return 0
    if choice in ("y", "coronal", "2"):
        return 1
    if choice in ("z", "axial", "horizontal", "3"):
        return 2
    raise ValueError(
        "Invalid `axis`; use 1/2/3, \"x\"/\"y\"/\"z\", or "
        "\"sagittal\"/\"coronal\"/\"axial\"/\"horizontal\"."
    )


class SliceGeometry:
    """A rectangular, evenly-sampled 2D slice through 3D physical space.

    Pure geometry: an origin, an orthonormal in-plane basis (direction_i,
    direction_j), per-axis step sizes (spacing), and per-axis sample counts
    (size). Sampling need not be isotropic, and the slice need not be
    parallel to any Cartesian axis. Holds no pixel/intensity data and has no
    dependency on any particular image -- analogous to how DICOM describes a
    single slice, or how a SimpleITK image stores its own
    origin/direction/spacing/size.
    """

    def __init__(self, origin, direction_i, direction_j, spacing, size):
        self._set_origin(origin)
        self._set_direction(direction_i, direction_j)
        self._set_spacing(spacing)
        self._set_size(size)
        self._sample_points_cache = None

    # ---- internal validated setters ----

    def _set_origin(self, origin):
        origin = np.asarray(origin, dtype=float)
        if origin.shape != (3,):
            raise ValueError("origin must be a length-3 vector.")
        self._origin = origin

    def _set_direction(self, direction_i, direction_j):
        di, dj = _validate_direction(direction_i, direction_j)
        self._direction = np.column_stack([di, dj])  # shape (3, 2): rows x,y,z; cols i,j

    def _set_spacing(self, spacing):
        spacing = np.asarray(spacing, dtype=float)
        if spacing.shape != (2,) or np.any(spacing <= 0):
            raise ValueError("spacing must be a length-2 vector with both entries > 0.")
        self._spacing = spacing

    def _set_size(self, size):
        size = np.asarray(size, dtype=float)
        if size.shape != (2,) or np.any(size < 1) or np.any(size != np.round(size)):
            raise ValueError("size must be an integer-valued length-2 vector with both entries >= 1.")
        self._size = size.astype(int)

    def _invalidate_cache(self):
        self._sample_points_cache = None

    # ---- read-only derived properties ----

    @property
    def direction(self):
        """The 3x2 direction matrix (columns i, j; rows x, y, z)."""
        return self._direction.copy()

    @property
    def direction_i(self):
        return self._direction[:, 0].copy()

    @property
    def direction_j(self):
        return self._direction[:, 1].copy()

    @property
    def normal(self):
        """Unit normal vector of the slice's plane (direction_i x direction_j)."""
        return np.cross(self._direction[:, 0], self._direction[:, 1])

    @property
    def plane(self):
        """The infinite plane this slice lies on, independent of its finite extent."""
        return {"point": self._origin.copy(), "normal": self.normal}

    @property
    def bounds(self):
        """World coordinates of the 4 corners of the sampling rectangle, as a DataFrame."""
        extent = self.extent
        di, dj = self._direction[:, 0], self._direction[:, 1]
        corners = np.stack([
            self._origin,
            self._origin + extent[0] * di,
            self._origin + extent[1] * dj,
            self._origin + extent[0] * di + extent[1] * dj,
        ])
        return pd.DataFrame({
            "corner": ["i0_j0", "i1_j0", "i0_j1", "i1_j1"],
            "x": corners[:, 0], "y": corners[:, 1], "z": corners[:, 2],
        })

    @property
    def sample_points(self):
        """Physical coordinates of every sample point, as a DataFrame with
        columns i, j, x, y, z. i/j are 0-indexed. Memoized; recomputed only
        after a property/method changes the geometry.
        """
        if self._sample_points_cache is not None:
            return self._sample_points_cache

        n_i, n_j = int(self._size[0]), int(self._size[1])
        i_flat = np.tile(np.arange(n_i), n_j)
        j_flat = np.repeat(np.arange(n_j), n_i)

        ij = np.column_stack([i_flat, j_flat]).astype(float)
        world = (ij * self._spacing) @ self._direction.T + self._origin

        out = pd.DataFrame({
            "i": i_flat, "j": j_flat,
            "x": world[:, 0], "y": world[:, 1], "z": world[:, 2],
        })
        self._sample_points_cache = out
        return out

    # ---- read/write properties backed by a single stored field ----

    @property
    def origin(self):
        return self._origin.copy()

    @origin.setter
    def origin(self, value):
        self._set_origin(value)
        self._invalidate_cache()

    @property
    def spacing(self):
        return self._spacing.copy()

    @spacing.setter
    def spacing(self, value):
        self._set_spacing(value)
        self._invalidate_cache()

    @property
    def size(self):
        return self._size.copy()

    @size.setter
    def size(self, value):
        self._set_size(value)
        self._invalidate_cache()

    # ---- read/write properties backed by a computation ----

    @property
    def extent(self):
        """Physical extent (width, height): spacing * (size - 1)."""
        return self._spacing * (self._size - 1)

    @extent.setter
    def extent(self, value):
        self.set_extent(value, adjust="spacing")

    @property
    def center(self):
        """World coordinates of the grid's midpoint."""
        extent = self.extent
        return (
            self._origin
            + 0.5 * extent[0] * self._direction[:, 0]
            + 0.5 * extent[1] * self._direction[:, 1]
        )

    @center.setter
    def center(self, value):
        self.set_center(value)

    # ---- explicit methods (joint args, multiple args, or extra options) ----

    def set_direction(self, direction_i, direction_j):
        """Update the in-plane direction basis (both vectors at once -- they
        are jointly constrained to stay orthonormal, so no per-axis setter
        is provided).
        """
        self._set_direction(direction_i, direction_j)
        self._invalidate_cache()

    def set_spacing_i(self, spacing_i):
        if not _is_scalar(spacing_i) or spacing_i <= 0:
            raise ValueError("spacing_i must be a single number > 0.")
        self._set_spacing([spacing_i, self._spacing[1]])
        self._invalidate_cache()

    def set_spacing_j(self, spacing_j):
        if not _is_scalar(spacing_j) or spacing_j <= 0:
            raise ValueError("spacing_j must be a single number > 0.")
        self._set_spacing([self._spacing[0], spacing_j])
        self._invalidate_cache()

    def set_size_i(self, size_i):
        if not _is_scalar(size_i) or size_i < 1 or size_i != round(size_i):
            raise ValueError("size_i must be a single integer >= 1.")
        self._set_size([size_i, self._size[1]])
        self._invalidate_cache()

    def set_size_j(self, size_j):
        if not _is_scalar(size_j) or size_j < 1 or size_j != round(size_j):
            raise ValueError("size_j must be a single integer >= 1.")
        self._set_size([self._size[0], size_j])
        self._invalidate_cache()

    def set_extent(self, extent, adjust="spacing"):
        """Set the physical extent directly, deriving either spacing
        (keeping size fixed) or size (keeping spacing fixed) to match.

        Parameters
        ----------
        extent : array-like of length 2
        adjust : {"spacing", "size"}
        """
        if adjust not in ("spacing", "size"):
            raise ValueError('adjust must be "spacing" or "size".')
        extent = np.asarray(extent, dtype=float)
        if extent.shape != (2,) or np.any(extent < 0):
            raise ValueError("extent must be a non-negative length-2 vector.")
        if adjust == "spacing":
            if np.any(self._size < 2):
                raise ValueError(
                    'Cannot derive spacing from extent when size < 2 along an axis; '
                    'use adjust="size" or set spacing directly.'
                )
            self._set_spacing(extent / (self._size - 1))
        else:
            self._set_size(np.round(extent / self._spacing) + 1)
        self._invalidate_cache()

    def set_center(self, center):
        """Reposition the rectangle so its midpoint is at `center`, keeping
        direction, spacing, and size unchanged (updates origin).
        """
        center = np.asarray(center, dtype=float)
        if center.shape != (3,):
            raise ValueError("center must be a length-3 vector.")
        extent = self.extent
        new_origin = (
            center
            - 0.5 * extent[0] * self._direction[:, 0]
            - 0.5 * extent[1] * self._direction[:, 1]
        )
        self._set_origin(new_origin)
        self._invalidate_cache()

    def translate(self, offset):
        """Return a new SliceGeometry, identical to this one but with its
        origin shifted by `offset`.
        """
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (3,):
            raise ValueError("offset must be a length-3 vector.")
        return SliceGeometry(
            origin=self._origin + offset,
            direction_i=self._direction[:, 0],
            direction_j=self._direction[:, 1],
            spacing=self._spacing,
            size=self._size,
        )

    def as_sitk_reference_image(self, spacing_k=1.0):
        """Build a pixel-less, single-voxel-thick SimpleITK reference image
        whose geometry exactly matches this slice, suitable as the
        `referenceImage` argument to `sitk.Resample()`.
        """
        direction_3x3 = np.column_stack([self._direction, self.normal])
        img = sitk.Image([int(self._size[0]), int(self._size[1]), 1], sitk.sitkFloat32)
        img.SetOrigin(tuple(float(v) for v in self._origin))
        img.SetSpacing((float(self._spacing[0]), float(self._spacing[1]), float(spacing_k)))
        img.SetDirection(tuple(direction_3x3.flatten()))  # default (C, row-major) flatten
        return img

    # ---- alternate constructors ----

    @classmethod
    def from_corners(cls, p0, p1, direction_i, spacing=None, size=None):
        """Construct a SliceGeometry from two opposite corners of the
        sampling rectangle.

        Two opposite corners alone do not uniquely determine a rectangle in
        3D: for a fixed diagonal `p1 - p0`, any orthogonal decomposition of
        that diagonal into two edge vectors gives a different, equally valid
        rectangle. Supplying `direction_i` resolves this: the diagonal is
        projected onto `direction_i` to get the extent along i, and
        `direction_j` is derived as the (automatically orthogonal)
        remainder, which also fixes the plane.
        """
        if (spacing is None) == (size is None):
            raise ValueError("Specify exactly one of `spacing` or `size`.")
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        direction_i = np.asarray(direction_i, dtype=float)
        if p0.shape != (3,) or p1.shape != (3,):
            raise ValueError("p0 and p1 must be length-3 vectors.")
        if direction_i.shape != (3,):
            raise ValueError("direction_i must be a length-3 vector.")
        if abs(np.linalg.norm(direction_i) - 1) > _ORTHONORMAL_TOL:
            raise ValueError("direction_i must be a unit vector (norm 1).")

        d = p1 - p0
        w = float(np.dot(d, direction_i))
        if w <= 0:
            raise ValueError("direction_i must point from p0 toward p1 (projected extent must be positive).")
        r = d - w * direction_i
        h = float(np.linalg.norm(r))
        if h <= _ORTHONORMAL_TOL:
            raise ValueError("p0, p1, and direction_i are collinear; no unique rectangle exists.")
        direction_j = r / h

        if size is not None:
            size = np.asarray(size, dtype=float)
            if size.shape != (2,):
                raise ValueError("size must be a length-2 vector.")
            if np.any(size < 2):
                raise ValueError(
                    "size must be >= 2 along both axes to derive spacing from the corners; "
                    "supply spacing explicitly for a single-sample axis."
                )
            spacing = np.array([w, h]) / (size - 1)
        else:
            spacing = np.asarray(spacing, dtype=float)
            if spacing.shape != (2,):
                raise ValueError("spacing must be a length-2 vector.")
            size = np.round(np.array([w, h]) / spacing) + 1

        return cls(origin=p0, direction_i=direction_i, direction_j=direction_j, spacing=spacing, size=size)

    @classmethod
    def from_center(cls, center, direction_i, direction_j, spacing, size):
        """Construct a SliceGeometry from its center point rather than its
        (i=0, j=0) corner.
        """
        s = cls(origin=np.zeros(3), direction_i=direction_i, direction_j=direction_j, spacing=spacing, size=size)
        s.set_center(center)
        return s

    @classmethod
    def from_normal(cls, origin, normal, spacing, size, direction_i=None):
        """Construct a SliceGeometry from a plane normal rather than an
        explicit in-plane basis.

        A normal alone does not fix the in-plane rotation. If `direction_i`
        is not supplied, a default seed vector (the world x-axis, or the
        y-axis if the normal is nearly parallel to x) is projected into the
        plane to pick one. If `direction_i` is supplied, it is projected the
        same way, so it need not already be exactly orthogonal to `normal`.
        """
        normal = np.asarray(normal, dtype=float)
        if normal.shape != (3,):
            raise ValueError("normal must be a length-3 vector.")
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= _ORTHONORMAL_TOL:
            raise ValueError("normal must be nonzero.")
        normal_unit = normal / normal_norm

        if direction_i is None:
            seed = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(seed, normal_unit)) > 1 - _ORTHONORMAL_TOL:
                seed = np.array([0.0, 1.0, 0.0])
        else:
            seed = np.asarray(direction_i, dtype=float)
            if seed.shape != (3,):
                raise ValueError("direction_i must be a length-3 vector.")

        proj = seed - np.dot(seed, normal_unit) * normal_unit
        proj_norm = np.linalg.norm(proj)
        if proj_norm <= _ORTHONORMAL_TOL:
            raise ValueError("direction_i is parallel to normal; supply a different seed direction.")
        di = proj / proj_norm
        dj = np.cross(normal_unit, di)

        return cls(origin=origin, direction_i=di, direction_j=dj, spacing=spacing, size=size)

    @classmethod
    def from_image_axis(cls, image, axis, coordinate):
        """Construct a SliceGeometry matching an axis-aligned slice through
        an existing 3D SimpleITK image -- the bridge to this package's
        legacy `slice_axis()` semantics, generalized to an oblique image
        direction matrix.

        The in-plane axes, their spacing, and their sample counts are taken
        directly from `image`'s own geometry; the out-of-plane position is
        snapped to the nearest voxel plane to `coordinate` along `axis`.
        """
        if image.GetDimension() != 3:
            raise ValueError("image must be a 3D image.")
        axis_index = _resolve_axis_index(axis)

        size_full = image.GetSize()
        spacing_full = image.GetSpacing()
        direction_mat = np.array(image.GetDirection()).reshape(3, 3)  # row-major, matches SetDirection
        in_plane = [a for a in range(3) if a != axis_index]

        n_along = int(size_full[axis_index])
        world_along = np.empty(n_along)
        for v in range(n_along):
            idx = [0, 0, 0]
            idx[axis_index] = v
            world_along[v] = image.TransformIndexToPhysicalPoint(tuple(idx))[axis_index]
        nearest_v = int(np.argmin(np.abs(world_along - coordinate)))
        fixed_idx = [0, 0, 0]
        fixed_idx[axis_index] = nearest_v
        origin = np.array(image.TransformIndexToPhysicalPoint(tuple(fixed_idx)))

        return cls(
            origin=origin,
            direction_i=direction_mat[:, in_plane[0]],
            direction_j=direction_mat[:, in_plane[1]],
            spacing=[spacing_full[in_plane[0]], spacing_full[in_plane[1]]],
            size=[size_full[in_plane[0]], size_full[in_plane[1]]],
        )

    @classmethod
    def from_bounds(cls, bounds, spacing=None, size=None):
        """Construct a SliceGeometry from the 4 labeled corners produced by
        `.bounds` -- the inverse of that property. Unlike `from_corners()`,
        no extra disambiguating direction is needed: 4 labeled corners fully
        determine the rectangle.

        Parameters
        ----------
        bounds : pandas.DataFrame
            Columns `corner, x, y, z`, with rows labeled "i0_j0", "i1_j0",
            and "i0_j1" (as produced by `.bounds`). If an "i1_j1" row is also
            present, it is checked for consistency with the other three.
        """
        if (spacing is None) == (size is None):
            raise ValueError("Specify exactly one of `spacing` or `size`.")
        required = ["i0_j0", "i1_j0", "i0_j1"]
        if "corner" not in bounds.columns or not all(r in set(bounds["corner"]) for r in required):
            raise ValueError(
                'bounds must contain rows labeled "i0_j0", "i1_j0", and "i0_j1" (as produced by `.bounds`).'
            )

        def corner_xyz(label):
            row = bounds.loc[bounds["corner"] == label, ["x", "y", "z"]]
            if len(row) != 1:
                raise ValueError(f'bounds must have exactly one row for corner "{label}".')
            return row.iloc[0].to_numpy(dtype=float)

        p00 = corner_xyz("i0_j0")
        p10 = corner_xyz("i1_j0")
        p01 = corner_xyz("i0_j1")

        edge_i = p10 - p00
        edge_j = p01 - p00
        extent_i = float(np.linalg.norm(edge_i))
        extent_j = float(np.linalg.norm(edge_j))
        if extent_i <= _ORTHONORMAL_TOL or extent_j <= _ORTHONORMAL_TOL:
            raise ValueError('Degenerate bounds: corners "i0_j0"/"i1_j0"/"i0_j1" must not coincide.')
        direction_i = edge_i / extent_i
        direction_j = edge_j / extent_j

        if "i1_j1" in set(bounds["corner"]):
            p11 = corner_xyz("i1_j1")
            expected_p11 = p00 + edge_i + edge_j
            if np.max(np.abs(p11 - expected_p11)) > 1e-4 * max(1, extent_i, extent_j):
                raise ValueError(
                    'Corner "i1_j1" is inconsistent with "i0_j0"/"i1_j0"/"i0_j1"; '
                    "bounds does not describe a rectangle."
                )

        if size is not None:
            size = np.asarray(size, dtype=float)
            if size.shape != (2,):
                raise ValueError("size must be a length-2 vector.")
            if np.any(size < 2):
                raise ValueError(
                    "size must be >= 2 along both axes to derive spacing from bounds; "
                    "supply spacing explicitly for a single-sample axis."
                )
            spacing = np.array([extent_i, extent_j]) / (size - 1)
        else:
            spacing = np.asarray(spacing, dtype=float)
            if spacing.shape != (2,):
                raise ValueError("spacing must be a length-2 vector.")
            size = np.round(np.array([extent_i, extent_j]) / spacing) + 1

        return cls(origin=p00, direction_i=direction_i, direction_j=direction_j, spacing=spacing, size=size)

    def __repr__(self):
        return (
            f"SliceGeometry(origin={self._origin.tolist()}, "
            f"direction_i={self.direction_i.tolist()}, direction_j={self.direction_j.tolist()}, "
            f"spacing={self._spacing.tolist()}, size={self._size.tolist()})"
        )

    def __str__(self):
        lines = [
            "<SliceGeometry>",
            f"  origin:      {', '.join(str(v) for v in np.round(self._origin, 4))}",
            f"  direction_i: {', '.join(str(v) for v in np.round(self.direction_i, 4))}",
            f"  direction_j: {', '.join(str(v) for v in np.round(self.direction_j, 4))}",
            f"  normal:      {', '.join(str(v) for v in np.round(self.normal, 4))}",
            f"  spacing:     {', '.join(str(v) for v in self._spacing)}",
            f"  size:        {', '.join(str(v) for v in self._size)}",
            f"  extent:      {', '.join(str(v) for v in np.round(self.extent, 4))}",
        ]
        return "\n".join(lines)


class SlicePackage:
    """A stack of parallel, evenly-spaced 2D slices.

    A regular rectangular-cuboid sampling volume, stacked along the shared
    normal direction of a single base SliceGeometry (the slice at k=0), at a
    fixed spacing -- exactly analogous to how a stack of 2D DICOM slices
    forms a 3D volume.
    """

    def __init__(self, base_slice, spacing_k, size_k):
        self._set_base_slice(base_slice)
        self._set_spacing_k(spacing_k)
        self._set_size_k(size_k)
        self._sample_points_cache = None

    def _set_base_slice(self, base_slice):
        if not isinstance(base_slice, SliceGeometry):
            raise ValueError("base_slice must be a SliceGeometry object.")
        self._base_slice = base_slice

    def _set_spacing_k(self, spacing_k):
        if not _is_scalar(spacing_k) or spacing_k <= 0:
            raise ValueError("spacing_k must be a single number > 0.")
        self._spacing_k = float(spacing_k)

    def _set_size_k(self, size_k):
        if not _is_scalar(size_k) or size_k < 1 or size_k != round(size_k):
            raise ValueError("size_k must be a single integer >= 1.")
        self._size_k = int(size_k)

    def _invalidate_cache(self):
        self._sample_points_cache = None

    # ---- properties ----

    @property
    def base_slice(self):
        """A copy of the SliceGeometry at k=0. A copy (not the live internal
        object) is returned so mutating it can't silently desynchronize this
        package's cached sample points; assign to `.base_slice` to actually
        change it.
        """
        return self._base_slice.translate(np.zeros(3))  # cheap, correct copy via the public API

    @base_slice.setter
    def base_slice(self, value):
        self._set_base_slice(value)
        self._invalidate_cache()

    @property
    def spacing_k(self):
        return self._spacing_k

    @spacing_k.setter
    def spacing_k(self, value):
        self._set_spacing_k(value)
        self._invalidate_cache()

    @property
    def size_k(self):
        return self._size_k

    @size_k.setter
    def size_k(self, value):
        self._set_size_k(value)
        self._invalidate_cache()

    @property
    def spacing(self):
        """Full 3D spacing, (spacing_i, spacing_j, spacing_k)."""
        return np.append(self._base_slice.spacing, self._spacing_k)

    @spacing.setter
    def spacing(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != (3,):
            raise ValueError("spacing must be a length-3 vector.")
        self._base_slice.spacing = value[:2]
        self._set_spacing_k(value[2])
        self._invalidate_cache()

    @property
    def size(self):
        """Full 3D size, (n_i, n_j, n_k)."""
        return np.append(self._base_slice.size, self._size_k)

    @size.setter
    def size(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != (3,):
            raise ValueError("size must be a length-3 vector.")
        self._base_slice.size = value[:2]
        self._set_size_k(value[2])
        self._invalidate_cache()

    @property
    def normal(self):
        """Unit normal of the base slice's plane (shared by every slice in the stack)."""
        return self._base_slice.normal

    def get_slice(self, k):
        """The k-th SliceGeometry in the stack."""
        if not _is_scalar(k) or k != round(k) or k < 0 or k >= self._size_k:
            raise ValueError("k must be a single integer in 0:(size_k - 1).")
        return self._base_slice.translate(k * self._spacing_k * self.normal)

    def __getitem__(self, k):
        return self.get_slice(k)

    @property
    def sample_points(self):
        """Physical coordinates of every sample point in the stack, as a
        DataFrame with columns i, j, k, x, y, z. Memoized.
        """
        if self._sample_points_cache is not None:
            return self._sample_points_cache

        n_i, n_j = (int(v) for v in self._base_slice.size)
        n_k = self._size_k
        i_flat = np.tile(np.arange(n_i), n_j * n_k)
        j_flat = np.tile(np.repeat(np.arange(n_j), n_i), n_k)
        k_flat = np.repeat(np.arange(n_k), n_i * n_j)

        spacing3 = np.append(self._base_slice.spacing, self._spacing_k)
        direction3 = np.column_stack([self._base_slice.direction, self.normal])

        ijk = np.column_stack([i_flat, j_flat, k_flat]).astype(float)
        world = (ijk * spacing3) @ direction3.T + self._base_slice.origin

        out = pd.DataFrame({
            "i": i_flat, "j": j_flat, "k": k_flat,
            "x": world[:, 0], "y": world[:, 1], "z": world[:, 2],
        })
        self._sample_points_cache = out
        return out

    def as_sitk_reference_image(self):
        """Build a pixel-less 3D SimpleITK reference image spanning the
        whole stack in a single geometry, suitable as the `referenceImage`
        argument to `sitk.Resample()`.
        """
        size = self.size
        spacing = self.spacing
        direction3x3 = np.column_stack([self._base_slice.direction, self.normal])

        img = sitk.Image([int(size[0]), int(size[1]), int(size[2])], sitk.sitkFloat32)
        img.SetOrigin(tuple(float(v) for v in self._base_slice.origin))
        img.SetSpacing(tuple(float(v) for v in spacing))
        img.SetDirection(tuple(direction3x3.flatten()))
        return img

    def sample_intensity(self, image, interpolator=sitk.sitkLinear):
        """Resample a 3D SimpleITK image onto this stack's geometry in a
        single `sitk.Resample()` call, returning `sample_points` with an
        added `intensity` column (`NaN` outside the source image's bounds).
        """
        if image.GetDimension() != 3:
            raise ValueError(
                "image must be a 3D image; extract a spatial sub-volume first for "
                "higher-dimensional images (this is what SlicePackageSet.sample_intensity() "
                "does automatically)."
            )
        ref = self.as_sitk_reference_image()
        resampled = sitk.Resample(image, ref, sitk.Transform(), interpolator, float("nan"))
        # Default (C, row-major) flatten of GetArrayFromImage's reversed
        # (k, j, i)-shaped array lines up with our (i-fastest) sample_points
        # order -- see the module docstring.
        intensity = sitk.GetArrayFromImage(resampled).ravel()

        out = self.sample_points.copy()
        out["intensity"] = intensity
        return out

    # ---- alternate constructors ----

    @classmethod
    def from_extent_k(cls, base_slice, extent_k, size_k):
        """Construct a SlicePackage from a total stack thickness rather than
        a per-step spacing.
        """
        if not _is_scalar(size_k) or size_k < 2 or size_k != round(size_k):
            raise ValueError(
                "size_k must be a single integer >= 2 to derive spacing_k from extent_k; "
                "use the primary constructor directly for size_k = 1."
            )
        if not _is_scalar(extent_k) or extent_k <= 0:
            raise ValueError("extent_k must be a single number > 0.")
        return cls(base_slice=base_slice, spacing_k=extent_k / (size_k - 1), size_k=size_k)

    @classmethod
    def from_center_k(cls, center_slice, spacing_k, size_k):
        """Construct a SlicePackage treating the given slice as the *middle*
        of the stack, rather than as its first (k=0) slice.
        """
        if not _is_scalar(spacing_k) or spacing_k <= 0:
            raise ValueError("spacing_k must be a single number > 0.")
        if not _is_scalar(size_k) or size_k < 1 or size_k != round(size_k):
            raise ValueError("size_k must be a single integer >= 1.")
        half_stack_extent = spacing_k * (size_k - 1) / 2
        base = center_slice.translate(-half_stack_extent * center_slice.normal)
        return cls(base_slice=base, spacing_k=spacing_k, size_k=size_k)

    @classmethod
    def from_slices(cls, slices):
        """Construct a SlicePackage by adopting a list of already-built,
        individually-defined SliceGeometry objects (e.g. one per DICOM
        slice), rather than generating a regular stack from a single base
        slice and a step size. Validates that every slice shares the same
        direction/spacing/size and that consecutive slices (in the order
        given) are evenly spaced along their shared normal.
        """
        slices = list(slices)
        if len(slices) < 2:
            raise ValueError("slices must be a list of at least 2 SliceGeometry objects.")
        if not all(isinstance(s, SliceGeometry) for s in slices):
            raise ValueError("Every element of slices must be a SliceGeometry object.")

        base = slices[0]
        ref_direction = base.direction
        ref_spacing = base.spacing
        ref_size = base.size

        for i, s in enumerate(slices[1:], start=1):
            if not np.allclose(s.direction, ref_direction):
                raise ValueError(f"All slices must share the same direction (be parallel); slice {i} does not.")
            if not np.allclose(s.spacing, ref_spacing):
                raise ValueError(f"All slices must share the same spacing; slice {i} does not.")
            if not np.array_equal(s.size, ref_size):
                raise ValueError(f"All slices must share the same size; slice {i} does not.")

        normal = base.normal
        base_origin = base.origin
        offsets = np.array([np.dot(s.origin - base_origin, normal) for s in slices])

        for i, s in enumerate(slices):
            expected_origin = base_origin + offsets[i] * normal
            if np.max(np.abs(s.origin - expected_origin)) > 1e-4:
                raise ValueError(f"Slice {i}'s origin is not aligned along the shared normal direction.")

        step_sizes = np.diff(offsets)
        spacing_k = step_sizes[0]
        if spacing_k <= 0 or not np.allclose(step_sizes, spacing_k, atol=1e-4):
            raise ValueError("slices must be evenly spaced, in order, along their shared normal direction.")

        return cls(base_slice=base, spacing_k=float(spacing_k), size_k=len(slices))

    @classmethod
    def from_image_axis(cls, image, axis):
        """Construct a SlicePackage spanning an entire 3D SimpleITK image
        along one of its axes, using the image's own resolution along that
        axis -- the SlicePackage equivalent of `SliceGeometry.from_image_axis()`.
        """
        if image.GetDimension() != 3:
            raise ValueError("image must be a 3D image.")
        axis_index = _resolve_axis_index(axis)  # 0-indexed; used only for local array indexing below

        size_full = image.GetSize()
        spacing_full = image.GetSpacing()
        # Pass the *original* `axis` through (not axis_index): SliceGeometry.from_image_axis
        # does its own resolution, and axis_index here is already 0-indexed, so re-resolving
        # it would silently pick the wrong axis.
        base = SliceGeometry.from_image_axis(image, axis, coordinate=image.GetOrigin()[axis_index])

        return cls(base_slice=base, spacing_k=spacing_full[axis_index], size_k=size_full[axis_index])

    def __repr__(self):
        return f"SlicePackage(base_slice={self._base_slice!r}, spacing_k={self._spacing_k}, size_k={self._size_k})"

    def __str__(self):
        lines = [
            "<SlicePackage>",
            f"  size:         {', '.join(str(v) for v in self.size)}",
            f"  spacing:      {', '.join(str(v) for v in self.spacing)}",
            f"  normal:       {', '.join(str(v) for v in np.round(self.normal, 4))}",
            f"  base origin:  {', '.join(str(v) for v in np.round(self._base_slice.origin, 4))}",
            f"  direction_i:  {', '.join(str(v) for v in np.round(self._base_slice.direction_i, 4))}",
            f"  direction_j:  {', '.join(str(v) for v in np.round(self._base_slice.direction_j, 4))}",
        ]
        return "\n".join(lines)


def _as_slice_package(x):
    """Coerce a single item to a SlicePackage: pass a SlicePackage through
    unchanged; wrap a bare SliceGeometry as a single-slice (size_k=1)
    package. Used so SlicePackageSet can accept either interchangeably.
    """
    if isinstance(x, SlicePackage):
        return x
    if isinstance(x, SliceGeometry):
        return SlicePackage(base_slice=x, spacing_k=1, size_k=1)
    raise ValueError("must be a SlicePackage or SliceGeometry object")


class SlicePackageSet:
    """A collection of SlicePackages, ready for combined intensity extraction.

    Aggregates multiple named SlicePackage objects (e.g. several different
    oblique orientations, or several structures of interest) into the
    single, final long-format table of points from which intensity data are
    extracted for plotting via the Grammar of Graphics. A bare SliceGeometry
    may be supplied anywhere a SlicePackage is expected and is automatically
    wrapped as a single-slice (size_k=1) package.

    Also handles images with more than 3 dimensions (time, channel, gradient
    direction, etc.): for each requested combination of non-spatial indices,
    the corresponding 3D spatial sub-volume is extracted once (and reused
    across every package that needs it) via `sitk.Extract()`, before each
    package resamples it in a single call. Images sampled through this class
    must therefore have at least 3 dimensions.
    """

    def __init__(self, packages=None):
        self._set_packages(packages if packages is not None else {})

    def _set_packages(self, packages):
        if not hasattr(packages, "items"):
            raise ValueError("packages must be a dict of SlicePackage/SliceGeometry objects.")
        if any((not isinstance(k, str)) or (not k) for k in packages):
            raise ValueError("packages must be a fully named dict (every key must be a non-empty string).")
        try:
            packages = {name: _as_slice_package(v) for name, v in packages.items()}
        except ValueError:
            raise ValueError("Every value in packages must be a SlicePackage or SliceGeometry object.")
        self._packages = packages

    @property
    def packages(self):
        return dict(self._packages)

    @packages.setter
    def packages(self, value):
        self._set_packages(value)

    @property
    def package_names(self):
        return list(self._packages.keys())

    def set_package(self, name, package):
        """Add (or replace) a package. `package` may be a SlicePackage or a
        bare SliceGeometry (automatically wrapped as a single-slice package).
        """
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a single non-empty string.")
        try:
            package = _as_slice_package(package)
        except ValueError:
            raise ValueError("package must be a SlicePackage or SliceGeometry object.")
        self._packages[name] = package

    def remove_package(self, name):
        self._packages.pop(name, None)

    def rename_package(self, old_name, new_name):
        if old_name not in self._packages:
            raise ValueError(f"No package named `{old_name}` in this set.")
        if new_name in self._packages:
            raise ValueError(f"A package named `{new_name}` already exists.")
        self._packages[new_name] = self._packages.pop(old_name)

    @property
    def sample_points(self):
        """Combined sample points across every package, as a DataFrame with
        columns package, i, j, k, x, y, z.
        """
        if not self._packages:
            return pd.DataFrame(columns=["package", "i", "j", "k", "x", "y", "z"])
        frames = []
        for name, pkg in self._packages.items():
            pts = pkg.sample_points.copy()
            pts.insert(0, "package", name)
            frames.append(pts)
        return pd.concat(frames, ignore_index=True)

    def sample_intensity(self, image, extra_index=None, interpolator=sitk.sitkLinear):
        """Sample intensities for every package -- and, for higher-
        dimensional images, every requested combination of non-spatial
        indices -- from `image`, combining everything into one long-format
        DataFrame.

        Parameters
        ----------
        image : SimpleITK.Image
            Dimension 3, 4, or 5. The first 3 dimensions are spatial; any
            further dimensions are non-spatial (e.g. time, channel) and
            addressed via `extra_index`.
        extra_index : dict of str -> sequence of int, optional
            One entry per non-spatial dimension of `image`, in dimension
            order (e.g. `{"t": range(10)}` for a 4D image with 10 time
            points). Every combination is sampled. Must be empty/None if
            `image` is 3D.
        interpolator : SimpleITK interpolator constant
            e.g. `sitk.sitkLinear` (default) or `sitk.sitkNearestNeighbor`.
        """
        extra_index = dict(extra_index) if extra_index else {}
        dim = image.GetDimension()
        if dim < 3 or dim > 5:
            raise ValueError("image must have dimension 3, 4, or 5.")
        n_extra = dim - 3
        if len(extra_index) != n_extra:
            raise ValueError(
                f"image has {n_extra} non-spatial dimension(s); extra_index must have exactly {n_extra} named entries."
            )
        if not self._packages:
            raise ValueError("No packages in this SlicePackageSet.")

        if n_extra == 0:
            combos = [{}]
        else:
            import itertools
            names = list(extra_index.keys())
            combos = [dict(zip(names, values)) for values in itertools.product(*extra_index.values())]

        subvolume_cache = {}
        results = []

        for combo in combos:
            if n_extra == 0:
                sub_image = image
            else:
                idx_values = tuple(int(combo[name]) for name in extra_index)
                if idx_values not in subvolume_cache:
                    extract_size = tuple(image.GetSize()[:3]) + (0,) * n_extra
                    extract_index = (0, 0, 0) + idx_values
                    subvolume_cache[idx_values] = sitk.Extract(image, extract_size, extract_index)
                sub_image = subvolume_cache[idx_values]

            for name, pkg in self._packages.items():
                pts = pkg.sample_intensity(sub_image, interpolator=interpolator).copy()
                pts.insert(0, "package", name)
                for offset, extra_name in enumerate(extra_index):
                    pts.insert(1 + offset, extra_name, combo[extra_name])
                results.append(pts)

        return pd.concat(results, ignore_index=True)

    @classmethod
    def from_orthogonal_triplet(cls, image, coordinates):
        """Construct a SlicePackageSet containing the classic sagittal,
        coronal, and axial single-slice planes through a reference image, at
        the given coordinates.

        Parameters
        ----------
        image : SimpleITK.Image
            A 3D image.
        coordinates : dict
            Up to 3 world coordinates, one per plane to include. Keys may be
            any of the axis synonyms accepted throughout this package
            (`1`/`2`/`3`, `"x"`/`"y"`/`"z"`,
            `"sagittal"`/`"coronal"`/`"axial"`/`"horizontal"`); any subset of
            the 3 planes may be supplied, but each plane may only be
            specified once. Resulting packages are always named "sagittal",
            "coronal", "axial", regardless of which synonym was used.
        """
        if not hasattr(coordinates, "items"):
            raise ValueError("coordinates must be a fully named mapping (e.g. a dict).")
        canonical_names = ["sagittal", "coronal", "axial"]
        seen = set()
        packages = {}
        for name, coordinate in coordinates.items():
            axis_index = _resolve_axis_index(name)  # 0-indexed; local use only
            canonical = canonical_names[axis_index]
            if canonical in seen:
                raise ValueError(
                    f"coordinates specifies the {canonical} plane more than once (via different synonyms)."
                )
            seen.add(canonical)
            # Pass the original `name`, not axis_index (see from_image_axis for why).
            packages[canonical] = SliceGeometry.from_image_axis(image, name, coordinate)
        return cls(packages)

    def __repr__(self):
        return f"SlicePackageSet(packages={self.package_names!r})"

    def __str__(self):
        lines = ["<SlicePackageSet>", f"  {len(self._packages)} package(s)"]
        if self._packages:
            name_width = max(len(n) for n in self._packages)
            for name, pkg in self._packages.items():
                lines.append(
                    f"    {name:<{name_width}}  size = {', '.join(str(v) for v in pkg.size)}"
                    f"  spacing = {', '.join(str(v) for v in pkg.spacing)}"
                )
        return "\n".join(lines)
