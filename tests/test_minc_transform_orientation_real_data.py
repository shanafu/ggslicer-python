"""Regression test for a specific question: does applying a MINC (.xfm-derived)
transform to world points coming from ReadImage_fix()-corrected images suffer
from the same x/y axis-flip issue that orientation_correction() corrects for
when *reading* MINC images?

Investigated directly (not assumed) using mouse_5's real registration outputs,
which conveniently include ground truth: DSURQE_40micron_labels_on_CCFv3_25um.mnc
is mouse_1's DSURQE labels already correctly warped onto CCFv3 space by the real
MINC registration pipeline. Finding: NO axis-flip bug -- ITK's physical-point
machinery (origin/direction/spacing) is self-consistent, so a world coordinate
means the same physical location whether it came from a raw or a
ReadImage_fix()-corrected reading of a MINC file; a transform defined in terms
of physical points (an affine matrix, or a DisplacementFieldTransform built
from a *plainly* read displacement-field volume -- never pass it through
ReadImage_fix()) works correctly on either. This test guards that property with
real data: several distinct anatomical labels from mouse_1, each transformed
through mouse_5's real affine+grid transform, must land on the *same* label in
the real, independently-computed ground-truth CCFv3-space labels.
"""

import numpy as np
import SimpleITK as sitk

from ggslicer import ReadImage_fix


def test_minc_transform_applied_to_readimage_fix_points_has_no_axis_flip_bug(testdata_dir):
    m1 = testdata_dir / "mouse_1"
    m5 = testdata_dir / "mouse_5"

    # Everything here is read exactly the way ggslicer users would read it.
    src_labels = ReadImage_fix(str(m1 / "DSURQE_40micron_labels.mnc"))
    target_labels = ReadImage_fix(str(m5 / "DSURQE_40micron_labels_on_CCFv3_25um.mnc"))

    # A pure single-Linear-block .xfm parses correctly via sitk.ReadTransform()
    # (confirmed separately -- it's specifically multi-block/Grid_Transform .xfm
    # files that ReadTransform() silently mis-parses).
    linear = sitk.ReadTransform(str(m5 / "affine" / "MICe_DSURQE_affine.xfm"))

    # The grid (displacement field) volume must be read *plainly*, never through
    # ReadImage_fix() -- this is the one part of the pipeline that would actually
    # break if handled naively (see CLAUDE.md for why, and why it turns out not
    # to matter for the *lookup* itself but does matter as a documented rule).
    grid_img = sitk.ReadImage(str(m5 / "MICe_DSURQE_grid_0.mnc"), sitk.sitkVectorFloat64)
    grid = sitk.DisplacementFieldTransform(grid_img)

    # File order is [Linear, Grid_Transform] (apply Linear first, then Grid);
    # SimpleITK's CompositeTransform applies the *last*-listed transform first,
    # so the list must be given in reverse: [grid, linear].
    composite = sitk.CompositeTransform([grid, linear])

    arr = sitk.GetArrayFromImage(src_labels)  # (z, y, x), corrected convention
    values, counts = np.unique(arr, return_counts=True)
    order = np.argsort(-counts)
    nonzero_labels = [v for v in values[order] if v != 0][:8]
    assert len(nonzero_labels) == 8  # sanity: fixture actually has this many distinct labels

    mismatches = []
    for label_value in nonzero_labels:
        coords = np.argwhere(arr == label_value)
        idx_zyx = coords[len(coords) // 2]
        idx_xyz = (int(idx_zyx[2]), int(idx_zyx[1]), int(idx_zyx[0]))

        world_point = src_labels.TransformIndexToPhysicalPoint(idx_xyz)
        transformed = composite.TransformPoint(world_point)

        try:
            got = target_labels.GetPixel(target_labels.TransformPhysicalPointToIndex(transformed))
        except RuntimeError:
            got = None

        if got != float(label_value):
            mismatches.append((label_value, got))

    assert not mismatches, f"label(s) landed on the wrong (or no) region: {mismatches}"
