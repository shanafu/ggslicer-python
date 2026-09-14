"""Regression test for a real bug and its fix: does a MINC (.xfm-derived)
transform correctly apply to world points coming from ReadImage_fix()-
corrected images?

History: an earlier version of this file concluded "no axis-flip bug" based
on this same 8-real-label round trip passing under the *old*, now-removed
FlipImageFilter-based orientation_correction(), combined with a *raw*
(unconjugated) transform built by hand from sitk.ReadTransform() +
DisplacementFieldTransform(). That conclusion was incomplete: the old
correction's images were themselves wrong (confirmed separately -- see
test_io.py and CLAUDE.md -- against real MINC/NIfTI ground truth via
mincheader/fslhd), but happened to combine with a raw transform in a way
that still produced correct results for this specific fixture's geometry --
not a general guarantee.

The real, verified picture (see CLAUDE.md for the full investigation): a
.xfm file's own matrix/displacement values are defined in MINC's *native*
coordinate convention (confirmed directly: they only reproduce real
registered anatomy when applied to points from *plainly*-read MINC images,
not ReadImage_fix()-corrected ones). ReadImage_fix()-corrected images,
however, use a different (RAS/LPS-consistent) convention. These two
conventions are related by a fixed, known operation (negate x/y, leave z),
so read_minc_transform()'s default (corrected=True) conjugates the parsed
transform by that same operation, making it correct for points from
ReadImage_fix()-corrected images -- which is what every function in this
package actually produces.
"""

import numpy as np
import SimpleITK as sitk

from ggslicer import ReadImage_fix, read_minc_transform


def _top_8_labels_and_mismatches(src_labels, target_labels, xfm):
    arr = sitk.GetArrayFromImage(src_labels)  # (z, y, x)
    values, counts = np.unique(arr, return_counts=True)
    order = np.argsort(-counts)
    nonzero_labels = [v for v in values[order] if v != 0][:8]

    mismatches = []
    for label_value in nonzero_labels:
        coords = np.argwhere(arr == label_value)
        idx_zyx = coords[len(coords) // 2]
        idx_xyz = (int(idx_zyx[2]), int(idx_zyx[1]), int(idx_zyx[0]))

        world_point = src_labels.TransformIndexToPhysicalPoint(idx_xyz)
        transformed = xfm.TransformPoint(world_point)

        try:
            got = target_labels.GetPixel(target_labels.TransformPhysicalPointToIndex(transformed))
        except RuntimeError:
            got = None

        if got != float(label_value):
            mismatches.append((label_value, got))

    return nonzero_labels, mismatches


def test_read_minc_transform_default_correctly_transforms_points_from_readimage_fix_corrected_images(testdata_dir):
    m1 = testdata_dir / "mouse_1"
    m5 = testdata_dir / "mouse_5"

    # Everything here is read exactly the way ggslicer users would read it.
    src_labels = ReadImage_fix(str(m1 / "DSURQE_40micron_labels.mnc"))
    target_labels = ReadImage_fix(str(m5 / "DSURQE_40micron_labels_on_CCFv3_25um.mnc"))

    # read_minc_transform()'s default corrected=True conjugates the whole
    # (Linear + Grid_Transform) chain at once.
    xfm = read_minc_transform(str(m5 / "MICe_DSURQE.xfm"))

    nonzero_labels, mismatches = _top_8_labels_and_mismatches(src_labels, target_labels, xfm)
    assert len(nonzero_labels) == 8  # sanity: fixture actually has this many distinct labels
    assert not mismatches, f"label(s) landed on the wrong (or no) region: {mismatches}"


def test_same_real_transform_also_works_on_plainly_read_images_with_corrected_false(testdata_dir):
    m1 = testdata_dir / "mouse_1"
    m5 = testdata_dir / "mouse_5"

    # Plain reads -- never through ReadImage_fix() -- paired with
    # corrected=False, i.e. the transform exactly as written in the file.
    src_labels = sitk.ReadImage(str(m1 / "DSURQE_40micron_labels.mnc"))
    target_labels = sitk.ReadImage(str(m5 / "DSURQE_40micron_labels_on_CCFv3_25um.mnc"))
    xfm = read_minc_transform(str(m5 / "MICe_DSURQE.xfm"), corrected=False)

    nonzero_labels, mismatches = _top_8_labels_and_mismatches(src_labels, target_labels, xfm)
    assert len(nonzero_labels) == 8
    assert not mismatches, f"label(s) landed on the wrong (or no) region: {mismatches}"


def test_mixing_corrected_and_uncorrected_conventions_gives_wrong_results(testdata_dir):
    """Confirms the conjugation is load-bearing, not a no-op: ReadImage_fix()-
    corrected images with the RAW (unconjugated) transform is a genuine
    coordinate-convention mismatch.
    """
    m1 = testdata_dir / "mouse_1"
    m5 = testdata_dir / "mouse_5"

    src_labels = ReadImage_fix(str(m1 / "DSURQE_40micron_labels.mnc"))
    target_labels = ReadImage_fix(str(m5 / "DSURQE_40micron_labels_on_CCFv3_25um.mnc"))
    xfm_raw = read_minc_transform(str(m5 / "MICe_DSURQE.xfm"), corrected=False)

    nonzero_labels, mismatches = _top_8_labels_and_mismatches(src_labels, target_labels, xfm_raw)
    assert len(nonzero_labels) == 8
    assert len(mismatches) == 8
