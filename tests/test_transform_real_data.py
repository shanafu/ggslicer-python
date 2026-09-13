import numpy as np
import SimpleITK as sitk

from ggslicer import ReadImage_fix, read_minc_transform


def test_read_minc_transform_on_real_mouse_5_fixtures_lands_real_anatomy_on_the_correct_label(testdata_dir):
    m1 = testdata_dir / "mouse_1"
    m5 = testdata_dir / "mouse_5"

    src_labels = ReadImage_fix(str(m1 / "DSURQE_40micron_labels.mnc"))
    target_labels = ReadImage_fix(str(m5 / "DSURQE_40micron_labels_on_CCFv3_25um.mnc"))

    # The multi-block, Grid_Transform-containing top-level file -- exactly the
    # kind sitk.ReadTransform() cannot parse correctly (confirmed separately;
    # see CLAUDE.md).
    xfm = read_minc_transform(str(m5 / "MICe_DSURQE.xfm"))
    assert isinstance(xfm, sitk.CompositeTransform)

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

    assert not mismatches, f"label(s) landed on the wrong (or no) region: {mismatches}"


def test_read_minc_transform_on_a_real_single_block_affine_xfm_matches_sitk_readtransform(testdata_dir):
    m5 = testdata_dir / "mouse_5"
    path = str(m5 / "affine" / "MICe_DSURQE_affine.xfm")

    via_parser = read_minc_transform(path)
    via_read_transform = sitk.ReadTransform(path)

    np.testing.assert_allclose(via_parser.GetParameters(), via_read_transform.GetParameters(), atol=1e-8)
