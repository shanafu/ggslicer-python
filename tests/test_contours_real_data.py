from ggslicer import ReadImage_fix, slice_contours, slice_label_contours


def test_slice_label_contours_traces_a_real_structure_boundary_from_human_1s_annotation(testdata_dir):
    base = testdata_dir / "human_1"
    annotation = ReadImage_fix(str(base / "annotation.nii.gz"))

    # Label 10557 is the largest nonzero label present at z=8.0 (confirmed by direct
    # inspection: 13071 voxels in that slice, next-largest is ~1500).
    out = slice_label_contours(annotation, axis="axial", coordinate=8.0, labels=[10557])

    assert len(out) > 0
    assert set(out["label"].unique()) == {10557.0}
    assert {"package", "k", "label", "obj", "vertex", "x", "y", "z"}.issubset(out.columns)


def test_mask_fill_nan_vs_zero_produce_a_real_dramatic_difference_on_mouse_1(testdata_dir):
    base = testdata_dir / "mouse_1"
    average = ReadImage_fix(str(base / "DSURQE_40micron_average.mnc"))
    mask = ReadImage_fix(str(base / "DSURQE_40micron_mask.mnc"))

    # At a very low level (0.5), real in-mask brain tissue intensity never crosses it
    # (confirmed by direct inspection), so any contour at this level under
    # mask_fill="zero" is purely a fabricated edge at the mask boundary;
    # mask_fill="nan" should then find no contour at all at this level.
    out_zero = slice_contours(average, axis="coronal", coordinate=0, levels=[0.5], mask=mask, mask_fill="zero")
    out_nan = slice_contours(average, axis="coronal", coordinate=0, levels=[0.5], mask=mask, mask_fill="nan")

    assert len(out_zero) > 0
    assert len(out_nan) == 0


def test_slice_contours_finds_a_real_iso_intensity_boundary_inside_mouse_1s_brain_mask(testdata_dir):
    base = testdata_dir / "mouse_1"
    average = ReadImage_fix(str(base / "DSURQE_40micron_average.mnc"))
    mask = ReadImage_fix(str(base / "DSURQE_40micron_mask.mnc"))

    # A mid-range level (500, well within the average's confirmed 0-2970 intensity
    # range) should produce real contours under both mask_fill modes.
    out_zero = slice_contours(average, axis="coronal", coordinate=0, levels=[500], mask=mask, mask_fill="zero")
    out_nan = slice_contours(average, axis="coronal", coordinate=0, levels=[500], mask=mask, mask_fill="nan")

    assert len(out_zero) > 0
    assert len(out_nan) > 0
