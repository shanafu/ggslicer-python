import numpy as np
import pytest
import SimpleITK as sitk

from ggslicer import ReadImage_fix, suggest_contour_levels, slice_image


def test_suggest_contour_levels_gives_sane_sorted_in_range_levels_on_mouse_1s_average_template(testdata_dir):
    base = testdata_dir / "mouse_1"
    average = ReadImage_fix(str(base / "DSURQE_40micron_average.mnc"))
    arr = sitk.GetArrayFromImage(average).ravel()
    lo, hi = arr.min(), arr.max()

    quantile_levels = suggest_contour_levels(average, method="quantile", n=5)
    assert len(quantile_levels) == 5
    np.testing.assert_array_equal(quantile_levels, np.sort(quantile_levels))
    assert np.all((quantile_levels >= lo) & (quantile_levels <= hi))

    trough_levels = suggest_contour_levels(average, method="troughs", min_n=1)
    assert len(trough_levels) >= 1
    np.testing.assert_array_equal(trough_levels, np.sort(trough_levels))
    assert np.all((trough_levels >= lo) & (trough_levels <= hi))


def test_suggest_contour_levels_rejects_mouse_1s_labels_and_mask(testdata_dir):
    base = testdata_dir / "mouse_1"
    average = ReadImage_fix(str(base / "DSURQE_40micron_average.mnc"))
    mask = ReadImage_fix(str(base / "DSURQE_40micron_mask.mnc"))
    labels = ReadImage_fix(str(base / "DSURQE_40micron_labels.mnc"))

    # image form: mask is caught by the data-driven check (integer-valued,
    # only 2 unique values). labels is NOT caught in image form -- a raw image
    # has no name to check, and DSURQE's atlas has 337 distinct integer labels,
    # above _looks_discrete()'s max_unique=50 heuristic threshold. This is a
    # confirmed, real limitation of the data-driven-only fallback, not a bug:
    # the name-based check (below, via a DataFrame column) is what actually
    # catches a labels image in practice.
    with pytest.raises(ValueError, match="discrete"):
        suggest_contour_levels(mask, method="quantile")
    suggest_contour_levels(labels, method="quantile")

    # column form: name-based check fires first, before any data is inspected.
    out = slice_image(
        average, axis="coronal", coordinate=0,
        extra_images={"mask": mask, "labels": labels},
    )
    with pytest.raises(ValueError, match="discrete"):
        suggest_contour_levels(out, column="mask")
    with pytest.raises(ValueError, match="discrete"):
        suggest_contour_levels(out, column="labels")
    suggest_contour_levels(out, column="value", method="quantile", n=3)
